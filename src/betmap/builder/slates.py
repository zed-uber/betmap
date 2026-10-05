"""Saved slates: sets of straights and parlays evaluated, sized, and compared together.

Slate legs point at the board (market, side, line), so prices and fair probabilities
refresh from the latest pull every time a slate is evaluated.
"""

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.builder.pricing import ParlayQuote, leg_for, price_parlay
from betmap.odds.scan import BoardEntry
from betmap.portfolio.optimize import Overlap, Risk, overlaps, risk, simulate_returns, size
from betmap.portfolio.positions import Position, open_positions
from betmap.tables import BetKind, Slate, SlateItem, SlateLeg, SlateStatus
from betmap.tracking import ledger

# --- editing -------------------------------------------------------------------------------


def _draft(slate: Slate) -> Slate:
    if slate.status != SlateStatus.DRAFT:
        raise ValueError(f"slate '{slate.name}' was already placed")
    return slate


def create_slate(session: Session, name: str) -> Slate:
    name = name.strip()
    if not name:
        raise ValueError("a slate needs a name")
    slate = Slate(name=name)
    session.add(slate)
    session.flush()
    return slate


def duplicate_slate(session: Session, slate: Slate, name: str | None = None) -> Slate:
    """A draft copy, to try a variation and compare it with the original."""
    copy = create_slate(session, name or f"{slate.name} (copy)")
    for item in slate.items:
        new = SlateItem(
            kind=item.kind,
            stake=item.stake,
            offered_price=item.offered_price,
            book=item.book,
            position=item.position,
        )
        new.legs = [
            SlateLeg(market_id=leg.market_id, side=leg.side, line=leg.line, book=leg.book,
                     price_at_add=leg.price_at_add)
            for leg in item.legs
        ]  # fmt: skip
        copy.items.append(new)
    session.flush()
    return copy


def _new_leg(entry: BoardEntry) -> SlateLeg:
    return SlateLeg(
        market_id=entry.market_id,
        side=entry.side,
        line=entry.line,
        book=entry.best.book,
        price_at_add=entry.best.price,
    )


def add_straight(session: Session, slate: Slate, entry: BoardEntry) -> SlateItem:
    _draft(slate)
    item = SlateItem(kind=BetKind.STRAIGHT, position=len(slate.items), legs=[_new_leg(entry)])
    slate.items.append(item)
    session.flush()
    return item


def add_to_parlay(
    session: Session, slate: Slate, entry: BoardEntry, item: SlateItem | None = None
) -> SlateItem:
    """Add a leg to `item` (a parlay in this slate), or start a new parlay with it."""
    _draft(slate)
    if item is None:
        item = SlateItem(kind=BetKind.PARLAY, position=len(slate.items))
        slate.items.append(item)
    elif item.slate_id != slate.id or item.kind != BetKind.PARLAY:
        raise ValueError("that's not a parlay in this slate")
    if any((leg.market_id, leg.side, leg.line) == entry.key for leg in item.legs):
        raise ValueError("that leg is already in this parlay")
    item.legs.append(_new_leg(entry))
    session.flush()
    return item


def remove_leg(session: Session, leg: SlateLeg) -> None:
    item = leg.item
    _draft(item.slate)
    item.legs.remove(leg)
    if not item.legs:
        item.slate.items.remove(item)
    session.flush()


def update_item(session: Session, item: SlateItem, **changes) -> SlateItem:
    """Set `stake` (None = suggested), `offered_price` (None = automatic), or `book`."""
    _draft(item.slate)
    for name, value in changes.items():
        if name not in ("stake", "offered_price", "book"):
            raise ValueError(f"can't set {name}")
        if name == "stake" and value is not None and value < 0:
            raise ValueError("stake can't be negative")
        if name == "offered_price" and value is not None and value <= 1:
            raise ValueError("price must be decimal odds > 1")
        setattr(item, name, value)
    session.flush()
    return item


# --- evaluating ----------------------------------------------------------------------------


@dataclass
class LegView:
    leg: SlateLeg
    entry: BoardEntry | None  # None: no longer on the board (game started, line pulled)

    @property
    def moved(self) -> bool:
        """The price at the leg's book changed since it was added."""
        if self.entry is None or self.leg.price_at_add is None:
            return False
        now = self.entry.offers.get(self.leg.book or "", self.entry.best.price)
        return abs(now - self.leg.price_at_add) > 1e-9


@dataclass
class ItemView:
    item: SlateItem
    legs: list[LegView]
    book: str | None = None
    price: float | None = None
    fair_prob: float | None = None
    quote: ParlayQuote | None = None
    suggested: float = 0.0  # bankroll fraction from joint sizing
    stake: float = 0.0  # entered, or suggested x equity
    problems: list[str] = field(default_factory=list)
    position: Position | None = None

    @property
    def ev(self) -> float | None:
        if self.price is None or self.fair_prob is None:
            return None
        return self.fair_prob * self.price - 1

    @property
    def label(self) -> str:
        parts = []
        for v in self.legs:
            line = "" if v.leg.line is None else f" {v.leg.line:g}"
            parts.append((v.entry.selection if v.entry else v.leg.side) + line)
        return " + ".join(parts)


@dataclass
class SlateView:
    slate: Slate
    items: list[ItemView]
    equity: float
    risk: Risk | None = None
    growth: float | None = None  # expected log growth of the bankroll from this slate
    overlaps: list[Overlap] = field(default_factory=list)

    @property
    def total_stake(self) -> float:
        return sum(v.stake for v in self.items)

    @property
    def problems(self) -> list[str]:
        return [f"{v.label}: {p}" for v in self.items for p in v.problems]


def _price_straight(session: Session, view: ItemView) -> None:
    """At the book it was added from while that book still offers it, else the best book."""
    entry = view.legs[0].entry
    book = view.legs[0].leg.book
    view.book = book if book in entry.offers else entry.best.book
    view.price = entry.offers[view.book]
    view.fair_prob = entry.fairs[view.book]
    view.position = Position(
        label=view.label,
        game_label=entry.event_label,
        leg=leg_for(session, entry),
        price=view.price,
        prob=view.fair_prob,
        prob_source="scan",
    )


def evaluate_slate(
    session: Session,
    slate: Slate,
    entries: list[BoardEntry],
    equity: float,
    kelly_mult: float = 0.25,
    max_bet: float = 0.03,
    max_game: float = 0.06,
) -> SlateView:
    """Price every item from the board, suggest joint stakes, and simulate the slate."""
    by_key = {e.key: e for e in entries}
    views = []
    for item in slate.items:
        view = ItemView(
            item,
            [LegView(leg, by_key.get((leg.market_id, leg.side, leg.line))) for leg in item.legs],
        )
        views.append(view)
        missing = [v for v in view.legs if v.entry is None]
        if missing:
            view.problems.append(
                f"{len(missing)} leg(s) no longer offered (game started or line pulled)"
            )
            continue
        if item.kind == BetKind.STRAIGHT:
            _price_straight(session, view)
            continue
        if len(item.legs) < 2:
            view.problems.append("add at least one more leg")
            continue
        legs = [v.entry for v in view.legs]
        quote = price_parlay(session, legs, offered=item.offered_price)
        view.quote, view.price, view.fair_prob = quote, quote.price, quote.fair_prob
        view.book = item.book or quote.book or view.legs[0].leg.book
        view.problems.extend(quote.problems)
        if view.price is not None:
            parts = [(leg_for(session, e), e.fair_prob) for e in legs]
            view.position = Position(
                label=view.label,
                game_label=" / ".join(dict.fromkeys(e.event_label for e in legs)),
                leg=parts[0][0],
                price=view.price,
                prob=quote.fair_prob,
                prob_source="scan",
                parts=parts,
            )

    priced = [v for v in views if v.position is not None]
    existing, _ = open_positions(session)
    sizing = size(
        [v.position for v in priced], existing, equity, kelly_mult, max_bet, max_game
    ) if priced and equity > 0 else []  # fmt: skip
    for v, s in zip(priced, sizing, strict=True):
        v.suggested = s.portfolio
    for v in views:
        # An entered stake is kept even when the item can't be priced, so placing it
        # reports the problem instead of silently skipping it.
        if v.item.stake is not None:
            v.stake = v.item.stake
        elif v.position is not None:
            v.stake = round(v.suggested * equity, 2)
        else:
            v.stake = 0.0
        if v.position is not None:
            v.position.stake = v.stake

    result = SlateView(slate, views, equity)
    staked = [v.position for v in priced if v.stake > 0]
    if staked:
        result.risk = risk(staked)
        if equity > 0:
            pnl = simulate_returns(staked) @ np.array([p.stake for p in staked])
            result.growth = float(np.log(np.maximum(1 + pnl / equity, 1e-9)).mean())
        mine = {id(p) for p in staked}
        result.overlaps = [
            o for o in overlaps(existing + staked) if id(o.a) in mine or id(o.b) in mine
        ]
    return result


# --- placing -------------------------------------------------------------------------------


def place_slate(session: Session, view: SlateView) -> list:
    """Log every item with a stake: straights as bets, parlays as parlays with legs."""
    slate = _draft(view.slate)
    to_place = [v for v in view.items if v.stake > 0]
    if not to_place:
        raise ValueError("nothing to place: no item has a stake")
    blocked = [
        f"{v.label}: {'; '.join(v.problems)}" for v in to_place if v.problems or v.price is None
    ]
    if blocked:
        raise ValueError("fix these first: " + " | ".join(blocked))
    bets = []
    for v in to_place:
        entries = [leg.entry for leg in v.legs]
        if v.item.kind == BetKind.STRAIGHT:
            e = entries[0]
            bets.append(
                ledger.place_bet(
                    session,
                    event_label=e.event_label,
                    market_type=e.market_type,
                    selection=e.selection,
                    line=e.line,
                    book=v.book,
                    price=v.price,
                    stake=v.stake,
                    fair_prob=v.fair_prob,
                    market_id=e.market_id,
                )
            )
            bets[-1].event_id = e.event_id
        else:
            legs = [
                {
                    "event_id": e.event_id,
                    "market_id": e.market_id,
                    "market_type": e.market_type,
                    "selection": e.selection,
                    "line": e.line,
                    "price": e.offers.get(v.book or "", e.best.price),
                    "fair_prob": e.fair_prob,
                }
                for e in entries
            ]
            bets.append(
                ledger.place_parlay(
                    session,
                    legs=legs,
                    book=v.book or "unknown",
                    price=v.price,
                    stake=v.stake,
                    fair_prob=v.fair_prob,
                    notes=f"from slate '{slate.name}'",
                )
            )
    slate.status = SlateStatus.PLACED
    session.flush()
    return bets


def draft_slates(session: Session) -> list[Slate]:
    return list(
        session.scalars(
            select(Slate).where(Slate.status == SlateStatus.DRAFT).order_by(Slate.updated_at.desc())
        )
    )
