"""Find +EV prices by comparing each book against the devigged market consensus.

For every market we take the latest pull, group quotes by line, and devig each book that
quotes every side at that line. Each book's price is then scored against the average fair
probability of the *other* books (leave-one-out), so an outlier can't mask its own edge.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from betmap.odds.client import PICKEM_BOOKS
from betmap.odds.math import (
    DEVIG_METHODS,
    after_exchange_fee,
    devig,
    expected_value,
    kelly_fraction,
    overround,
)
from betmap.tables import Event, Market, OddsSnapshot, as_utc, utcnow
from betmap.teams import abbr, matchup

DEVIG_CHOICES = tuple(DEVIG_METHODS)
# A book's two-way prices only feed the consensus if they look like a real market. Thin
# exchange order books can quote both sides at -150 or worse (a 30%+ margin) or at 1.0.
MAX_OVERROUND = 0.15
MIN_OVERROUND = -0.03  # exchanges can sit slightly under 100% between bid and ask


def usable_market(prices: list[float]) -> bool:
    if any(p <= 1.0 for p in prices):
        return False
    return MIN_OVERROUND <= overround(prices) <= MAX_OVERROUND


@dataclass
class Opportunity:
    market_id: int
    event_label: str
    kickoff: datetime
    market_type: str
    selection: str
    line: float | None
    book: str
    price: float
    fair_prob: float
    ev: float
    n_books: int  # books in the consensus (excluding this one)
    best_other: float | None  # best price at any other book, for context
    kelly: float  # suggested bankroll fraction after multiplier and cap
    model_prob: float | None = None  # game model's probability, when one was blended in
    fee_adjusted: bool = False  # price is net of an exchange fee
    side: str = ""  # raw side as quoted: team name, Over/Under, Yes/No
    event_id: int = 0


def pickem_quotes(session: Session, now: datetime | None = None) -> int:
    """Pick'em prices in the latest pull for upcoming games (stored but not scanned)."""
    now = now or utcnow()
    return sum(
        1
        for snap, _, event in latest_quotes(session)
        if snap.book in PICKEM_BOOKS and event.kickoff and as_utc(event.kickoff) > now
    )


def last_pull_at(session: Session) -> datetime | None:
    value = session.scalar(select(func.max(OddsSnapshot.fetched_at)))
    return None if value is None else as_utc(value)


def line_key(market_type: str, side: str, line: float | None, home_team: str) -> float | None:
    """Key that puts both sides of the same proposition together.

    Spread sides carry opposite signs (BUF -2.5 / KC +2.5), so key them from the home side.
    """
    if line is None:
        return None
    if market_type.endswith("spreads"):
        return line if side == home_team else -line
    return line


def selection_label(market: Market, side: str, event: Event) -> str:
    if market.player:
        return f"{market.player} {side}"
    if side in (event.home_team, event.away_team):
        return abbr(side)
    return side


def latest_quotes(session: Session) -> list[tuple[OddsSnapshot, Market, Event]]:
    latest = (
        select(OddsSnapshot.market_id, func.max(OddsSnapshot.fetched_at).label("at"))
        .group_by(OddsSnapshot.market_id)
        .subquery()
    )
    query = (
        select(OddsSnapshot, Market, Event)
        .join(
            latest,
            (OddsSnapshot.market_id == latest.c.market_id)
            & (OddsSnapshot.fetched_at == latest.c.at),
        )
        .join(Market, OddsSnapshot.market_id == Market.id)
        .join(Event, Market.event_id == Event.id)
    )
    return [tuple(row) for row in session.execute(query)]


def scan(
    session: Session,
    *,
    min_ev: float | None = 0.01,
    min_books: int = 3,
    method: str = "power",
    market_type: str | None = None,
    books: set[str] | None = None,
    kelly_mult: float = 0.25,
    max_bet_fraction: float = 0.03,
    model_probs: dict[tuple[int, str, float | None], float] | None = None,
    model_weight: float = 0.0,
    fees: dict[str, float] | None = None,
    now: datetime | None = None,
) -> list[Opportunity]:
    """Return +EV prices sorted by EV, best first (every price when `min_ev` is None).

    `books` limits which books are reported (e.g. the ones you have accounts at); every
    book still contributes to the consensus. With `model_weight` > 0, the fair probability
    is that blend of the model's probability (keyed by market id, side, line) and the
    consensus; lines the model doesn't cover use the consensus alone. `fees` maps exchange
    books to taker fee rates: their prices are scored (and reported) net of fees, while the
    consensus uses quoted prices. Pick'em books are skipped entirely.
    """
    fees = fees or {}
    now = now or utcnow()
    # (market_id, line key) -> book -> side -> (price, line)
    groups: dict[tuple[int, float | None], dict[str, dict[str, tuple[float, float | None]]]]
    groups = defaultdict(lambda: defaultdict(dict))
    context: dict[int, tuple[Market, Event]] = {}

    for snap, market, event in latest_quotes(session):
        if event.kickoff is None or as_utc(event.kickoff) <= now:
            continue
        if market_type and market.market_type != market_type:
            continue
        if snap.book in PICKEM_BOOKS:
            continue
        key = line_key(market.market_type, snap.side, snap.line, event.home_team)
        groups[(market.id, key)][snap.book][snap.side] = (snap.price, snap.line)
        context[market.id] = (market, event)

    opportunities = []
    for (market_id, _), by_book in groups.items():
        market, event = context[market_id]
        sides = sorted({side for quotes in by_book.values() for side in quotes})
        if len(sides) < 2:
            continue
        devigged = {
            book: devig([q[s][0] for s in sides], method)
            for book, q in by_book.items()
            if all(s in q for s in sides) and usable_market([q[s][0] for s in sides])
        }
        if len(devigged) < min_books:
            continue

        for i, side in enumerate(sides):
            # What a bet at each book actually pays, after any exchange fee.
            offers = {
                book: (after_exchange_fee(q[side][0], fees.get(book, 0.0)), q[side][1])
                for book, q in by_book.items()
                if side in q and q[side][0] > 1.0
            }
            for book, (price, line) in offers.items():
                if books and book not in books:
                    continue
                # Leave-one-out: a book's own (possibly stale) price must not pull the
                # consensus toward itself and hide the edge.
                consensus = [probs[i] for b, probs in devigged.items() if b != book]
                if len(consensus) < min_books:
                    continue
                fair = fmean(consensus)
                model_p = (model_probs or {}).get((market_id, side, line))
                if model_weight and model_p is not None:
                    fair = model_weight * model_p + (1 - model_weight) * fair
                ev = expected_value(fair, price)
                if min_ev is not None and ev < min_ev:
                    continue
                others = [p for b, (p, _) in offers.items() if b != book]
                opportunities.append(
                    Opportunity(
                        market_id=market_id,
                        event_label=matchup(event.away_team, event.home_team),
                        kickoff=as_utc(event.kickoff),
                        market_type=market.market_type,
                        selection=selection_label(market, side, event),
                        line=line,
                        book=book,
                        price=price,
                        fair_prob=fair,
                        ev=ev,
                        n_books=len(consensus),
                        best_other=max(others) if others else None,
                        kelly=min(kelly_fraction(fair, price) * kelly_mult, max_bet_fraction),
                        model_prob=model_p if model_weight else None,
                        side=side,
                        event_id=event.id,
                        fee_adjusted=fees.get(book, 0.0) > 0,
                    )
                )
    opportunities.sort(key=lambda o: o.ev, reverse=True)
    return opportunities


@dataclass
class BoardEntry:
    """One side at one line, with every book's price and the best one."""

    best: Opportunity  # highest-paying book; its fair prob and EV
    offers: dict[str, float]  # book -> decimal price (net of exchange fees)
    fairs: dict[str, float]  # book -> fair prob from the other books (leave-one-out)

    def __getattr__(self, name: str):
        # market_id, event_id, event_label, kickoff, market_type, selection, side, line, ...
        if name == "best":  # not set yet (e.g. during copying): don't recurse
            raise AttributeError(name)
        return getattr(self.best, name)

    @property
    def key(self) -> tuple[int, str, float | None]:
        return (self.best.market_id, self.best.side, self.best.line)


def board(session: Session, min_books: int = 1, **filters) -> list[BoardEntry]:
    """Every priced side and line for upcoming games (the scan without an EV cutoff).

    Takes the same filters as `scan` except `min_ev`. Unlike the scan, a side only needs one
    other book to compare against, so thin markets (many props) still appear; each entry's
    `n_books` says how many books its fair price rests on. Sorted by kickoff, game, market.
    """
    entries: dict[tuple[int, str, float | None], BoardEntry] = {}
    for o in scan(session, min_ev=None, min_books=min_books, **filters):
        key = (o.market_id, o.side, o.line)
        entry = entries.get(key)
        if entry is None:
            entries[key] = BoardEntry(o, {o.book: o.price}, {o.book: o.fair_prob})
            continue
        entry.offers[o.book] = o.price
        entry.fairs[o.book] = o.fair_prob
        if o.price > entry.best.price:
            entry.best = o
    return sorted(
        entries.values(),
        key=lambda e: (e.kickoff, e.event_label, e.market_type, e.selection, e.line or 0),
    )
