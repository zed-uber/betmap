from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.tables import MANUAL, BankrollEntry, Bet, BetKind, BetLeg, BetStatus, Event, utcnow
from betmap.teams import matchup


def record_transfer(
    session: Session, kind: str, amount: float, book: str | None = None
) -> BankrollEntry:
    if kind not in ("deposit", "withdrawal"):
        raise ValueError(f"unknown transfer kind: {kind}")
    if amount <= 0:
        raise ValueError("amount must be positive")
    entry = BankrollEntry(kind=kind, amount=amount, book=book)
    session.add(entry)
    session.flush()
    return entry


def place_bet(
    session: Session,
    *,
    event_label: str,
    market_type: str,
    selection: str,
    book: str,
    price: float,
    stake: float,
    line: float | None = None,
    fair_prob: float | None = None,
    notes: str | None = None,
    market_id: int | None = None,
    source: str = MANUAL,
) -> Bet:
    if stake <= 0:
        raise ValueError("stake must be positive")
    if price <= 1:
        raise ValueError("price must be decimal odds > 1")
    bet = Bet(
        event_label=event_label,
        market_type=market_type,
        selection=selection,
        line=line,
        book=book,
        price=price,
        stake=stake,
        fair_prob=fair_prob,
        notes=notes,
        market_id=market_id,
        source=source,
    )
    session.add(bet)
    session.flush()
    return bet


def payout(status: BetStatus, stake: float, price: float) -> float:
    """Total returned, stake included, for a settled result."""
    return {
        BetStatus.WIN: stake * price,
        BetStatus.LOSS: 0.0,
        BetStatus.PUSH: stake,
        BetStatus.VOID: stake,
    }[status]


# Changing any of these changes what the bet is, so its game link and closing line no
# longer apply; results sync works them out again.
IDENTITY_FIELDS = ("event_label", "market_type", "selection", "line")


def edit_bet(session: Session, bet_id: int, **changes) -> Bet:
    """Change a bet's fields and/or its result ("status"; "open" un-settles it).

    Settled bets get their payout recomputed from the new stake, price, and result.
    """
    bet = session.get(Bet, bet_id)
    if bet is None:
        raise ValueError(f"no bet with id {bet_id}")
    allowed = {
        "event_label", "market_type", "selection", "line", "book", "price", "stake",
        "fair_prob", "notes", "status",
    }  # fmt: skip
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"can't edit {', '.join(sorted(unknown))}")
    if "stake" in changes and changes["stake"] <= 0:
        raise ValueError("stake must be positive")
    if "price" in changes and changes["price"] <= 1:
        raise ValueError("price must be decimal odds > 1")
    prob = changes.get("fair_prob")
    if prob is not None and not 0 < prob < 1:
        raise ValueError("fair prob must be between 0 and 1")
    status = BetStatus(changes.pop("status", bet.status))

    if any(f in changes and changes[f] != getattr(bet, f) for f in IDENTITY_FIELDS):
        if bet.is_parlay:
            raise ValueError("a parlay's legs can't be edited; void it and log it again")
        bet.event_id = None
        bet.market_id = None
        bet.closing_price = None
        bet.closing_fair_prob = None
    for field, value in changes.items():
        setattr(bet, field, value)

    if status == BetStatus.OPEN:
        bet.payout = None
        bet.settled_at = None
    else:
        bet.payout = payout(status, bet.stake, bet.price)
        bet.settled_at = bet.settled_at or utcnow()
    bet.status = status
    session.flush()
    return bet


def place_parlay(
    session: Session,
    *,
    legs: list[dict],
    book: str,
    price: float,
    stake: float,
    fair_prob: float | None = None,
    notes: str | None = None,
    source: str = MANUAL,
) -> Bet:
    """Log a parlay as one bet with its legs.

    Each leg dict has market_type, selection, and optionally line, event_id, market_id,
    price (the leg's own odds), and fair_prob. `price` is the parlay's price: the product of
    the legs for a cross-game parlay, or the book's quoted price for a same-game parlay.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two legs")
    games = []
    for leg in legs:
        event = session.get(Event, leg["event_id"]) if leg.get("event_id") else None
        label = matchup(event.away_team, event.home_team) if event else None
        if label and label not in games:
            games.append(label)

    def short(leg: dict) -> str:
        line = leg.get("line")
        return leg["selection"] + ("" if line is None else f" {line:g}")

    bet = place_bet(
        session,
        event_label=" / ".join(games) or "parlay",
        market_type="parlay",
        selection=f"{len(legs)}-leg: " + " + ".join(short(leg) for leg in legs),
        book=book,
        price=price,
        stake=stake,
        fair_prob=fair_prob,
        notes=notes,
        source=source,
    )
    bet.kind = BetKind.PARLAY
    for leg in legs:
        bet.legs.append(
            BetLeg(
                event_id=leg.get("event_id"),
                market_id=leg.get("market_id"),
                market_type=leg["market_type"],
                selection=leg["selection"],
                line=leg.get("line"),
                price=leg.get("price"),
                fair_prob=leg.get("fair_prob"),
            )
        )
    session.flush()
    return bet


def settle_bet(
    session: Session,
    bet_id: int,
    result: BetStatus,
    closing_price: float | None = None,
    payout_amount: float | None = None,
) -> Bet:
    """Grade an open bet. `payout_amount` overrides the usual payout (parlays with pushed legs)."""
    bet = session.get(Bet, bet_id)
    if bet is None:
        raise ValueError(f"no bet with id {bet_id}")
    if bet.status != BetStatus.OPEN:
        raise ValueError(f"bet {bet_id} is already settled ({bet.status})")
    if result == BetStatus.OPEN:
        raise ValueError("cannot settle a bet as open")
    bet.status = result
    bet.payout = (
        payout_amount if payout_amount is not None else payout(result, bet.stake, bet.price)
    )
    if closing_price is not None:
        bet.closing_price = closing_price
    bet.settled_at = utcnow()
    session.flush()
    return bet


@dataclass
class BankrollSummary:
    deposits: float
    withdrawals: float
    settled_profit: float
    open_exposure: float
    total_staked_settled: float
    wins: int
    losses: int
    pushes: int
    open_bets: int
    expected_profit_open: float | None
    clv_values: list[float]

    @property
    def avg_clv(self) -> float | None:
        return sum(self.clv_values) / len(self.clv_values) if self.clv_values else None

    @property
    def beat_close(self) -> float | None:
        """Share of bets with positive CLV."""
        if not self.clv_values:
            return None
        return sum(v > 0 for v in self.clv_values) / len(self.clv_values)

    @property
    def bankroll(self) -> float:
        """Cash on hand: net transfers plus settled P&L, minus stakes tied up in open bets."""
        return self.deposits - self.withdrawals + self.settled_profit - self.open_exposure

    @property
    def equity(self) -> float:
        """Bankroll counting open stakes at cost; the base used for sizing."""
        return self.bankroll + self.open_exposure

    @property
    def roi(self) -> float | None:
        if self.total_staked_settled == 0:
            return None
        return self.settled_profit / self.total_staked_settled


def summarize(session: Session) -> BankrollSummary:
    entries = session.scalars(select(BankrollEntry)).all()
    bets = session.scalars(select(Bet)).all()
    open_bets = [b for b in bets if b.status == BetStatus.OPEN]
    settled = [b for b in bets if b.status != BetStatus.OPEN]
    graded = [b for b in settled if b.status != BetStatus.VOID]

    expected = None
    if open_bets and all(b.fair_prob is not None for b in open_bets):
        expected = sum(b.stake * (b.fair_prob * b.price - 1) for b in open_bets)

    return BankrollSummary(
        deposits=sum(e.amount for e in entries if e.kind == "deposit"),
        withdrawals=sum(e.amount for e in entries if e.kind == "withdrawal"),
        settled_profit=sum(b.profit for b in settled),
        open_exposure=sum(b.stake for b in open_bets),
        total_staked_settled=sum(b.stake for b in graded),
        wins=sum(b.status == BetStatus.WIN for b in bets),
        losses=sum(b.status == BetStatus.LOSS for b in bets),
        pushes=sum(b.status == BetStatus.PUSH for b in bets),
        open_bets=len(open_bets),
        expected_profit_open=expected,
        clv_values=[b.clv for b in bets if b.clv is not None and b.status != BetStatus.VOID],
    )
