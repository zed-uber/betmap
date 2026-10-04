from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.tables import BankrollEntry, Bet, BetStatus, utcnow


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
    )
    session.add(bet)
    session.flush()
    return bet


def settle_bet(
    session: Session, bet_id: int, result: BetStatus, closing_price: float | None = None
) -> Bet:
    bet = session.get(Bet, bet_id)
    if bet is None:
        raise ValueError(f"no bet with id {bet_id}")
    if bet.status != BetStatus.OPEN:
        raise ValueError(f"bet {bet_id} is already settled ({bet.status})")
    if result == BetStatus.OPEN:
        raise ValueError("cannot settle a bet as open")
    bet.status = result
    bet.payout = {
        BetStatus.WIN: bet.stake * bet.price,
        BetStatus.LOSS: 0.0,
        BetStatus.PUSH: bet.stake,
        BetStatus.VOID: bet.stake,
    }[result]
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
