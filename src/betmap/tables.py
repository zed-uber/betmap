from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(dt: datetime) -> datetime:
    # SQLite hands back naive datetimes; everything is stored in UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class BetStatus(StrEnum):
    OPEN = "open"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    nflverse_game_id: Mapped[str | None] = mapped_column(String, unique=True)
    odds_api_id: Mapped[str | None] = mapped_column(String, unique=True)
    season: Mapped[int | None]
    week: Mapped[int | None]
    home_team: Mapped[str]
    away_team: Mapped[str]
    kickoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Final scores from nflverse; both set means the game is final.
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    markets: Mapped[list["Market"]] = relationship(back_populates="event")


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    # Odds API market key: h2h, spreads, totals, player_pass_yds, ...
    market_type: Mapped[str]
    player: Mapped[str | None]

    event: Mapped[Event] = relationship(back_populates="markets")


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    book: Mapped[str]
    side: Mapped[str]  # team name, "Over"/"Under", etc.
    line: Mapped[float | None]
    price: Mapped[float]  # decimal odds
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    side: Mapped[str]
    line: Mapped[float | None]
    model: Mapped[str]
    fair_prob: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BetKind(StrEnum):
    STRAIGHT = "straight"
    PARLAY = "parlay"


class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Parlays are one bet (price = the parlay's price) with their legs in bet_legs.
    kind: Mapped[str | None] = mapped_column(String, default=BetKind.STRAIGHT)
    # Linked market when the bet came from ingested odds; manual bets may leave it empty.
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"))
    # The game this bet is on, once matched (via market_id or the event label).
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))
    event_label: Mapped[str]  # e.g. "KC @ BUF"
    market_type: Mapped[str]
    selection: Mapped[str]
    line: Mapped[float | None]
    book: Mapped[str]
    price: Mapped[float]  # decimal odds
    stake: Mapped[float]
    fair_prob: Mapped[float | None]
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String, default=BetStatus.OPEN)
    payout: Mapped[float | None] = mapped_column(Float)  # total returned incl. stake
    closing_price: Mapped[float | None] = mapped_column(Float)  # same book's closing price
    # Devigged consensus probability for this side at the close.
    closing_fair_prob: Mapped[float | None] = mapped_column(Float)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None]

    legs: Mapped[list["BetLeg"]] = relationship(
        back_populates="bet", order_by="BetLeg.id", cascade="all, delete-orphan"
    )

    @property
    def is_parlay(self) -> bool:
        return self.kind == BetKind.PARLAY

    @property
    def profit(self) -> float | None:
        return None if self.payout is None else self.payout - self.stake

    @property
    def clv(self) -> float | None:
        """Closing line value: EV of our price at the closing fair probability.

        Falls back to the price ratio against the same book's close when no consensus exists.
        """
        if self.closing_fair_prob is not None:
            return self.closing_fair_prob * self.price - 1
        if self.closing_price is not None:
            return self.price / self.closing_price - 1
        return None


class BetLeg(Base):
    """One selection in a parlay. Graded like a straight bet; the parlay settles from its legs."""

    __tablename__ = "bet_legs"

    id: Mapped[int] = mapped_column(primary_key=True)
    bet_id: Mapped[int] = mapped_column(ForeignKey("bets.id"), index=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"))
    market_type: Mapped[str]
    selection: Mapped[str]
    line: Mapped[float | None]
    price: Mapped[float | None]  # this leg's own decimal price, for reference
    fair_prob: Mapped[float | None]  # fair win probability when placed
    status: Mapped[str] = mapped_column(String, default=BetStatus.OPEN)
    closing_fair_prob: Mapped[float | None] = mapped_column(Float)

    bet: Mapped[Bet] = relationship(back_populates="legs")

    @property
    def book(self) -> str:
        return self.bet.book

    @property
    def placed_at(self) -> datetime:
        return self.bet.placed_at


class BankrollEntry(Base):
    __tablename__ = "bankroll_ledger"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str]  # deposit | withdrawal
    amount: Mapped[float]
    book: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PlayerGameStat(Base):
    """One player's box score line for one game (nflverse weekly player stats)."""

    __tablename__ = "player_game_stats"
    __table_args__ = (UniqueConstraint("nflverse_game_id", "player_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    nflverse_game_id: Mapped[str] = mapped_column(String, index=True)
    player_id: Mapped[str]
    player_name: Mapped[str]
    position: Mapped[str | None]
    team: Mapped[str | None]
    opponent_team: Mapped[str | None]
    season: Mapped[int]
    week: Mapped[int]
    completions: Mapped[int | None]
    attempts: Mapped[int | None]
    passing_yards: Mapped[int | None]
    passing_tds: Mapped[int | None]
    passing_interceptions: Mapped[int | None]
    carries: Mapped[int | None]
    rushing_yards: Mapped[int | None]
    rushing_tds: Mapped[int | None]
    targets: Mapped[int | None]
    receptions: Mapped[int | None]
    receiving_yards: Mapped[int | None]
    receiving_tds: Mapped[int | None]
    special_teams_tds: Mapped[int | None]
