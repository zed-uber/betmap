from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


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


class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Linked market when the bet came from ingested odds; manual bets may leave it empty.
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"))
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
    closing_price: Mapped[float | None] = mapped_column(Float)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None]

    @property
    def profit(self) -> float | None:
        return None if self.payout is None else self.payout - self.stake


class BankrollEntry(Base):
    __tablename__ = "bankroll_ledger"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str]  # deposit | withdrawal
    amount: Mapped[float]
    book: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
