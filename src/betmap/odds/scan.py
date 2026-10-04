"""Find +EV prices by comparing each book against the devigged market consensus.

For every market we take the latest pull, group quotes by line, and devig each book that
quotes every side at that line. Each book's price is then scored against the average fair
probability of the *other* books (leave-one-out), so an outlier can't mask its own edge.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from betmap.odds.math import DEVIG_METHODS, devig, expected_value, kelly_fraction
from betmap.tables import Event, Market, OddsSnapshot, utcnow
from betmap.teams import abbr, matchup

DEVIG_CHOICES = tuple(DEVIG_METHODS)


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


def as_utc(dt: datetime) -> datetime:
    # SQLite hands back naive datetimes; everything is stored in UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def last_pull_at(session: Session) -> datetime | None:
    value = session.scalar(select(func.max(OddsSnapshot.fetched_at)))
    return None if value is None else as_utc(value)


def _line_key(market_type: str, side: str, line: float | None, home_team: str) -> float | None:
    """Key that puts both sides of the same proposition together.

    Spread sides carry opposite signs (BUF -2.5 / KC +2.5), so key them from the home side.
    """
    if line is None:
        return None
    if market_type.endswith("spreads"):
        return line if side == home_team else -line
    return line


def _selection(market: Market, side: str, event: Event) -> str:
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
    min_ev: float = 0.01,
    min_books: int = 3,
    method: str = "power",
    market_type: str | None = None,
    books: set[str] | None = None,
    kelly_mult: float = 0.25,
    max_bet_fraction: float = 0.03,
    now: datetime | None = None,
) -> list[Opportunity]:
    """Return +EV prices sorted by EV, best first.

    `books` limits which books are reported (e.g. the ones you have accounts at); every
    book still contributes to the consensus.
    """
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
        key = _line_key(market.market_type, snap.side, snap.line, event.home_team)
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
            if all(s in q for s in sides)
        }
        if len(devigged) < min_books:
            continue

        for i, side in enumerate(sides):
            offers = {book: q[side] for book, q in by_book.items() if side in q}
            for book, (price, line) in offers.items():
                if books and book not in books:
                    continue
                # Leave-one-out: a book's own (possibly stale) price must not pull the
                # consensus toward itself and hide the edge.
                consensus = [probs[i] for b, probs in devigged.items() if b != book]
                if len(consensus) < min_books:
                    continue
                fair = fmean(consensus)
                ev = expected_value(fair, price)
                if ev < min_ev:
                    continue
                others = [p for b, (p, _) in offers.items() if b != book]
                opportunities.append(
                    Opportunity(
                        market_id=market_id,
                        event_label=matchup(event.away_team, event.home_team),
                        kickoff=as_utc(event.kickoff),
                        market_type=market.market_type,
                        selection=_selection(market, side, event),
                        line=line,
                        book=book,
                        price=price,
                        fair_prob=fair,
                        ev=ev,
                        n_books=len(consensus),
                        best_other=max(others) if others else None,
                        kelly=min(kelly_fraction(fair, price) * kelly_mult, max_bet_fraction),
                    )
                )
    opportunities.sort(key=lambda o: o.ev, reverse=True)
    return opportunities
