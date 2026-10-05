"""Closing line value from the last odds pull before kickoff.

The closing fair probability is the devigged consensus of every book quoting both sides
at the bet's line. CLV is then the EV our price had at that probability (Bet.clv).
Only as good as your pulls: pull odds shortly before kickoff to get a real close.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.config import get_settings
from betmap.odds.client import PICKEM_BOOKS
from betmap.odds.math import after_exchange_fee, devig
from betmap.odds.scan import line_key, selection_label, usable_market
from betmap.tables import Bet, Event, Market, OddsSnapshot, as_utc, utcnow
from betmap.tracking.grading import (
    find_event,
    normalize_name,
    parse_prop_selection,
    team_side,
)

# A pull older than this before kickoff is too stale to call a close.
CLOSE_WINDOW = timedelta(hours=6)
MIN_CLOSING_BOOKS = 2


@dataclass
class ClosingLine:
    fair_prob: float | None
    price: float | None  # the bet's own book, when it was in the pull
    n_books: int
    fetched_at: datetime


def find_market(session: Session, bet: Bet, event: Event) -> Market | None:
    if bet.market_id is not None:
        return session.get(Market, bet.market_id)
    player = None
    if bet.market_type.startswith("player_"):
        parsed = parse_prop_selection(bet.selection)
        if parsed is None:
            return None
        player = normalize_name(parsed[0])
    markets = session.scalars(
        select(Market).where(Market.event_id == event.id, Market.market_type == bet.market_type)
    )
    for market in markets:
        if (normalize_name(market.player) if market.player else None) == player:
            return market
    return None


def _is_bet_side(bet: Bet, side: str, market: Market, event: Event) -> bool:
    wanted = bet.selection.strip().lower()
    if wanted in (side.lower(), selection_label(market, side, event).lower()):
        return True
    team = team_side(bet.selection, event)
    return team is not None and side == (event.home_team if team == "home" else event.away_team)


def closing_quotes(session: Session, market: Market, event: Event) -> list[OddsSnapshot]:
    """Quotes from the last pull before kickoff, if it was close enough to count."""
    if event.kickoff is None:
        return []
    kickoff = as_utc(event.kickoff)
    snaps = session.scalars(select(OddsSnapshot).where(OddsSnapshot.market_id == market.id)).all()
    pulls = [as_utc(s.fetched_at) for s in snaps if as_utc(s.fetched_at) <= kickoff]
    if not pulls or kickoff - max(pulls) > CLOSE_WINDOW:
        return []
    return [s for s in snaps if as_utc(s.fetched_at) == max(pulls)]


def consensus(
    quotes: list[OddsSnapshot],
    market: Market,
    event: Event,
    side: str,
    line: float | None,
    method: str = "power",
) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Devigged consensus probability for `side` at `line`, and the quotes it came from."""
    key = line_key(market.market_type, side, line, event.home_team)
    by_book: dict[str, dict[str, float]] = defaultdict(dict)
    for s in quotes:
        if s.book in PICKEM_BOOKS:
            continue
        if line_key(market.market_type, s.side, s.line, event.home_team) == key:
            by_book[s.book][s.side] = s.price
    sides = sorted({s for q in by_book.values() for s in q})
    complete = [
        q
        for q in by_book.values()
        if len(sides) >= 2 and all(s in q for s in sides) and usable_market([q[s] for s in sides])
    ]
    if side not in sides or len(complete) < MIN_CLOSING_BOOKS:
        return None, by_book
    i = sides.index(side)
    return fmean(devig([q[s] for s in sides], method)[i] for q in complete), by_book


def closing_line(
    session: Session, bet: Bet, event: Event, method: str = "power"
) -> ClosingLine | None:
    market = find_market(session, bet, event)
    if market is None:
        return None
    quotes = closing_quotes(session, market, event)
    bet_side = next((s.side for s in quotes if _is_bet_side(bet, s.side, market, event)), None)
    if bet_side is None:
        return None
    fair, by_book = consensus(quotes, market, event, bet_side, bet.line, method)
    complete = sum(1 for q in by_book.values() if len(q) >= 2)
    book = bet.book.strip().lower()
    own = by_book.get(book, {}).get(bet_side)
    if own is not None:  # logged exchange prices are net of fees; compare like with like
        own = after_exchange_fee(own, get_settings().fee_rates.get(book, 0.0))
    return ClosingLine(fair, own, complete, max(s.fetched_at for s in quotes))


def fill_closing_lines(session: Session) -> int:
    """Record closing lines for bets whose game has kicked off; returns bets updated."""
    now = utcnow()
    updated = 0
    for bet in session.scalars(select(Bet).where(Bet.closing_fair_prob.is_(None))):
        event = find_event(session, bet)
        if event is None or event.kickoff is None or as_utc(event.kickoff) > now:
            continue
        close = closing_line(session, bet, event)
        if close is None or (close.fair_prob is None and close.price is None):
            continue
        bet.event_id = event.id
        bet.closing_fair_prob = close.fair_prob
        if bet.closing_price is None:
            bet.closing_price = close.price
        updated += 1
    session.flush()
    return updated
