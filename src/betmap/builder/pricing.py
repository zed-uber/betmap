"""Fair prices for parlays, including same-game parlays.

The fair probability is the joint probability that every leg wins, from each leg's
consensus fair probability and the portfolio correlation model (exact for legs in
different games: their product). Same-game fair prices are only as good as that model's
correlation estimates.
"""

from dataclasses import dataclass, field
from math import prod

from sqlalchemy.orm import Session

from betmap.odds.client import EXCHANGE_BOOKS
from betmap.odds.scan import BoardEntry
from betmap.portfolio.correlation import Leg, joint_probability
from betmap.portfolio.positions import make_leg
from betmap.tables import Event
from betmap.teams import abbr, full_name
from betmap.tracking.grading import LABEL_SPLIT, team_side


@dataclass
class ParlayQuote:
    legs: list[BoardEntry]
    fair_prob: float  # joint probability every leg wins
    naive_prob: float  # product of leg probabilities, as if independent
    same_game: bool
    book: str | None = None  # book whose price is used
    price: float | None = None  # decimal: product of that book's legs, or the typed SGP price
    problems: list[str] = field(default_factory=list)

    @property
    def fair_price(self) -> float:
        return 1 / self.fair_prob if self.fair_prob > 0 else float("inf")

    @property
    def ev(self) -> float | None:
        return None if self.price is None else self.fair_prob * self.price - 1


def leg_for(session: Session, entry: BoardEntry) -> Leg:
    event = session.get(Event, entry.event_id)
    made = event and make_leg(session, event, entry.market_type, entry.selection)
    # Unrecognized selections still count; they just don't correlate with anything.
    return made or Leg(entry.event_id, entry.market_type, 1)


def price_parlay(
    session: Session,
    legs: list[BoardEntry],
    book: str | None = None,
    offered: float | None = None,
) -> ParlayQuote:
    """Fair joint probability and the price to compare it with.

    Cross-game parlays are priced as the product of one book's leg prices (`book`, or the
    best sportsbook that offers every leg; exchanges don't sell parlays). Same-game parlays need the book's quoted price in
    `offered`; books price those themselves. `offered` overrides either way.
    """
    parts = [(leg_for(session, e), e.fair_prob) for e in legs]
    quote = ParlayQuote(
        legs=legs,
        fair_prob=joint_probability(parts) if parts else 0.0,
        naive_prob=prod(e.fair_prob for e in legs),
        same_game=len({e.event_id for e in legs}) < len(legs),
        book=book,
    )
    if len(legs) < 2:
        quote.problems.append("a parlay needs at least two legs")
    keys = [e.key for e in legs]
    if len(set(keys)) < len(keys):
        quote.problems.append("the same leg is in twice")
    distinct = {e.key: e for e in legs}.values()
    lines = {(e.market_id, e.line if e.line is None else abs(e.line)) for e in distinct}
    if len(lines) < len(distinct):
        quote.problems.append("both sides of one line are in the parlay")

    if offered is not None:
        quote.price = offered
        return quote
    if quote.same_game:
        quote.problems.append("same-game parlay: enter the price the book quotes")
        return quote
    common = set.intersection(*(set(e.offers) for e in legs)) if legs else set()
    if book is not None:
        common &= {book}
    else:
        common -= EXCHANGE_BOOKS
    if not common:
        quote.problems.append(
            f"{book} doesn't offer every leg"
            if book
            else "no single sportsbook offers every leg; enter a quoted price instead"
        )
        return quote
    quote.book = max(common, key=lambda b: prod(e.offers[b] for e in legs))
    quote.price = prod(e.offers[quote.book] for e in legs)
    return quote


def board_event(session: Session, entries: list[BoardEntry], label: str) -> Event | None:
    """The upcoming game a label like 'DET @ CAR' or 'DET at CAR' names, among the board's games.

    Unlike matching bets (nearest meeting to when it was logged), pricing only makes sense
    for games that haven't started.
    """
    teams = LABEL_SPLIT.split(label.strip())
    if len(teams) != 2:
        return None
    wanted = {abbr(full_name(t.strip())) for t in teams}
    for e in entries:
        away, _, home = e.event_label.partition(" @ ")
        if {away, home} == wanted:
            return session.get(Event, e.event_id)
    return None


def find_entry(
    entries: list[BoardEntry],
    event_id: int,
    market_type: str,
    selection: str,
    line: float | None,
    event: Event,
) -> BoardEntry | None:
    """The board entry a typed leg means: 'BUF', 'Buffalo Bills', 'Over', 'Josh Allen Over'."""
    wanted = selection.strip().lower()
    team = team_side(selection, event)
    team_name = {"home": event.home_team, "away": event.away_team}.get(team or "")
    for e in entries:
        if e.event_id != event_id or e.market_type != market_type or e.line != line:
            continue
        if wanted in (e.selection.lower(), e.side.lower()) or (team_name and e.side == team_name):
            return e
    return None
