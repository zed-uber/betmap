"""Match open bets to games and grade them from final scores and box scores.

Anything that can't be graded with confidence is reported with a reason, never guessed.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from betmap.tables import Bet, BetStatus, Event, Market, OddsSnapshot, PlayerGameStat, as_utc
from betmap.teams import abbr, full_name
from betmap.tracking import ledger

LABEL_SPLIT = re.compile(r"\s+(?:@|at|vs\.?|v\.?)\s+", re.IGNORECASE)
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PROP_SIDES = {"over", "under", "yes", "no"}
# A game-total bet this far from the books' total is probably a team or half total
# entered as "totals"; grading it against the full-game score would be wrong.
TOTAL_SANITY_POINTS = 10


def _n(value: int | None) -> int:
    return value or 0


PROP_STATS: dict[str, Callable[[PlayerGameStat], int]] = {
    "player_pass_yds": lambda s: _n(s.passing_yards),
    "player_pass_tds": lambda s: _n(s.passing_tds),
    "player_pass_completions": lambda s: _n(s.completions),
    "player_pass_attempts": lambda s: _n(s.attempts),
    "player_pass_interceptions": lambda s: _n(s.passing_interceptions),
    "player_rush_yds": lambda s: _n(s.rushing_yards),
    "player_rush_attempts": lambda s: _n(s.carries),
    "player_receptions": lambda s: _n(s.receptions),
    "player_reception_yds": lambda s: _n(s.receiving_yards),
    "player_rush_reception_yds": lambda s: _n(s.rushing_yards) + _n(s.receiving_yards),
    # Books count rushing, receiving, and return TDs; passing TDs don't count.
    "player_anytime_td": lambda s: (
        _n(s.rushing_tds) + _n(s.receiving_tds) + _n(s.special_teams_tds)
    ),
}


@dataclass
class GradeOutcome:
    bet: Bet
    status: BetStatus | None  # None: not graded
    reason: str = ""
    needs_manual: bool = False  # True when waiting won't help


def normalize_name(name: str) -> str:
    """'Amon-Ra St. Brown' -> 'amon ra st brown'; 'Marvin Harrison Jr.' -> 'marvin harrison'."""
    words = re.sub(r"[.'’]", "", name.lower()).replace("-", " ").split()
    return " ".join(w for w in words if w not in NAME_SUFFIXES)


def base_market(market_type: str) -> str:
    """Alternate lines grade like their main market."""
    return market_type.removeprefix("alternate_").removesuffix("_alternate")


def parse_prop_selection(selection: str) -> tuple[str, str] | None:
    """'Josh Allen Over' -> ('Josh Allen', 'over')."""
    player, _, side = selection.strip().rpartition(" ")
    side = side.lower()
    return (player, side) if player and side in PROP_SIDES else None


def team_side(selection: str, event: Event) -> str | None:
    """Which team a selection names ('home'/'away'): abbreviation, full name, or nickname."""
    wanted = selection.strip().lower()
    for side, team in (("home", event.home_team), ("away", event.away_team)):
        if wanted in {team.lower(), abbr(team).lower(), team.split()[-1].lower()}:
            return side
    return None


def find_event(session: Session, bet: Bet) -> Event | None:
    if bet.event_id is not None:
        return session.get(Event, bet.event_id)
    if bet.market_id is not None:
        market = session.get(Market, bet.market_id)
        if market is not None:
            return market.event
    teams = LABEL_SPLIT.split(bet.event_label.strip())
    if len(teams) != 2:
        return None
    away, home = (full_name(t.strip()) for t in teams)
    query = select(Event).where(
        ((Event.home_team == home) & (Event.away_team == away))
        | ((Event.home_team == away) & (Event.away_team == home))
    )
    # The meeting nearest to when the bet was logged: covers bets logged ahead of time,
    # live, or after the fact. Rematches are weeks apart, so this isn't ambiguous.
    placed = as_utc(bet.placed_at)
    candidates = [e for e in session.scalars(query) if e.kickoff]
    return min(candidates, key=lambda e: abs(as_utc(e.kickoff) - placed), default=None)


def market_total(session: Session, event: Event) -> float | None:
    """Median game-total line across books in the latest pull, if we have one."""
    market_id = session.scalar(
        select(Market.id).where(Market.event_id == event.id, Market.market_type == "totals")
    )
    if market_id is None:
        return None
    last = select(func.max(OddsSnapshot.fetched_at)).where(OddsSnapshot.market_id == market_id)
    lines = session.scalars(
        select(OddsSnapshot.line).where(
            OddsSnapshot.market_id == market_id,
            OddsSnapshot.fetched_at == last.scalar_subquery(),
            OddsSnapshot.line.is_not(None),
        )
    ).all()
    return median(lines) if lines else None


def _compare(diff: float) -> BetStatus:
    return BetStatus.WIN if diff > 0 else BetStatus.LOSS if diff < 0 else BetStatus.PUSH


def _over_under(value: float, line: float, side: str) -> BetStatus:
    return _compare(value - line if side == "over" else line - value)


def grade(session: Session, bet: Bet, event: Event) -> GradeOutcome:
    def manual(reason: str) -> GradeOutcome:
        return GradeOutcome(bet, None, reason, needs_manual=True)

    if not event.is_final:
        return GradeOutcome(bet, None, "game not final")
    market = base_market(bet.market_type)
    home, away = event.home_score, event.away_score

    if market in ("h2h", "spreads"):
        side = team_side(bet.selection, event)
        if side is None:
            return manual(f"can't tell which team '{bet.selection}' is")
        margin = home - away if side == "home" else away - home
        if market == "spreads":
            if bet.line is None:
                return manual("spread bet has no line")
            margin += bet.line
        return GradeOutcome(bet, _compare(margin))

    if market == "totals":
        side = bet.selection.strip().lower()
        if side not in ("over", "under") or bet.line is None:
            return manual("totals bet needs an Over/Under selection and a line")
        books_total = market_total(session, event)
        if books_total is not None and abs(bet.line - books_total) > TOTAL_SANITY_POINTS:
            return manual(
                f"line {bet.line:g} is far from the game total {books_total:g}; "
                "if it's a team total, change the market to team_totals"
            )
        return GradeOutcome(bet, _over_under(home + away, bet.line, side))

    if market == "team_totals":
        parsed = parse_prop_selection(bet.selection)  # 'WAS Over'
        team = parsed and team_side(parsed[0], event)
        if not team or parsed[1] not in ("over", "under") or bet.line is None:
            return manual("team total needs a selection like 'WAS Over' and a line")
        points = home if team == "home" else away
        return GradeOutcome(bet, _over_under(points, bet.line, parsed[1]))

    if market in PROP_STATS:
        parsed = parse_prop_selection(bet.selection)
        if parsed is None:
            return manual(f"can't read player and side from '{bet.selection}'")
        player, side = parsed
        rows = session.scalars(
            select(PlayerGameStat).where(PlayerGameStat.nflverse_game_id == event.nflverse_game_id)
        ).all()
        if not rows:
            # nflverse box scores usually land the morning after the game.
            return GradeOutcome(bet, None, "box score not available yet")
        wanted = normalize_name(player)
        row = next((r for r in rows if normalize_name(r.player_name) == wanted), None)
        if row is None:
            # nflverse omits players with no stats, so DNP and a name typo look the same.
            return manual(f"{player} not in box score (didn't play, or name mismatch)")
        value = PROP_STATS[market](row)
        if side in ("yes", "no"):
            scored = value >= 1
            return GradeOutcome(bet, BetStatus.WIN if scored == (side == "yes") else BetStatus.LOSS)
        if bet.line is None:
            return manual("over/under prop has no line")
        return GradeOutcome(bet, _over_under(value, bet.line, side))

    return manual(f"no automatic grading for {bet.market_type}")


def settle_open_bets(session: Session, dry_run: bool = False) -> list[GradeOutcome]:
    """Grade every open bet whose game is final; settle them unless `dry_run`."""
    outcomes = []
    open_bets = session.scalars(
        select(Bet).where(Bet.status == BetStatus.OPEN).order_by(Bet.id)
    ).all()
    for bet in open_bets:
        event = find_event(session, bet)
        if event is None:
            outcomes.append(
                GradeOutcome(bet, None, f"no game matches '{bet.event_label}'", needs_manual=True)
            )
            continue
        if not dry_run:
            bet.event_id = event.id
        outcome = grade(session, bet, event)
        if outcome.status is not None and not dry_run:
            ledger.settle_bet(session, bet.id, outcome.status)
        outcomes.append(outcome)
    return outcomes
