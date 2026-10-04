"""Schedules, final scores, and player box scores from nflverse.

Fetching needs the optional `stats` extra (nflreadpy). Syncing takes plain row dicts,
so it can be tested without it.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from betmap.tables import Event, PlayerGameStat, as_utc
from betmap.teams import full_name

EASTERN = ZoneInfo("America/New_York")
# Odds API events and nflverse games are the same game if the teams match and kickoffs
# are this close (they can disagree by minutes; flexed games move by hours, not days).
SAME_GAME_WINDOW = timedelta(hours=36)

STAT_COLUMNS = [
    "game_id",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "season",
    "week",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "special_teams_tds",
]


class StatsUnavailable(Exception):
    pass


def fetch(seasons: list[int]) -> tuple[list[dict], list[dict]]:
    """Download schedules and weekly player stats; returns (games, player_stats) rows."""
    try:
        import nflreadpy
    except ImportError:
        raise StatsUnavailable(
            "nflverse sync needs the stats extra: pip install -e '.[stats]'"
        ) from None
    games = nflreadpy.load_schedules(seasons).to_dicts()
    stats = nflreadpy.load_player_stats(seasons, summary_level="week")
    return games, stats.select(STAT_COLUMNS).to_dicts()


def current_season(today: datetime | None = None) -> int:
    # The NFL season is named for the year it starts; Jan-Feb games belong to the prior one.
    today = today or datetime.now(UTC)
    return today.year if today.month >= 3 else today.year - 1


def kickoff_utc(gameday: str, gametime: str | None) -> datetime:
    """nflverse gives local US Eastern date and time."""
    local = datetime.fromisoformat(f"{gameday}T{gametime or '13:00'}").replace(tzinfo=EASTERN)
    return local.astimezone(UTC)


def find_same_game(
    session: Session, home: str, away: str, kickoff: datetime, **filters
) -> Event | None:
    """An existing event for the same two teams near this kickoff (either home/away order)."""
    query = select(Event).where(
        ((Event.home_team == home) & (Event.away_team == away))
        | ((Event.home_team == away) & (Event.away_team == home))
    )
    for column, value in filters.items():
        query = query.where(getattr(Event, column) == value)
    for event in session.scalars(query):
        if event.kickoff and abs(as_utc(event.kickoff) - kickoff) <= SAME_GAME_WINDOW:
            return event
    return None


def sync_games(session: Session, rows: Iterable[dict]) -> int:
    """Upsert games by nflverse game_id, linking events first seen via The Odds API."""
    by_game_id = {
        e.nflverse_game_id: e
        for e in session.scalars(select(Event).where(Event.nflverse_game_id.is_not(None)))
    }
    count = 0
    for row in rows:
        home, away = full_name(row["home_team"]), full_name(row["away_team"])
        kickoff = kickoff_utc(row["gameday"], row["gametime"])
        event = by_game_id.get(row["game_id"]) or find_same_game(
            session, home, away, kickoff, nflverse_game_id=None
        )
        if event is None:
            event = Event(home_team=home, away_team=away, kickoff=kickoff)
            session.add(event)
        event.nflverse_game_id = row["game_id"]
        event.season = row["season"]
        event.week = row["week"]
        event.home_score = row["home_score"]
        event.away_score = row["away_score"]
        if event.odds_api_id is None:
            # Odds API kickoff times are more precise; keep theirs when we have them.
            event.kickoff = kickoff
        by_game_id[row["game_id"]] = event
        count += 1
    session.flush()
    return count


def sync_player_stats(session: Session, rows: Iterable[dict]) -> int:
    """Replace stored box scores for every game present in `rows`."""
    # nflverse includes a few blank team placeholder rows with no player; skip them.
    rows = [r for r in rows if r.get("player_id") and r.get("player_display_name")]
    game_ids = {r["game_id"] for r in rows}
    if game_ids:
        session.execute(delete(PlayerGameStat).where(PlayerGameStat.nflverse_game_id.in_(game_ids)))
    session.add_all(
        PlayerGameStat(
            nflverse_game_id=r["game_id"],
            player_name=r["player_display_name"],
            **{k: r.get(k) for k in STAT_COLUMNS if k not in ("game_id", "player_display_name")},
        )
        for r in rows
    )
    session.flush()
    return len(rows)
