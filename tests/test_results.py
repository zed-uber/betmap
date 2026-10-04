from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, inspect, select, text

from betmap.data.nflverse import kickoff_utc, sync_games, sync_player_stats
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.tables import Bet, BetStatus, Event, utcnow
from betmap.tracking import ledger
from betmap.tracking.clv import fill_closing_lines
from betmap.tracking.grading import find_event, normalize_name, settle_open_bets
from betmap.tracking.results import update_results

from .odds_fixtures import book, game, h2h, spreads


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def eastern(dt: datetime) -> tuple[str, str]:
    """nflverse-style gameday/gametime strings for a UTC datetime (EDT, UTC-4)."""
    local = dt - timedelta(hours=4)
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M")


def schedule_row(
    game_id: str = "2026_04_KC_BUF",
    away: str = "KC",
    home: str = "BUF",
    kickoff: datetime | None = None,
    score: tuple[int, int] | None = None,  # (away, home)
    week: int = 4,
) -> dict:
    gameday, gametime = eastern(kickoff or utcnow() + timedelta(hours=48))
    return {
        "game_id": game_id,
        "season": 2026,
        "week": week,
        "gameday": gameday,
        "gametime": gametime,
        "away_team": away,
        "home_team": home,
        "away_score": score[0] if score else None,
        "home_score": score[1] if score else None,
    }


def stat_row(name: str, game_id: str = "2026_04_KC_BUF", **stats: int) -> dict:
    return {
        "game_id": game_id,
        "player_id": f"id-{name}",
        "player_display_name": name,
        "position": "QB",
        "team": "BUF",
        "season": 2026,
        "week": 4,
        **stats,
    }


def bet(session, selection: str, market: str = "h2h", line: float | None = None, **kw) -> Bet:
    defaults = {"event_label": "KC @ BUF", "book": "dk", "price": 1.91, "stake": 10}
    return ledger.place_bet(
        session, market_type=market, selection=selection, line=line, **defaults | kw
    )


def final_game(session, away_score: int, home_score: int, stats: list[dict] = ()) -> Event:
    kickoff = utcnow() - timedelta(hours=5)
    sync_games(session, [schedule_row(kickoff=kickoff, score=(away_score, home_score))])
    sync_player_stats(session, stats)
    return session.scalars(select(Event)).one()


# --- nflverse sync -------------------------------------------------------------------------


def test_kickoff_is_converted_from_eastern():
    assert kickoff_utc("2026-10-04", "13:00") == datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    assert kickoff_utc("2026-12-06", "13:00") == datetime(2026, 12, 6, 18, 0, tzinfo=UTC)


def test_sync_games_upserts_with_full_team_names(session):
    sync_games(session, [schedule_row()])
    sync_games(session, [schedule_row(score=(20, 24))])
    event = session.scalars(select(Event)).one()
    assert (event.away_team, event.home_team) == ("Kansas City Chiefs", "Buffalo Bills")
    assert (event.away_score, event.home_score, event.week) == (20, 24, 4)
    assert event.is_final


def test_odds_event_and_nflverse_game_are_one_event(session):
    ingest_events(session, [game()])  # Odds API event first
    sync_games(session, [schedule_row()])
    ingest_events(session, [game(event_id="evt1")])
    event = session.scalars(select(Event)).one()
    assert (event.odds_api_id, event.nflverse_game_id) == ("evt1", "2026_04_KC_BUF")


def test_odds_event_links_to_existing_game(session):
    sync_games(session, [schedule_row()])
    ingest_events(session, [game()])
    assert session.scalar(select(func.count(Event.id))) == 1


def test_player_stats_resync_replaces_rows(session):
    sync_player_stats(session, [stat_row("Josh Allen", passing_yards=100)])
    sync_player_stats(session, [stat_row("Josh Allen", passing_yards=280)])
    from betmap.tables import PlayerGameStat

    row = session.scalars(select(PlayerGameStat)).one()
    assert (row.player_name, row.passing_yards) == ("Josh Allen", 280)


def test_player_stats_skip_rows_without_player(session):
    blank = stat_row("x") | {"player_id": None, "player_display_name": None, "position": None}
    assert sync_player_stats(session, [blank, stat_row("Josh Allen")]) == 1


# --- grading -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("market", "selection", "line", "score", "expected"),
    [
        ("h2h", "BUF", None, (20, 24), BetStatus.WIN),
        ("h2h", "Kansas City Chiefs", None, (20, 24), BetStatus.LOSS),
        ("h2h", "Bills", None, (24, 24), BetStatus.PUSH),
        ("spreads", "BUF", -3.5, (20, 24), BetStatus.WIN),
        ("spreads", "BUF", -4, (20, 24), BetStatus.PUSH),
        ("spreads", "KC", 4.5, (20, 24), BetStatus.WIN),
        ("spreads", "KC", 2.5, (20, 24), BetStatus.LOSS),
        ("alternate_spreads", "BUF", -6.5, (20, 24), BetStatus.LOSS),
        ("totals", "Over", 43.5, (20, 24), BetStatus.WIN),
        ("totals", "under", 44, (20, 24), BetStatus.PUSH),
        ("team_totals", "BUF Over", 23.5, (20, 24), BetStatus.WIN),
        ("team_totals", "Chiefs Over", 20.5, (20, 24), BetStatus.LOSS),
    ],
)
def test_grade_game_markets(session, market, selection, line, score, expected):
    final_game(session, *score)
    b = bet(session, selection, market, line)
    [outcome] = settle_open_bets(session)
    assert outcome.status == expected
    assert b.status == expected and b.event_id is not None


@pytest.mark.parametrize(
    ("market", "selection", "line", "expected"),
    [
        ("player_pass_yds", "Josh Allen Over", 249.5, BetStatus.WIN),
        ("player_pass_yds", "Josh Allen Under", 280, BetStatus.PUSH),
        ("player_rush_reception_yds", "Josh Allen Over", 35.5, BetStatus.LOSS),
        ("player_anytime_td", "Josh Allen Yes", None, BetStatus.WIN),
        ("player_anytime_td", "Josh Allen No", None, BetStatus.LOSS),
        ("player_reception_yds", "Amon-Ra St. Brown Over", 79.5, BetStatus.WIN),
        ("player_reception_yds", "Marvin Harrison Jr. Under", 40.5, BetStatus.WIN),
    ],
)
def test_grade_props(session, market, selection, line, expected):
    final_game(
        session,
        20,
        24,
        stats=[
            stat_row("Josh Allen", passing_yards=280, rushing_yards=30, rushing_tds=1),
            stat_row("Amon-Ra St. Brown", receiving_yards=95),
            stat_row("Marvin Harrison", receiving_yards=12),
        ],
    )
    bet(session, selection, market, line)
    [outcome] = settle_open_bets(session)
    assert outcome.status == expected, outcome.reason


def test_ungradable_bets_are_left_open(session):
    final_game(session, 20, 24, stats=[stat_row("Josh Allen", passing_yards=280)])
    bets = [
        bet(session, "Patrick Mahomes Over", "player_pass_yds", 250.5),
        bet(session, "Seahawks", "h2h"),
        bet(session, "BUF", "first_half_spreads", -1.5),
        bet(session, "BUF", "h2h", event_label="nonsense"),
    ]
    outcomes = settle_open_bets(session)
    assert all(o.status is None and o.needs_manual for o in outcomes)
    assert all(b.status == BetStatus.OPEN for b in bets)
    assert "Patrick Mahomes not in box score" in outcomes[0].reason
    assert "no game matches 'nonsense'" in outcomes[3].reason


def test_total_far_from_market_needs_manual(session):
    data = game(
        bookmakers=[
            book(
                "dk",
                {
                    "totals": [
                        {"name": "Over", "price": 1.91, "point": 46.5},
                        {"name": "Under", "price": 1.91, "point": 46.5},
                    ]
                },
            )
        ]
    )
    ingest_events(session, [data])
    final = session.scalars(select(Event)).one()
    final.home_score, final.away_score = 24, 20
    team_total = bet(session, "Over", "totals", 21.5)
    alt_total = bet(session, "Over", "totals", 40.5)
    outcomes = settle_open_bets(session)
    assert outcomes[0].needs_manual and "far from the game total 46.5" in outcomes[0].reason
    assert team_total.status == BetStatus.OPEN
    assert alt_total.status == BetStatus.WIN


def test_waiting_bets_are_pending_not_manual(session):
    sync_games(session, [schedule_row()])  # not played yet
    bet(session, "BUF")
    final = utcnow() - timedelta(hours=5)
    sync_games(session, [schedule_row("2026_04_NE_NYJ", "NE", "NYJ", final, (10, 13))])
    bet(session, "Josh Allen Over", "player_pass_yds", 249.5, event_label="NE @ NYJ")
    outcomes = settle_open_bets(session)
    assert [(o.status, o.needs_manual, o.reason) for o in outcomes] == [
        (None, False, "game not final"),
        (None, False, "box score not available yet"),
    ]


def test_label_matches_nearest_meeting(session):
    now = utcnow()
    sync_games(
        session,
        [
            schedule_row("2026_02_KC_BUF", kickoff=now - timedelta(days=14), score=(10, 3)),
            schedule_row("2026_15_KC_BUF", kickoff=now + timedelta(days=60), week=15),
            schedule_row("2026_04_KC_BUF", kickoff=now + timedelta(days=2)),
        ],
    )
    b = bet(session, "BUF", event_label="kc at buf")
    assert find_event(session, b).nflverse_game_id == "2026_04_KC_BUF"
    # Logged after the fact: two days after week 2, long before week 15.
    late = bet(session, "BUF", event_label="KC @ BUF")
    late.placed_at = now - timedelta(days=12)
    assert find_event(session, late).nflverse_game_id == "2026_02_KC_BUF"


def test_normalize_name():
    assert normalize_name("D.J. Moore") == normalize_name("DJ Moore") == "dj moore"
    assert normalize_name("Kenneth Walker III") == "kenneth walker"


# --- CLV -----------------------------------------------------------------------------------


def closing_setup(session, pulled_before_kickoff: timedelta):
    """Game kicked off 3h ago; one odds pull at kickoff minus `pulled_before_kickoff`."""
    kickoff = utcnow() - timedelta(hours=3)
    sync_games(session, [schedule_row(kickoff=kickoff)])
    books = [
        book(k, {"h2h": h2h(1.8, 2.1), "spreads": spreads(-3, 1.95, 1.87)})
        for k in ("draftkings", "fanduel", "betmgm")
    ]
    data = game(bookmakers=books)
    data["commence_time"] = kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    ingest_events(session, [data], fetched_at=kickoff - pulled_before_kickoff)


def test_closing_line_for_manual_bet(session):
    closing_setup(session, timedelta(minutes=30))
    ml = bet(session, "BUF", "h2h", book="draftkings", price=2.0)
    spread = bet(session, "KC", "spreads", 3, price=2.0)
    moved = bet(session, "KC", "spreads", 2.5, price=2.0)
    assert fill_closing_lines(session) == 2
    # 1.8 / 2.1 devigs to roughly 53.6%; our 2.0 had about +7% EV at the close.
    assert ml.closing_fair_prob == pytest.approx(0.536, abs=0.005)
    assert ml.closing_price == 1.8
    assert ml.clv == pytest.approx(ml.closing_fair_prob * 2.0 - 1)
    # KC +3 closed at 1.87 vs BUF -3 at 1.95, so KC was the (slight) favourite.
    assert 0.5 < spread.closing_fair_prob < 0.52 and spread.closing_price is None
    assert moved.closing_fair_prob is None  # no books at +2.5 at the close


def test_stale_pull_is_not_a_close(session):
    closing_setup(session, timedelta(hours=10))
    b = bet(session, "BUF", "h2h")
    assert fill_closing_lines(session) == 0 and b.clv is None


def test_summary_reports_clv(session):
    closing_setup(session, timedelta(minutes=30))
    bet(session, "BUF", "h2h", price=2.0)
    bet(session, "KC", "h2h", price=1.9)
    fill_closing_lines(session)
    summary = ledger.summarize(session)
    assert len(summary.clv_values) == 2 and summary.beat_close == 0.5


# --- end to end ----------------------------------------------------------------------------


def test_update_results(session):
    kickoff = utcnow() - timedelta(hours=5)

    def fetch(seasons):
        assert seasons == [2025]
        games = [schedule_row(kickoff=kickoff, score=(20, 24))]
        return games, [stat_row("Josh Allen", passing_yards=280)]

    bet(session, "BUF")
    bet(session, "Josh Allen Over", "player_pass_yds", 300.5)
    bet(session, "Josh Allen Over", "player_kicking_points", 1.5)

    dry = update_results(session, fetch, seasons=[2025], dry_run=True)
    assert [o.status for o in dry.outcomes] == [BetStatus.WIN, BetStatus.LOSS, None]
    assert session.scalar(select(func.count()).where(Bet.status == BetStatus.OPEN)) == 3

    update = update_results(session, fetch, seasons=[2025])
    assert (update.games, update.player_lines) == (1, 1)
    assert update.summary() == (
        "Synced 1 games. Settled 2 bets (1 win, 1 loss); recorded 0 closing lines. "
        "Settle manually: #3 no automatic grading for player_kicking_points."
    )


# --- migration -----------------------------------------------------------------------------


def test_init_db_adds_new_columns_to_old_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, nflverse_game_id VARCHAR, "
                "odds_api_id VARCHAR, season INTEGER, week INTEGER, home_team VARCHAR NOT NULL, "
                "away_team VARCHAR NOT NULL, kickoff DATETIME)"
            )
        )
        conn.execute(text("INSERT INTO events (home_team, away_team) VALUES ('A', 'B')"))
    init_db(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("events")}
    assert {"home_score", "away_score"} <= columns
    with session_scope(engine) as s:
        assert s.scalars(select(Event)).one().home_team == "A"
