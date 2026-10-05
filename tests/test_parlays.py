from datetime import timedelta

import pytest
from sqlalchemy import select

from betmap.data.nflverse import sync_games, sync_player_stats
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.portfolio.correlation import Leg, joint_probability
from betmap.portfolio.optimize import overlaps, risk, simulate_returns
from betmap.portfolio.positions import Position, open_positions
from betmap.tables import BetKind, BetStatus, Event, utcnow
from betmap.tracking import ledger
from betmap.tracking.clv import fill_closing_lines
from betmap.tracking.grading import settle_open_bets

from .odds_fixtures import book, game, h2h, spreads
from .test_results import schedule_row, stat_row


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def games(session, *scores, hours_ago=5):
    """KC @ BUF and NE @ NYJ, final with the given (away, home) scores, or None = not final."""
    kickoff = utcnow() - timedelta(hours=hours_ago)
    rows = [
        schedule_row("2026_04_KC_BUF", "KC", "BUF", kickoff, scores[0]),
        schedule_row("2026_04_NE_NYJ", "NE", "NYJ", kickoff, scores[1]),
    ]
    sync_games(session, rows)
    by_id = {e.nflverse_game_id: e for e in session.scalars(select(Event))}
    return by_id["2026_04_KC_BUF"], by_id["2026_04_NE_NYJ"]


def parlay(session, legs, price=3.6, stake=10.0):
    return ledger.place_parlay(session, legs=legs, book="fanduel", price=price, stake=stake)


def leg(event, market, selection, line=None, price=1.91, **extra):
    return {"event_id": event.id, "market_type": market, "selection": selection, "line": line,
            "price": price, **extra}  # fmt: skip


# --- logging -------------------------------------------------------------------------------


def test_place_parlay(session):
    buf, nyj = games(session, None, None)
    bet = parlay(session, [leg(buf, "spreads", "BUF", -2.5), leg(nyj, "totals", "Over", 41.5)])
    assert bet.kind == BetKind.PARLAY and bet.market_type == "parlay"
    assert bet.selection == "2-leg: BUF -2.5 + Over 41.5"
    assert bet.event_label == "KC @ BUF / NE @ NYJ"
    assert [lg.status for lg in bet.legs] == [BetStatus.OPEN, BetStatus.OPEN]
    with pytest.raises(ValueError, match="at least two legs"):
        parlay(session, [leg(buf, "h2h", "BUF")])


# --- grading -------------------------------------------------------------------------------


def test_parlay_wins_when_every_leg_wins(session):
    buf, nyj = games(session, (20, 24), (10, 17))
    bet = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")], price=3.6)
    [outcome] = settle_open_bets(session)
    assert outcome.status == BetStatus.WIN and bet.payout == pytest.approx(36.0)
    assert [lg.status for lg in bet.legs] == [BetStatus.WIN, BetStatus.WIN]


def test_parlay_loses_as_soon_as_a_leg_loses(session):
    buf, _ = games(session, (24, 20), None)  # BUF lost; NE @ NYJ not played yet
    nyj = session.scalars(select(Event).where(Event.nflverse_game_id == "2026_04_NE_NYJ")).one()
    bet = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")])
    [outcome] = settle_open_bets(session)
    assert outcome.status == BetStatus.LOSS and bet.profit == -10


def test_parlay_waits_for_pending_legs(session):
    buf, nyj = games(session, (20, 24), None)
    bet = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")])
    [outcome] = settle_open_bets(session)
    assert outcome.status is None and not outcome.needs_manual
    assert "leg 2: game not final" in outcome.reason
    assert bet.legs[0].status == BetStatus.WIN and bet.status == BetStatus.OPEN


def test_pushed_cross_game_leg_drops_out(session):
    buf, nyj = games(session, (20, 24), (10, 17))
    bet = parlay(
        session,
        [leg(buf, "spreads", "BUF", -4, price=1.91), leg(nyj, "h2h", "NYJ", price=1.80)],
        price=1.91 * 1.80,
    )
    [outcome] = settle_open_bets(session)
    assert outcome.status == BetStatus.WIN
    assert bet.payout == pytest.approx(10 * 1.80)  # only the NYJ leg's price counts


def test_all_legs_push_refunds(session):
    buf, nyj = games(session, (20, 24), (10, 17))
    bet = parlay(session, [leg(buf, "spreads", "BUF", -4), leg(nyj, "spreads", "NYJ", -7)])
    settle_open_bets(session)
    assert bet.status == BetStatus.PUSH and bet.profit == 0


def test_same_game_push_needs_manual(session):
    buf, _ = games(session, (20, 24), None)
    bet = parlay(session, [leg(buf, "spreads", "BUF", -4), leg(buf, "totals", "Over", 40.5)])
    [outcome] = settle_open_bets(session)
    assert outcome.needs_manual and "same-game leg pushed" in outcome.reason
    assert bet.status == BetStatus.OPEN


def test_prop_leg_waits_for_box_score_then_grades(session):
    buf, nyj = games(session, (20, 24), (10, 17))
    bet = parlay(
        session, [leg(buf, "player_pass_yds", "Josh Allen Over", 249.5), leg(nyj, "h2h", "NYJ")]
    )
    [outcome] = settle_open_bets(session)
    assert outcome.status is None and "box score" in outcome.reason
    sync_player_stats(session, [stat_row("Josh Allen", passing_yards=301)])
    [outcome] = settle_open_bets(session)
    assert outcome.status == BetStatus.WIN and bet.status == BetStatus.WIN


def test_unreadable_leg_needs_manual_and_dry_run_changes_nothing(session):
    buf, nyj = games(session, (20, 24), (10, 17))
    bad = parlay(session, [leg(buf, "h2h", "Seahawks"), leg(nyj, "h2h", "NYJ")])
    good = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")])
    outcomes = settle_open_bets(session, dry_run=True)
    assert outcomes[0].needs_manual and outcomes[0].reason.startswith("leg 1:")
    assert outcomes[1].status == BetStatus.WIN
    assert good.status == BetStatus.OPEN and good.legs[0].status == BetStatus.OPEN
    assert bad.status == BetStatus.OPEN


def test_parlay_legs_cant_be_edited(session):
    buf, nyj = games(session, None, None)
    bet = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")])
    with pytest.raises(ValueError, match="void it"):
        ledger.edit_bet(session, bet.id, selection="something else")
    ledger.edit_bet(session, bet.id, stake=20, status=BetStatus.VOID)
    assert bet.status == BetStatus.VOID and bet.payout == 20


# --- probability and risk ------------------------------------------------------------------


def test_joint_probability():
    home_ml, home_spread = Leg(1, "h2h", 1), Leg(1, "spreads", 1)
    other = Leg(2, "h2h", 1)
    assert joint_probability([(home_ml, 0.6), (other, 0.5)]) == pytest.approx(0.3)
    # Covering implies winning, so ML + spread is about the spread alone, not the product.
    both = joint_probability([(home_ml, 0.6), (home_spread, 0.45)])
    assert both == pytest.approx(0.45, abs=0.01) and both > 0.6 * 0.45
    assert joint_probability([(Leg(1, "totals", 1), 0.5), (Leg(1, "totals", -1), 0.5)]) < 0.02


def test_simulated_parlay_matches_joint_probability():
    parts = [(Leg(1, "h2h", 1), 0.6), (Leg(1, "totals", 1), 0.5), (Leg(2, "h2h", -1), 0.55)]
    p = Position("parlay", "g", parts[0][0], 6.0, joint_probability(parts), "fair", parts=parts)
    wins = simulate_returns([p], n=60_000)[:, 0] > 0
    assert wins.mean() == pytest.approx(0.6 * 0.5 * 0.55, abs=0.01)


def test_open_parlay_in_risk_and_overlaps(session):
    buf, nyj = games(session, None, None, hours_ago=-48)
    parlay(
        session,
        [leg(buf, "h2h", "BUF", fair_prob=0.6), leg(nyj, "h2h", "NYJ", fair_prob=0.5)],
        price=3.8, stake=10,
    )  # fmt: skip
    ledger.place_bet(session, event_label="KC @ BUF", market_type="spreads", selection="BUF",
                     line=-2.5, book="dk", price=1.91, stake=20, fair_prob=0.5)  # fmt: skip
    positions, skipped = open_positions(session)
    assert not skipped and len(positions) == 2
    par = next(p for p in positions if p.parts)
    assert par.prob == pytest.approx(0.3) and len(par.parts) == 2
    r = risk(positions)
    assert r.expected == pytest.approx(10 * (0.3 * 3.8 - 1) + 20 * (0.5 * 1.91 - 1), abs=1.5)
    assert overlaps(positions)[0].correlation > 0.9  # BUF ML leg vs BUF spread


# --- CLV -----------------------------------------------------------------------------------


def test_parlay_closing_line(session):
    kickoff = utcnow() - timedelta(hours=3)
    buf, _ = games(session, None, None, hours_ago=3)

    def odds(event_id, home, away):
        data = game(event_id=event_id, bookmakers=[
            book(k, {"h2h": h2h(1.8, 2.1), "spreads": spreads(-3, 1.95, 1.87)})
            for k in ("draftkings", "fanduel")
        ])  # fmt: skip
        data["home_team"], data["away_team"] = home, away
        for b in data["bookmakers"]:
            for m in b["markets"]:
                for o in m["outcomes"]:
                    o["name"] = home if o["name"] == "Buffalo Bills" else away
        data["commence_time"] = kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        return data

    ingest_events(
        session,
        [
            odds("e1", "Buffalo Bills", "Kansas City Chiefs"),
            odds("e2", "New York Jets", "New England Patriots"),
        ],
        fetched_at=kickoff - timedelta(minutes=30),
    )
    nyj = session.scalars(select(Event).where(Event.nflverse_game_id == "2026_04_NE_NYJ")).one()
    cross = parlay(session, [leg(buf, "h2h", "BUF"), leg(nyj, "h2h", "NYJ")], price=3.6)
    sgp = parlay(session, [leg(buf, "h2h", "BUF"), leg(buf, "spreads", "BUF", -3)], price=2.6)
    assert fill_closing_lines(session) == 2
    p_ml = cross.legs[0].closing_fair_prob
    assert cross.closing_fair_prob == pytest.approx(p_ml * cross.legs[1].closing_fair_prob)
    assert cross.clv == pytest.approx(cross.closing_fair_prob * 3.6 - 1)
    # Same game: winning by more than 3 implies winning, so the joint isn't the product.
    p_spread = sgp.legs[1].closing_fair_prob
    assert sgp.closing_fair_prob > p_ml * p_spread
    assert sgp.closing_fair_prob == pytest.approx(min(p_ml, p_spread), abs=0.02)


# --- CLI -----------------------------------------------------------------------------------


def test_cli_bet_parlay(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from betmap.cli import app
    from betmap.config import get_settings
    from betmap.db import make_engine as engine_for

    db = tmp_path / "cli.db"
    monkeypatch.setenv("BETMAP_DB_PATH", str(db))
    get_settings.cache_clear()
    try:
        engine = engine_for(db)
        init_db(engine)
        with session_scope(engine) as s:
            games(s, None, None, hours_ago=-24)
        runner = CliRunner()
        r = runner.invoke(app, ["bet", "parlay", "--leg", "KC @ BUF|spreads|BUF|-2.5",
                                "--leg", "NE at NYJ|totals|Over|41.5|-110", "--odds", "+264",
                                "--stake", "10", "--book", "fanduel"])  # fmt: skip
        assert r.exit_code == 0, r.output
        assert "#1 2-leg: BUF -2.5 + Over 41.5 +264" in r.output
        r = runner.invoke(app, ["bet", "parlay", "--leg", "KC @ BUF|h2h|BUF",
                                "--leg", "SEA @ LA|h2h|SEA", "--odds", "+264",
                                "--stake", "10", "--book", "fanduel"])  # fmt: skip
        assert r.exit_code == 1 and "no game matches" in r.output
    finally:
        get_settings.cache_clear()
