import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from betmap.backtest.games import ClosingLines, backtest
from betmap.cli import parse_seasons
from betmap.db import init_db, make_engine, session_scope
from betmap.models.game_model import Game, GameModel, GamePrediction, outcome_probs, side_prob
from betmap.models.predict import MODEL_NAME, load_predictions, predict_upcoming, side_probability
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import scan
from betmap.tables import Event, Prediction, utcnow

from .odds_fixtures import game
from .test_results import schedule_row

TEAMS = {f"T{i}": r for i, r in enumerate([-9, -6, -3, -1, 1, 3, 6, 9])}
HOME_FIELD = 2.0
START = datetime(2020, 9, 10, 17, tzinfo=UTC)


def simulate(seasons: int = 3, noise: float = 13.5, seed: int = 7):
    """Round-robin weeks between teams with known ratings and totals around 44."""
    rng = random.Random(seed)
    teams = list(TEAMS)
    history = []
    for season in range(seasons):
        for week in range(1, 15):
            rng.shuffle(teams)
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                kickoff = START + timedelta(days=365 * season + 7 * week)
                expected = TEAMS[home] - TEAMS[away] + HOME_FIELD
                margin = round(expected + rng.gauss(0, noise))
                total = max(abs(margin), round(44 + rng.gauss(0, 10)))
                if (total + margin) % 2:
                    total += 1
                g = Game(
                    f"{season}_{week}_{away}_{home}",
                    kickoff,
                    home,
                    away,
                    home_score=(total + margin) // 2,
                    away_score=(total - margin) // 2,
                )
                history.append((g, 2020 + season, week, expected))
    return history


def lines(spread: float, total: float = 44.5) -> ClosingLines:
    return ClosingLines(spread, -110, -110, total, -110, -110, None, None)


def test_fit_recovers_ratings():
    games = [g for g, *_ in simulate(seasons=6)]
    model = GameModel().fit(games, as_of=games[-1].kickoff + timedelta(days=1))
    fitted = [model.ratings[t] for t in TEAMS]
    assert np.corrcoef(fitted, list(TEAMS.values()))[0, 1] > 0.9
    assert 0 < model.home_field < 4
    assert 40 < model.base_total < 48


def test_fit_ignores_future_and_unplayed_games():
    games = [g for g, *_ in simulate(seasons=1)]
    cutoff = games[20].kickoff
    model = GameModel().fit(games + [Game("x", START, "T0", "T1")], as_of=cutoff)
    assert model.n_games == sum(g.kickoff < cutoff for g in games)
    with pytest.raises(ValueError):
        GameModel().fit(games, as_of=START)


def test_outcome_probs_handle_pushes():
    win, push, lose = outcome_probs(0.0, 13.5, 3)
    assert push > 0.02 and win + push + lose == pytest.approx(1)
    _, push, _ = outcome_probs(0.0, 13.5, 2.5)
    assert push == pytest.approx(0, abs=1e-12)
    assert side_prob(0.0, 13.5, 0) == pytest.approx(0.5)
    assert side_prob(3.0, 13.5, 2.5) > 0.5 > side_prob(3.0, 13.5, 3.5)


def test_prediction_sides_are_complementary():
    pred = GamePrediction(margin=3.0, total=44.0, margin_sd=13.5, total_sd=13.0)
    event = Event(home_team="Buffalo Bills", away_team="Kansas City Chiefs")
    home = side_probability(pred, "spreads", "Buffalo Bills", -2.5, event)
    away = side_probability(pred, "spreads", "Kansas City Chiefs", 2.5, event)
    assert home > 0.5 and home + away == pytest.approx(1)
    over = side_probability(pred, "totals", "Over", 43.5, event)
    under = side_probability(pred, "totals", "Under", 43.5, event)
    assert over > 0.5 and over + under == pytest.approx(1)
    assert side_probability(pred, "h2h", "Kansas City Chiefs", None, event) < 0.5


def test_backtest_finds_edge_against_a_bad_line():
    # The "market" is wrong by 6 points toward the away team: home should cover a lot.
    history = [(g, s, w, lines(exp - 6)) for g, s, w, exp in simulate(seasons=4)]
    report = backtest(history, [2022, 2023], min_ev=0.02)
    spreads = report.markets["spreads"]
    assert spreads.bets > 50 and spreads.roi > 0.05
    assert report.margin_mae_model < report.margin_mae_market


def test_backtest_loses_the_vig_against_a_sharp_line():
    history = [(g, s, w, lines(exp)) for g, s, w, exp in simulate(seasons=4)]
    report = backtest(history, [2022, 2023], min_ev=0.0)
    spreads = report.markets["spreads"]
    assert spreads.bets > 50 and spreads.roi < 0.02
    assert report.margin_mae_market <= report.margin_mae_model
    # Leaning on the market removes the (false) edges.
    blended = backtest(history, [2022, 2023], min_ev=0.02, model_weight=0.0)
    assert blended.markets["spreads"].bets == 0


def test_backtest_only_trains_on_the_past():
    history = [(g, s, w, lines(exp)) for g, s, w, exp in simulate(seasons=2)]
    report = backtest(history, [2020], min_ev=0.02)
    # Week 1 of the first season has nothing to train on, so it's skipped, not a crash.
    first_season = sum(1 for _, s, *_ in history if s == 2020)
    assert report.games == first_season - len(TEAMS) // 2


def test_parse_seasons():
    assert parse_seasons("2015-2017,2020") == [2015, 2016, 2017, 2020]


def test_predict_upcoming_and_scan_with_model():
    engine = make_engine(":memory:")
    init_db(engine)
    now = utcnow()
    past = [
        schedule_row(
            f"2026_{w:02d}_KC_BUF", kickoff=now - timedelta(days=7 * w), score=(10, 30), week=w
        )
        for w in range(1, 6)
    ]
    upcoming = schedule_row("2026_09_KC_BUF", kickoff=now + timedelta(hours=48), week=9)
    with session_scope(engine) as s:
        ingest_events(s, [game()])
        n = predict_upcoming(s, past + [upcoming])
        assert n == 4  # h2h and spread (-2.5), both sides
        event = s.scalars(select(Event).where(Event.odds_api_id == "evt1")).one()
        assert event.nflverse_game_id == "2026_09_KC_BUF"
        last_season = schedule_row(
            "2025_01_KC_BUF", kickoff=now - timedelta(days=300), score=(3, 7)
        )
        last_season["season"] = 2025
        predict_upcoming(s, [last_season, *past, upcoming])
        assert not s.scalars(select(Event).where(Event.nflverse_game_id == "2025_01_KC_BUF")).all()
        probs = load_predictions(s)
        home_ml = probs[
            (next(m for m, side, _ in probs if side == "Buffalo Bills"), "Buffalo Bills", None)
        ]
        assert home_ml > 0.6  # BUF won every meeting by 20

        # Re-running replaces rather than duplicates.
        predict_upcoming(s, past + [upcoming])
        assert len(s.scalars(select(Prediction).where(Prediction.model == MODEL_NAME)).all()) == 4

        plain = {(o.selection, o.book): o for o in scan(s, min_ev=-1)}
        blended = {
            (o.selection, o.book): o
            for o in scan(s, min_ev=-1, model_probs=probs, model_weight=0.5)
        }
        assert blended[("BUF", "soft")].fair_prob > plain[("BUF", "soft")].fair_prob
        assert blended[("BUF", "soft")].model_prob is not None
        assert plain[("BUF", "soft")].model_prob is None
