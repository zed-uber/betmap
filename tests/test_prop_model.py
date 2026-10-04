import random
from datetime import timedelta

import pytest
from sqlalchemy import select

from betmap.backtest.props import backtest_props
from betmap.data.nflverse import sync_games, sync_player_stats
from betmap.db import init_db, make_engine, session_scope
from betmap.models.evaluate import evaluate
from betmap.models.predict import (
    PROP_MODEL_NAME,
    load_predictions,
    predict_props,
    prop_side_probability,
)
from betmap.models.prop_model import PropModel, StatDistribution, stat_value, week_key
from betmap.odds.ingest import ingest_events
from betmap.tables import Event, Prediction, utcnow

from .odds_fixtures import book, game, prop_event
from .test_results import schedule_row, stat_row

ZERO = {
    "completions": 0,
    "attempts": 0,
    "passing_yards": 0,
    "passing_tds": 0,
    "passing_interceptions": 0,
    "carries": 0,
    "rushing_yards": 0,
    "rushing_tds": 0,
    "targets": 0,
    "receptions": 0,
    "receiving_yards": 0,
    "receiving_tds": 0,
    "special_teams_tds": 0,
}


def row(pid, name, position, team, opp, season, week, **stats):
    return {
        "game_id": f"{season}_{week:02d}_{team}_{opp}",
        "player_id": pid,
        "player_display_name": name,
        "position": position,
        "team": team,
        "opponent_team": opp,
        "season": season,
        "week": week,
        **ZERO,
        **stats,
    }


def receivers(seasons=(2023, 2024), seed=3):
    """A 70-yard WR1, a 25-yard WR3; DEF 'BAD' gives up over twice what 'AVG' does."""
    rng = random.Random(seed)
    rows = []
    for season in seasons:
        for week in range(1, 18):
            opp = "BAD" if week % 2 else "AVG"
            boost = 1.5 if opp == "BAD" else 0.7
            for pid, name, mean in (("wr1", "Star Wideout", 70), ("wr3", "Depth Guy", 25)):
                yards = max(0, round(rng.gauss(mean * boost, mean * 0.25)))
                rows.append(
                    row(
                        pid,
                        name,
                        "WR",
                        "BUF",
                        opp,
                        season,
                        week,
                        receiving_yards=yards,
                        receptions=max(0, round(yards / 12)),
                        targets=max(1, round(yards / 9)),
                    )
                )
    return rows


def test_distribution_probabilities():
    dist = StatDistribution(mean=60.0, inv_dispersion=0.3)
    assert dist.over(40.5) > dist.over(60.5) > dist.over(90.5)
    over, push, under = dist.outcome_probs(60)
    assert push > 0 and over + push + under == pytest.approx(1)
    assert dist.quantile(0.1) < 60 < dist.quantile(0.9)
    td = StatDistribution(mean=0.4, inv_dispersion=0.0)  # Poisson
    assert td.over(0.5) == pytest.approx(1 - 2.718281828**-0.4, abs=1e-6)


def test_fit_separates_players_and_reads_the_defense():
    rows = receivers()
    model = PropModel().fit(rows, before=week_key(2025, 1))
    star = model.predict("wr1", "player_reception_yds", "AVG")
    depth = model.predict("wr3", "player_reception_yds", "AVG")
    assert star.mean > 2 * depth.mean
    vs_bad = model.predict("wr1", "player_reception_yds", "BAD")
    assert vs_bad.mean > star.mean * 1.2
    # Markets a position doesn't play aren't predicted.
    assert model.predict("wr1", "player_pass_yds", "AVG") is None
    assert model.predict("nobody", "player_reception_yds", "AVG") is None


def test_fit_only_uses_earlier_weeks():
    rows = receivers()
    model = PropModel().fit(rows, before=week_key(2023, 5))
    assert all(
        week_key(g["season"], g["week"]) < week_key(2023, 5) for g in model.player_games["wr1"]
    )
    assert len(model.player_games["wr1"]) == 4


def test_stat_value_matches_grading_definitions():
    r = row(
        "x",
        "X",
        "RB",
        "BUF",
        "KC",
        2024,
        1,
        rushing_yards=50,
        receiving_yards=20,
        rushing_tds=1,
        receiving_tds=1,
    )
    assert stat_value("player_rush_reception_yds", r) == 70
    assert stat_value("player_anytime_td", r) == 2


def test_backtest_props_reports_accuracy_and_calibration():
    reports = backtest_props(receivers(), [2024])
    rec = reports["player_reception_yds"]
    assert rec.n > 20
    assert rec.mae_model < rec.mae_baseline  # it knows about the soft defense
    assert sum(b.n for b in rec.buckets.values()) == rec.n
    assert 0.5 < rec.in_80 / rec.n <= 1


def test_prop_side_probability():
    dist = StatDistribution(mean=60.0, inv_dispersion=0.3)
    assert prop_side_probability(dist, "Over", 55.5) + prop_side_probability(
        dist, "Under", 55.5
    ) == pytest.approx(1)
    td = StatDistribution(mean=0.5, inv_dispersion=0.0)
    assert prop_side_probability(td, "Yes", None) == pytest.approx(1 - 2.718281828**-0.5)
    assert prop_side_probability(dist, "Over", None) is None


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def test_predict_props_matches_players_by_name_and_team(session):
    ingest_events(session, [prop_event()])  # Josh Allen pass yds 249.5, KC @ BUF
    qb = [
        row("ja", "Josh Allen", "QB", "BUF", "MIA", 2025, w, passing_yards=280, attempts=34)
        for w in range(1, 9)
    ]
    # A namesake on another team must not be confused with him.
    other = [row("ja2", "Josh Allen", "QB", "JAX", "TEN", 2025, 1, passing_yards=100)]
    n, unmatched = predict_props(session, qb + other)
    assert n == 2 and unmatched == []
    probs = load_predictions(session, PROP_MODEL_NAME)
    over = next(p for (m, side, line), p in probs.items() if side == "Over")
    assert over > 0.6  # averages 280 against a 249.5 line

    n, unmatched = predict_props(session, other)
    assert n == 0 and unmatched == ["Josh Allen"]


def test_evaluate_compares_model_with_closing_line(session):
    kickoff = utcnow() - timedelta(hours=4)
    data = game(
        bookmakers=[
            book(
                k,
                {
                    "totals": [
                        {"name": "Over", "price": 1.91, "point": 44.5},
                        {"name": "Under", "price": 1.91, "point": 44.5},
                    ]
                },
            )
            for k in ("draftkings", "fanduel")
        ]
    )
    data["commence_time"] = kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    ingest_events(session, [data], fetched_at=kickoff - timedelta(minutes=20))
    event = session.scalars(select(Event)).one()
    market = event.markets[0]
    session.add_all(
        [
            Prediction(market_id=market.id, side="Over", line=44.5, model="m", fair_prob=0.7),
            Prediction(market_id=market.id, side="Under", line=44.5, model="m", fair_prob=0.3),
        ]
    )
    session.flush()

    assert evaluate(session).pending == 2  # not final yet

    sync_games(session, [schedule_row(kickoff=kickoff, score=(30, 24))])  # 54 points: over
    result = evaluate(session)
    group = result.groups[("m", "totals")]
    assert group.n == 1  # the Under is the same observation
    assert group.brier_model == pytest.approx(0.09)
    assert group.brier_market == pytest.approx(0.25)


def test_evaluate_grades_props_from_box_scores(session):
    kickoff = utcnow() - timedelta(hours=30)
    data = prop_event()
    data["commence_time"] = kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    ingest_events(session, [data], fetched_at=kickoff - timedelta(hours=1))
    market = session.scalars(select(Event)).one().markets[0]
    session.add(Prediction(market_id=market.id, side="Over", line=249.5, model="p", fair_prob=0.4))
    sync_games(session, [schedule_row(kickoff=kickoff, score=(20, 24))])
    session.flush()
    assert evaluate(session).pending == 1  # box score not in yet
    sync_player_stats(session, [stat_row("Josh Allen", passing_yards=310)])
    group = evaluate(session).groups[("p", "player_pass_yds")]
    assert group.n == 1 and group.hits == [True]
