from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import select

from betmap.data.nflverse import sync_games, sync_player_stats
from betmap.db import init_db, make_engine, session_scope
from betmap.portfolio.correlation import Leg, correlation_matrix, pair_correlation
from betmap.portfolio.optimize import overlaps, risk, simulate_returns, size
from betmap.portfolio.positions import Position, make_leg, open_positions
from betmap.tables import Event, utcnow
from betmap.tracking import ledger

from .test_results import schedule_row, stat_row


def pos(leg: Leg, prob: float, price: float = 2.0, stake: float = 0.0, label: str = "") -> Position:
    return Position(label or leg.market, f"game {leg.game}", leg, price, prob, "fair", stake)


HOME_ML = Leg(1, "h2h", 1)
HOME_SPREAD = Leg(1, "spreads", 1)
AWAY_ML = Leg(1, "h2h", -1)
OVER = Leg(1, "totals", 1)
UNDER = Leg(1, "totals", -1)


# --- correlation ---------------------------------------------------------------------------


def test_game_market_correlations():
    assert pair_correlation(HOME_ML, HOME_SPREAD) > 0.99
    assert pair_correlation(HOME_ML, AWAY_ML) < -0.99
    assert pair_correlation(OVER, UNDER) < -0.99
    assert pair_correlation(HOME_ML, OVER) == 0  # margin and total are independent
    assert pair_correlation(HOME_ML, Leg(2, "h2h", 1)) == 0  # other game
    home_tt = Leg(1, "team_totals", 1, team_sign=1)
    assert pair_correlation(home_tt, HOME_ML) == pytest.approx(0.72, abs=0.002)
    assert pair_correlation(Leg(1, "totals_h1", 1), OVER) == pytest.approx(0.7, abs=0.002)


def test_prop_correlations():
    qb = Leg(1, "player_pass_yds", 1, team_sign=1, player="josh allen", group="QB")
    wr = Leg(1, "player_reception_yds", 1, team_sign=1, player="khalil shakir", group="WR")
    opp_wr = Leg(1, "player_reception_yds", 1, team_sign=-1, player="travis kelce", group="TE")
    rb = Leg(1, "player_rush_yds", 1, team_sign=1, player="james cook", group="RB")
    catches = Leg(1, "player_receptions", 1, team_sign=1, player="khalil shakir", group="WR")
    assert pair_correlation(qb, wr) > 0.3  # QB-receiver stack
    assert pair_correlation(qb, wr) > pair_correlation(qb, opp_wr) > 0  # shootout only
    assert pair_correlation(rb, HOME_ML) == pytest.approx(0.23, abs=0.002)
    assert pair_correlation(rb, AWAY_ML) == pytest.approx(-0.23, abs=0.002)
    assert pair_correlation(wr, catches) > 0.75
    under = Leg(1, "player_reception_yds", -1, team_sign=1, player="khalil shakir", group="WR")
    assert pair_correlation(wr, under) < -0.99  # both sides of one prop


def test_correlation_matrix_is_valid_even_for_inconsistent_inputs():
    legs = [HOME_ML, HOME_SPREAD, AWAY_ML, OVER, UNDER, Leg(1, "totals_h1", 1)]
    corr = correlation_matrix(legs)
    assert np.allclose(np.diag(corr), 1)
    assert np.linalg.eigvalsh(corr).min() > 0


# --- simulation and risk -------------------------------------------------------------------


def test_simulated_marginals_and_nesting():
    ml, spread = pos(HOME_ML, 0.6), pos(HOME_SPREAD, 0.45)
    other = pos(Leg(2, "h2h", 1), 0.5)
    wins = simulate_returns([ml, spread, other], n=40_000) > 0
    assert wins.mean(axis=0) == pytest.approx([0.6, 0.45, 0.5], abs=0.01)
    # Covering the spread (harder) implies winning outright.
    assert (wins[:, 1] & ~wins[:, 0]).mean() < 0.01
    assert abs(np.corrcoef(wins[:, 0], wins[:, 2])[0, 1]) < 0.02


def test_risk_summary():
    bets = [pos(HOME_ML, 0.55, 2.0, stake=100), pos(Leg(2, "totals", 1), 0.5, 1.91, stake=50)]
    r = risk(bets)
    assert r.expected == pytest.approx(100 * (0.55 * 2 - 1) + 50 * (0.5 * 1.91 - 1), abs=2)
    assert r.worst == -150 and 0 < r.p_loss < 1
    assert r.by_game == {"game 1": 100, "game 2": 50}


def test_overlaps_flag_linked_pairs():
    found = overlaps([pos(HOME_ML, 0.5), pos(HOME_SPREAD, 0.5), pos(Leg(2, "h2h", 1), 0.5)])
    assert len(found) == 1 and found[0].correlation > 0.99


# --- sizing --------------------------------------------------------------------------------


def test_independent_bets_size_like_single_kelly():
    a, b = pos(Leg(1, "h2h", 1), 0.55), pos(Leg(2, "h2h", 1), 0.55)
    sized = size([a, b], [], equity=1000, kelly_mult=0.25, max_bet=0.05, max_game=0.1)
    for s in sized:
        assert s.independent == pytest.approx(0.025)
        assert s.portfolio == pytest.approx(0.025, abs=0.003)


def test_correlated_bets_are_sized_down_together():
    ml, spread = pos(HOME_ML, 0.55), pos(HOME_SPREAD, 0.55)
    sized = size([ml, spread], [], equity=1000, kelly_mult=0.25, max_bet=0.05, max_game=0.1)
    together = sum(s.portfolio for s in sized)
    assert together < 0.8 * sum(s.independent for s in sized)


def test_game_cap_counts_existing_bets():
    held = pos(HOME_ML, 0.55, stake=50)  # 5% of a 1000 bankroll already on game 1
    new = pos(OVER, 0.56)
    [s] = size([new], [held], equity=1000, kelly_mult=0.25, max_bet=0.03, max_game=0.06)
    assert s.independent == pytest.approx(0.03)
    assert s.portfolio <= 0.01 + 1e-6


def test_no_edge_no_stake():
    [s] = size([pos(HOME_ML, 0.45)], [], equity=1000)
    assert s.independent == 0 and s.portfolio == 0


# --- positions from the database -----------------------------------------------------------


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def test_make_leg_and_open_positions(session):
    sync_games(session, [schedule_row(kickoff=utcnow() + timedelta(days=2))])
    sync_player_stats(session, [stat_row("Josh Allen", passing_yards=250)])  # BUF, home
    event = session.scalars(select(Event)).one()
    assert make_leg(session, event, "spreads", "KC").direction == -1
    assert make_leg(session, event, "team_totals", "BUF Over").team_sign == 1
    assert make_leg(session, event, "totals_h1", "Over").direction == 1
    prop = make_leg(session, event, "player_pass_yds", "Josh Allen Under")
    assert (prop.direction, prop.team_sign, prop.player) == (-1, 1, "josh allen")
    assert make_leg(session, event, "h2h", "Seahawks") is None

    common = {"event_label": "KC @ BUF", "book": "dk", "stake": 20}
    ledger.place_bet(
        session, market_type="h2h", selection="BUF", price=1.8, fair_prob=0.6, **common
    )
    ledger.place_bet(
        session, market_type="totals", selection="Over", line=44.5, price=1.91, **common
    )
    ledger.place_bet(session, market_type="h2h", selection="Seahawks", price=2.0, **common)
    positions, skipped = open_positions(session)
    assert [p.prob_source for p in positions] == ["fair", "assumed"]
    assert positions[1].prob == pytest.approx(1 / 1.91 / 1.045)
    assert [b.selection for b in skipped] == ["Seahawks"]
