"""Joint simulation of positions and correlated fractional Kelly sizing.

Sizing maximizes expected log growth of the whole portfolio, existing bets included,
over simulated joint outcomes. Fractional Kelly with multiplier k is full Kelly on a
bankroll k times larger, so the objective uses stakes / k; caps apply to the real stakes.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from betmap.odds.math import kelly_fraction
from betmap.portfolio.correlation import correlation_matrix, pair_correlation
from betmap.portfolio.positions import Position


def _thresholds(probs) -> np.ndarray:
    return norm.ppf(1 - np.clip(probs, 1e-6, 1 - 1e-6))


def simulate_returns(positions: list[Position], n: int = 20_000, seed: int = 0) -> np.ndarray:
    """(n, len(positions)) profit per unit staked: price - 1 on a win, -1 on a loss.

    Every leg of every position gets its own latent variable; a parlay wins only if all
    of its legs do.
    """
    if not positions:
        return np.zeros((n, 0))
    owner, legs, probs = [], [], []
    for i, p in enumerate(positions):
        for leg, prob in p.components:
            owner.append(i)
            legs.append(leg)
            probs.append(prob)
    corr = correlation_matrix(legs)
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, len(legs))) @ np.linalg.cholesky(corr).T
    leg_wins = latent > _thresholds(probs)
    owner = np.array(owner)
    wins = np.column_stack([leg_wins[:, owner == i].all(axis=1) for i in range(len(positions))])
    prices = np.array([p.price for p in positions])
    return np.where(wins, prices - 1, -1.0)


@dataclass
class Risk:
    expected: float
    sd: float
    p_loss: float
    p05: float  # 5th percentile outcome
    worst: float  # every position loses
    by_game: dict[str, float]  # stake at risk per game


def risk(positions: list[Position], n: int = 20_000) -> Risk:
    """Distribution of total P/L (in currency) for positions with stakes."""
    stakes = np.array([p.stake for p in positions])
    pnl = simulate_returns(positions, n) @ stakes if positions else np.zeros(n)
    by_game: dict[str, float] = defaultdict(float)
    for p in positions:
        by_game[p.game_label] += p.stake
    return Risk(
        expected=float(pnl.mean()),
        sd=float(pnl.std()),
        p_loss=float((pnl < 0).mean()),
        p05=float(np.percentile(pnl, 5)),
        worst=-float(stakes.sum()),
        by_game=dict(by_game),
    )


@dataclass
class Overlap:
    a: Position
    b: Position
    correlation: float


def overlaps(positions: list[Position], threshold: float = 0.25) -> list[Overlap]:
    """Pairs whose outcomes are meaningfully linked, strongest first."""
    found = []
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            # For parlays, the most strongly linked pair of legs.
            c = max(
                (pair_correlation(x, y) for x, _ in a.components for y, _ in b.components),
                key=abs,
            )
            if abs(c) >= threshold:
                found.append(Overlap(a, b, c))
    return sorted(found, key=lambda o: -abs(o.correlation))


@dataclass
class Sizing:
    position: Position
    independent: float  # bankroll fraction if sized alone (current scan behavior)
    portfolio: float  # bankroll fraction from the joint optimization


def size(
    candidates: list[Position],
    existing: list[Position],
    equity: float,
    kelly_mult: float = 0.25,
    max_bet: float = 0.03,
    max_game: float = 0.06,
    n: int = 20_000,
) -> list[Sizing]:
    """Joint fractional-Kelly stakes for `candidates`, holding `existing` bets fixed."""
    if not candidates:
        return []
    everything = existing + candidates
    returns = simulate_returns(everything, n)
    held = (
        returns[:, : len(existing)] @ np.array([p.stake / equity for p in existing])
        if existing and equity > 0
        else np.zeros(n)
    )
    R = returns[:, len(existing) :]

    def objective(f):
        wealth = np.maximum(1 + (held + R @ f) / kelly_mult, 1e-9)
        grad = -(R / wealth[:, None]).mean(axis=0) / kelly_mult
        return -np.log(wealth).mean(), grad

    # Per-game caps include what's already staked on that game.
    games: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(candidates):
        games[p.leg.game].append(i)
    held_by_game: dict[int, float] = defaultdict(float)
    for p in existing:
        held_by_game[p.leg.game] += p.stake / equity if equity > 0 else 0.0
    constraints = [
        {
            "type": "ineq",
            "fun": lambda f, idx=idx, room=max_game - held_by_game[g]: room - f[idx].sum(),
            "jac": lambda f, idx=idx: -np.isin(np.arange(len(f)), idx).astype(float),
        }
        for g, idx in games.items()
    ]
    # Never stake so much that losing everything breaks the log (with a margin).
    total_room = 0.95 * kelly_mult - sum(held_by_game.values())
    constraints.append(
        {"type": "ineq", "fun": lambda f: total_room - f.sum(), "jac": lambda f: -np.ones(len(f))}
    )
    independent = np.array(
        [min(kelly_fraction(p.prob, p.price) * kelly_mult, max_bet) for p in candidates]
    )
    start = np.clip(independent, 0, None) * 0.5
    if any(max_game - held_by_game[g] <= 0 for g in games) or total_room <= 0:
        start = np.zeros(len(candidates))
    result = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, max(0.0, max_bet))] * len(candidates),
        constraints=constraints,
    )
    stakes = np.clip(result.x, 0, max_bet)
    stakes[stakes < 1e-4] = 0.0
    return [
        Sizing(p, float(ind), float(s))
        for p, ind, s in zip(candidates, independent, stakes, strict=True)
    ]
