"""Team-strength model for spreads, totals, and moneylines.

Two recency-weighted ridge regressions over past results:
  margin (home - away) = home_field + rating[home] - rating[away]
  total  (home + away) = base + pace[home] + pace[away]
Ridge shrinks every team toward average, which matters early in a season. Predicted
means become probabilities through a normal distribution, with whole-number lines
treated as discrete (a push on exactly the number).
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ModelConfig:
    half_life_days: float = 150.0  # a game this old counts half as much as one today
    ridge: float = 4.0  # shrinkage toward an average team, in games' worth of weight
    margin_sd: float = 13.5  # NFL margins around the expected margin
    total_sd: float = 13.0


@dataclass(frozen=True)
class Game:
    game_id: str
    kickoff: datetime
    home: str
    away: str
    neutral: bool = False
    home_score: int | None = None
    away_score: int | None = None

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None


@dataclass(frozen=True)
class GamePrediction:
    margin: float  # expected home - away
    total: float
    margin_sd: float
    total_sd: float

    def home_spread(self, home_line: float) -> float:
        """P(home covers home_line), e.g. home_line=-3 for home -3; pushes excluded."""
        return side_prob(self.margin, self.margin_sd, -home_line)

    def over(self, line: float) -> float:
        return side_prob(self.total, self.total_sd, line)

    def home_win(self) -> float:
        return side_prob(self.margin, self.margin_sd, 0.0)


def outcome_probs(mean: float, sd: float, line: float) -> tuple[float, float, float]:
    """(P(value > line), P(push), P(value < line)) for an integer-valued outcome."""
    win = 1 - norm.cdf((math.floor(line) + 0.5 - mean) / sd)
    lose = norm.cdf((math.ceil(line) - 0.5 - mean) / sd)
    return win, max(0.0, 1 - win - lose), lose


def side_prob(mean: float, sd: float, line: float) -> float:
    """P(value > line) given no push: comparable to a devigged two-way price."""
    win, _, lose = outcome_probs(mean, sd, line)
    return win / (win + lose)


def _weighted_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float, free: list[int]
) -> np.ndarray:
    """Solve min sum w (y - X b)^2 + ridge * |b|^2, leaving columns in `free` unpenalized."""
    penalty = np.full(X.shape[1], ridge)
    penalty[free] = 0.0
    A = X.T @ (X * w[:, None]) + np.diag(penalty)
    return np.linalg.solve(A, X.T @ (w * y))


class GameModel:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.teams: dict[str, int] = {}
        self.ratings: dict[str, float] = {}
        self.pace: dict[str, float] = {}
        self.home_field = 0.0
        self.base_total = 0.0
        self.n_games = 0

    def fit(self, games: list[Game], as_of: datetime) -> "GameModel":
        """Fit on final games that kicked off before `as_of`."""
        played = [g for g in games if g.is_final and g.kickoff < as_of]
        if not played:
            raise ValueError("no completed games to fit on")
        teams = sorted({t for g in played for t in (g.home, g.away)})
        self.teams = {t: i for i, t in enumerate(teams)}
        n, k = len(played), len(teams)
        age_days = np.array([(as_of - g.kickoff).total_seconds() / 86400 for g in played])
        w = 0.5 ** (age_days / self.config.half_life_days)
        home = np.array([self.teams[g.home] for g in played])
        away = np.array([self.teams[g.away] for g in played])
        rows = np.arange(n)

        # Margin: one column per team (+1 home, -1 away), last column is home field.
        X = np.zeros((n, k + 1))
        X[rows, home] += 1
        X[rows, away] -= 1
        X[:, k] = [0.0 if g.neutral else 1.0 for g in played]
        margin = np.array([g.home_score - g.away_score for g in played], dtype=float)
        beta = _weighted_ridge(X, margin, w, self.config.ridge, free=[k])
        self.ratings = {t: beta[i] for t, i in self.teams.items()}
        self.home_field = float(beta[k])

        # Total: +1 for both teams, last column is the league-average total.
        X = np.zeros((n, k + 1))
        X[rows, home] += 1
        X[rows, away] += 1
        X[:, k] = 1.0
        total = np.array([g.home_score + g.away_score for g in played], dtype=float)
        beta = _weighted_ridge(X, total, w, self.config.ridge, free=[k])
        self.pace = {t: beta[i] for t, i in self.teams.items()}
        self.base_total = float(beta[k])
        self.n_games = n
        return self

    def predict(self, home: str, away: str, neutral: bool = False) -> GamePrediction:
        # Teams with no history (e.g. relocation codes) are treated as average.
        margin = self.ratings.get(home, 0.0) - self.ratings.get(away, 0.0)
        margin += 0.0 if neutral else self.home_field
        total = self.base_total + self.pace.get(home, 0.0) + self.pace.get(away, 0.0)
        return GamePrediction(margin, total, self.config.margin_sd, self.config.total_sd)
