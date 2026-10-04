"""Player stat distributions for prop markets.

For each player and stat:
  mean = recency-weighted average of the player's recent games, shrunk toward the
         average for his position (so a few big games don't make a star)
       x opponent factor: how much that defense allows to the position vs the league,
         also shrunk toward 1
A negative binomial around that mean (dispersion fit per stat) gives P(over line).
Only games a player appeared in exist in nflverse, so this models "given he plays",
which matches how books void props for players who don't.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from types import SimpleNamespace

from scipy.stats import nbinom, poisson

from betmap.tracking.grading import PROP_STATS

POSITION_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
# Markets modeled per position group; others (e.g. a QB's receptions) are skipped.
MARKETS_BY_GROUP = {
    "QB": {
        "player_pass_yds",
        "player_pass_attempts",
        "player_pass_completions",
        "player_pass_tds",
        "player_pass_interceptions",
        "player_rush_yds",
        "player_rush_attempts",
        "player_anytime_td",
    },
    "RB": {
        "player_rush_yds",
        "player_rush_attempts",
        "player_receptions",
        "player_reception_yds",
        "player_rush_reception_yds",
        "player_anytime_td",
    },
    "WR": {"player_receptions", "player_reception_yds", "player_anytime_td"},
    "TE": {"player_receptions", "player_reception_yds", "player_anytime_td"},
}
POISSON_MARKETS = {"player_anytime_td", "player_pass_tds", "player_pass_interceptions"}


@dataclass(frozen=True)
class PropConfig:
    half_life_games: float = 8.0  # a player's game this many games ago counts half
    max_games: int = 32
    player_prior_games: float = 4.0  # shrinkage toward the position average, in games
    opponent_prior_games: float = 6.0  # shrinkage of the opponent factor toward 1
    opponent_games: int = 16


def week_key(season: int, week: int) -> int:
    return season * 100 + week


def stat_value(market: str, row: dict) -> float:
    """The row's value for a prop market, cached on the row (walk-forward refits reuse rows)."""
    cache = row.setdefault("_values", {})
    if market not in cache:
        fields = {k: v for k, v in row.items() if k != "_values"}
        cache[market] = float(PROP_STATS[market](SimpleNamespace(**fields)))
    return cache[market]


@dataclass(frozen=True)
class StatDistribution:
    mean: float
    inv_dispersion: float  # 1/r for the negative binomial; 0 means Poisson

    def _cdf(self, k: float) -> float:
        if k < 0:
            return 0.0
        if self.inv_dispersion <= 0 or self.mean <= 0:
            return float(poisson.cdf(k, max(self.mean, 1e-9)))
        r = 1 / self.inv_dispersion
        return float(nbinom.cdf(k, r, r / (r + self.mean)))

    def outcome_probs(self, line: float) -> tuple[float, float, float]:
        """(P(over), P(push), P(under)) for an integer stat."""
        over = 1 - self._cdf(math.floor(line))
        under = self._cdf(math.ceil(line) - 1)
        return over, max(0.0, 1 - over - under), under

    def over(self, line: float) -> float:
        """P(over | no push)."""
        over, _, under = self.outcome_probs(line)
        return over / (over + under) if over + under else 0.5

    def quantile(self, q: float) -> float:
        if self.inv_dispersion <= 0 or self.mean <= 0:
            return float(poisson.ppf(q, max(self.mean, 1e-9)))
        r = 1 / self.inv_dispersion
        return float(nbinom.ppf(q, r, r / (r + self.mean)))


class PropModel:
    def __init__(self, config: PropConfig | None = None):
        self.config = config or PropConfig()
        self.player_games: dict[str, list[dict]] = {}
        self.player_group: dict[str, str] = {}
        self.position_mean: dict[tuple[str, str], float] = {}
        self.opponent_factor: dict[tuple[str, str, str], float] = {}
        self.inv_dispersion: dict[str, float] = {}

    def fit(self, rows: list[dict], before: int) -> "PropModel":
        """Fit on player-game rows with week_key(season, week) < `before`."""
        cfg = self.config
        past = [r for r in rows if week_key(r["season"], r["week"]) < before]
        past.sort(key=lambda r: week_key(r["season"], r["week"]))
        by_player: dict[str, list[dict]] = defaultdict(list)
        for r in past:
            group = POSITION_GROUP.get(r.get("position") or "")
            if group:
                by_player[r["player_id"]].append(r)
                self.player_group[r["player_id"]] = group
        self.player_games = {p: games[-cfg.max_games :] for p, games in by_player.items()}

        # Position averages and per-game totals allowed, over roughly the last two seasons.
        recent = [r for r in past if week_key(r["season"], r["week"]) >= before - 200]
        sums: dict[tuple[str, str], list[float]] = defaultdict(list)
        allowed: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        games_by_opp: dict[str, list[str]] = defaultdict(list)
        for r in recent:
            group = POSITION_GROUP.get(r.get("position") or "")
            if not group or not r.get("opponent_team"):
                continue
            if r["game_id"] not in games_by_opp[r["opponent_team"]]:
                games_by_opp[r["opponent_team"]].append(r["game_id"])
            for market in MARKETS_BY_GROUP[group]:
                value = stat_value(market, r)
                sums[(group, market)].append(value)
                allowed[(r["opponent_team"], group, market)][r["game_id"]] += value
        self.position_mean = {k: sum(v) / len(v) for k, v in sums.items()}

        # League per-game total allowed to each position, then each defense vs that.
        league_game_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (_, group, market), per_game in allowed.items():
            league_game_totals[(group, market)].extend(per_game.values())
        league_avg = {k: sum(v) / len(v) for k, v in league_game_totals.items() if v}
        k2 = cfg.opponent_prior_games
        for (opp, group, market), per_game in allowed.items():
            avg = league_avg.get((group, market))
            if not avg:
                continue
            last = games_by_opp[opp][-cfg.opponent_games :]
            values = [per_game.get(g, 0.0) for g in last]
            self.opponent_factor[(opp, group, market)] = (sum(values) + k2 * avg) / (
                (len(values) + k2) * avg
            )

        # Dispersion: how much a player's games vary beyond Poisson noise, per market.
        excess: dict[str, list[float]] = defaultdict(list)
        for pid, games in self.player_games.items():
            if len(games) < 6:
                continue
            for market in MARKETS_BY_GROUP[self.player_group[pid]] - POISSON_MARKETS:
                xs = [stat_value(market, g) for g in games]
                m = sum(xs) / len(xs)
                if m > 0:
                    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
                    excess[market].append(max(0.0, (var - m) / m**2))
        self.inv_dispersion = {m: median(v) for m, v in excess.items()}
        return self

    def predict(self, player_id: str, market: str, opponent: str | None) -> StatDistribution | None:
        group = self.player_group.get(player_id)
        if group is None or market not in MARKETS_BY_GROUP[group]:
            return None
        cfg = self.config
        games = self.player_games.get(player_id, [])
        prior = self.position_mean.get((group, market), 0.0)
        weighted, total_w = 0.0, 0.0
        for ago, g in enumerate(reversed(games)):
            w = 0.5 ** (ago / cfg.half_life_games)
            weighted += w * stat_value(market, g)
            total_w += w
        mean = (weighted + cfg.player_prior_games * prior) / (total_w + cfg.player_prior_games)
        mean *= self.opponent_factor.get((opponent, group, market), 1.0) if opponent else 1.0
        inv = 0.0 if market in POISSON_MARKETS else self.inv_dispersion.get(market, 0.5)
        return StatDistribution(max(mean, 0.0), inv)

    def baseline(self, player_id: str, market: str, games: int = 8) -> float | None:
        """Naive comparison: plain average of the player's last `games` games."""
        recent = self.player_games.get(player_id, [])[-games:]
        if not recent:
            return None
        return sum(stat_value(market, g) for g in recent) / len(recent)
