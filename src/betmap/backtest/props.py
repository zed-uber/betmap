"""Walk-forward accuracy and calibration backtest for the prop model.

There are no free historical prop lines, so this can't simulate betting. It answers two
questions that come first:
  - Does the model predict player stats better than a plain recent average?
  - Are its probabilities honest? Against a stand-in line (the player's recent average,
    rounded to x.5), events it calls 60% should happen about 60% of the time.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from betmap.models.prop_model import (
    MARKETS_BY_GROUP,
    POSITION_GROUP,
    PropConfig,
    PropModel,
    stat_value,
    week_key,
)

BUCKETS = ((0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.0))
# Only score players with enough volume that a book would post the prop; fringe players
# give degenerate stand-in lines (e.g. 0.5 receiving yards) that no book offers.
PROP_FLOORS = {
    "player_pass_yds": 150,
    "player_pass_attempts": 20,
    "player_pass_completions": 12,
    "player_pass_tds": 0.8,
    "player_pass_interceptions": 0.4,
    "player_rush_yds": 20,
    "player_rush_attempts": 6,
    "player_receptions": 2,
    "player_reception_yds": 20,
    "player_rush_reception_yds": 35,
    "player_anytime_td": 0.15,
}


@dataclass
class Bucket:
    predicted: list[float] = field(default_factory=list)
    hits: int = 0

    @property
    def n(self) -> int:
        return len(self.predicted)

    @property
    def actual(self) -> float | None:
        return self.hits / self.n if self.n else None


@dataclass
class PropMarketReport:
    abs_err_model: list[float] = field(default_factory=list)
    abs_err_baseline: list[float] = field(default_factory=list)
    in_50: int = 0
    in_80: int = 0
    brier: list[float] = field(default_factory=list)
    buckets: dict[tuple[float, float], Bucket] = field(
        default_factory=lambda: {b: Bucket() for b in BUCKETS}
    )

    @property
    def n(self) -> int:
        return len(self.abs_err_model)

    @property
    def mae_model(self) -> float:
        return fmean(self.abs_err_model)

    @property
    def mae_baseline(self) -> float:
        return fmean(self.abs_err_baseline)


def backtest_props(
    rows: list[dict],
    test_seasons: list[int],
    config: PropConfig | None = None,
    min_games: int = 4,
) -> dict[str, PropMarketReport]:
    """Predict each test week from earlier weeks only; report per market."""
    reports: dict[str, PropMarketReport] = defaultdict(PropMarketReport)
    by_week: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["season"] in test_seasons and POSITION_GROUP.get(r.get("position") or ""):
            by_week[week_key(r["season"], r["week"])].append(r)

    for key in sorted(by_week):
        model = PropModel(config).fit(rows, before=key)
        for row in by_week[key]:
            pid = row["player_id"]
            if len(model.player_games.get(pid, [])) < min_games:
                continue  # books rarely post props on players without a track record
            group = model.player_group[pid]
            for market in MARKETS_BY_GROUP[group]:
                dist = model.predict(pid, market, row.get("opponent_team"))
                base = model.baseline(pid, market)
                if dist is None or base is None or base < PROP_FLOORS.get(market, 0):
                    continue
                actual = stat_value(market, row)
                rep = reports[market]
                rep.abs_err_model.append(abs(actual - dist.mean))
                rep.abs_err_baseline.append(abs(actual - base))
                rep.in_50 += dist.quantile(0.25) <= actual <= dist.quantile(0.75)
                rep.in_80 += dist.quantile(0.10) <= actual <= dist.quantile(0.90)

                line = math.floor(base) + 0.5  # stand-in line; x.5 so there's no push
                p_over = dist.over(line)
                hit = actual > line
                rep.brier.append((p_over - hit) ** 2)
                for lo, hi in BUCKETS:
                    if lo <= p_over < hi or (hi == 1.0 and p_over == 1.0):
                        rep.buckets[(lo, hi)].predicted.append(p_over)
                        rep.buckets[(lo, hi)].hits += hit
                        break
    return dict(reports)
