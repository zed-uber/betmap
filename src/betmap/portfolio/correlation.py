"""How bet outcomes move together: a Gaussian copula over each game's margin and total.

Every position gets a latent normal variable; it wins when that variable clears the
threshold its win probability implies. Within a game, latents are correlated through:
  - the home margin M and the total T (independent of each other in NFL data), which
    fully determine moneylines, spreads, totals, and team totals
  - for props, the player's link to his team's margin and to the total ("game script"),
    plus correlations between one player's stats and between teammates' stats
Bets on different games are independent.

The prop numbers were estimated from nflverse 2021-2025 by `estimate_correlations`
(`betmap portfolio correlations` reruns it): each stat standardized within player-season,
regressed on the team's surprise vs the closing spread and total, with the remaining
residuals correlated within player and within team.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, pstdev

import numpy as np

from betmap.backtest.props import PROP_FLOORS
from betmap.models.prop_model import MARKETS_BY_GROUP, POSITION_GROUP, stat_value

# Prop stat (Over side) vs the player's team margin surprise and the total surprise.
GAME_SCRIPT = {
    "player_anytime_td": (0.17, 0.20),
    "player_pass_attempts": (-0.17, 0.10),
    "player_pass_completions": (-0.02, 0.17),
    "player_pass_interceptions": (-0.32, -0.02),
    "player_pass_tds": (0.30, 0.46),
    "player_pass_yds": (0.10, 0.31),
    "player_reception_yds": (0.04, 0.15),
    "player_receptions": (-0.03, 0.09),
    "player_rush_attempts": (0.28, 0.02),
    "player_rush_reception_yds": (0.25, 0.12),
    "player_rush_yds": (0.23, 0.08),
}

# Same player, two markets, after game script (symmetric; unlisted pairs ~0).
SAME_PLAYER = {
    frozenset(p): r
    for p, r in {
        ("player_anytime_td", "player_pass_tds"): -0.28,
        ("player_anytime_td", "player_reception_yds"): 0.31,
        ("player_anytime_td", "player_receptions"): 0.25,
        ("player_anytime_td", "player_rush_attempts"): 0.28,
        ("player_anytime_td", "player_rush_reception_yds"): 0.36,
        ("player_anytime_td", "player_rush_yds"): 0.32,
        ("player_pass_attempts", "player_pass_completions"): 0.88,
        ("player_pass_attempts", "player_pass_interceptions"): 0.21,
        ("player_pass_attempts", "player_pass_tds"): 0.23,
        ("player_pass_attempts", "player_pass_yds"): 0.72,
        ("player_pass_attempts", "player_rush_attempts"): 0.21,
        ("player_pass_completions", "player_pass_interceptions"): 0.14,
        ("player_pass_completions", "player_pass_tds"): 0.31,
        ("player_pass_completions", "player_pass_yds"): 0.79,
        ("player_pass_completions", "player_rush_attempts"): 0.20,
        ("player_pass_tds", "player_pass_yds"): 0.36,
        ("player_pass_yds", "player_rush_attempts"): 0.13,
        ("player_pass_interceptions", "player_pass_yds"): 0.12,
        ("player_reception_yds", "player_receptions"): 0.77,
        ("player_reception_yds", "player_rush_reception_yds"): 0.58,
        ("player_receptions", "player_rush_reception_yds"): 0.42,
        ("player_rush_attempts", "player_rush_reception_yds"): 0.67,
        ("player_rush_attempts", "player_rush_yds"): 0.73,
        ("player_rush_reception_yds", "player_rush_yds"): 0.86,
    }.items()
}

# Teammates, after game script.
PASSING = {"player_pass_yds", "player_pass_completions", "player_pass_attempts", "player_pass_tds"}
RECEIVING = {"player_reception_yds", "player_receptions"}
QB_TO_RECEIVER = 0.29
TD_TO_TD = -0.11

# Team total = (T +/- M) / 2 with sd(M)=13.5, sd(T)=13.0, as loadings on standardized M, T.
TEAM_TOTAL = (0.72, 0.69)
# Partial-game markets move with the full game, but not one for one.
PERIOD_SCALE = {"_h1": 0.7, "_h2": 0.6, "_q1": 0.5, "_q2": 0.5, "_q3": 0.5, "_q4": 0.5}


@dataclass(frozen=True)
class Leg:
    """What correlation needs to know about one position."""

    game: int  # event id
    market: str
    direction: int  # +1 home / over / yes, -1 away / under / no
    team_sign: int = 0  # props and team totals: +1 home team, -1 away team, 0 unknown
    player: str | None = None
    group: str | None = None  # QB / RB / WR / TE for props


def _period(market: str) -> tuple[str, float]:
    for suffix, scale in PERIOD_SCALE.items():
        if market.endswith(suffix):
            return market.removesuffix(suffix), scale
    return market.removeprefix("alternate_").removesuffix("_alternate"), 1.0


def loadings(leg: Leg) -> np.ndarray:
    """(M, T) loadings of the leg's winning direction; |loadings| <= 1."""
    base, scale = _period(leg.market)
    if base in ("h2h", "spreads"):
        vec = (1.0, 0.0)
    elif base == "totals":
        vec = (0.0, 1.0)
    elif base == "team_totals":
        vec = (TEAM_TOTAL[0] * leg.team_sign, TEAM_TOTAL[1])
    elif base in GAME_SCRIPT:
        margin, total = GAME_SCRIPT[base]
        vec = (margin * leg.team_sign, total)
    else:
        vec = (0.0, 0.0)
    out = np.array(vec) * scale * leg.direction
    norm = np.linalg.norm(out)
    return out / norm * 0.999 if norm > 0.999 else out


def pair_correlation(a: Leg, b: Leg) -> float:
    if a.game != b.game:
        return 0.0
    la, lb = loadings(a), loadings(b)
    corr = float(la @ lb)
    resid = np.sqrt(max(0.0, 1 - la @ la)) * np.sqrt(max(0.0, 1 - lb @ lb))
    signs = a.direction * b.direction
    if a.player and b.player:
        if a.player == b.player:
            # Same stat (e.g. two lines, or both sides) is the same variable.
            r = (
                1.0
                if a.market == b.market
                else SAME_PLAYER.get(frozenset((a.market, b.market)), 0.0)
            )
            corr += r * signs * resid
        elif a.team_sign and a.team_sign == b.team_sign:
            pair = {a.market, b.market}
            if pair & PASSING and pair & RECEIVING and {a.group, b.group} & {"QB"}:
                corr += QB_TO_RECEIVER * signs * resid
            elif a.market == b.market == "player_anytime_td":
                corr += TD_TO_TD * signs * resid
    return float(np.clip(corr, -0.999, 0.999))


def correlation_matrix(legs: list[Leg]) -> np.ndarray:
    """Pairwise correlations, repaired to the nearest valid (positive definite) matrix."""
    n = len(legs)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = pair_correlation(legs[i], legs[j])
    values, vectors = np.linalg.eigh(corr)
    if values.min() < 1e-6:
        values = np.clip(values, 1e-6, None)
        corr = vectors @ np.diag(values) @ vectors.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def estimate_correlations(player_rows: list[dict], schedule_rows: list[dict]) -> dict:
    """Recompute GAME_SCRIPT, SAME_PLAYER, and teammate correlations from nflverse rows."""
    games = {r["game_id"]: r for r in schedule_rows if r.get("spread_line") is not None}
    by_player_season: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in player_rows:
        if POSITION_GROUP.get(r.get("position") or "") and r["game_id"] in games:
            by_player_season[(r["player_id"], r["season"])].append(r)

    z: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    context: dict[tuple[str, str], tuple[str, str, float, float]] = {}
    for rows in by_player_season.values():
        if len(rows) < 6:
            continue
        group = POSITION_GROUP[rows[0]["position"]]
        for market in MARKETS_BY_GROUP[group]:
            xs = [stat_value(market, r) for r in rows]
            mu, sd = fmean(xs), pstdev(xs)
            if mu < PROP_FLOORS[market] or sd == 0:
                continue
            for r, x in zip(rows, xs, strict=True):
                g = games[r["game_id"]]
                sign = 1 if r["team"] == g["home_team"] else -1
                key = (r["game_id"], r["player_id"])
                z[key][market] = (x - mu) / sd
                context[key] = (
                    group,
                    r["team"],
                    sign * (g["result"] - g["spread_line"]) / 13.5,
                    (g["total"] - g["total_line"]) / 13.0,
                )

    script = {}
    for market in sorted({m for d in z.values() for m in d}):
        keys = [k for k in z if market in z[k]]
        X = np.array([context[k][2:] for k in keys])
        y = np.array([z[k][market] for k in keys])
        script[market] = tuple(float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0])

    def resid(key, market):
        a, b = script[market]
        return z[key][market] - a * context[key][2] - b * context[key][3]

    def corr(pairs):
        x, y = zip(*pairs, strict=True)
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 300 else None

    same: dict[frozenset, list] = defaultdict(list)
    for key, markets in z.items():
        ms = sorted(markets)
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                same[frozenset((a, b))].append((resid(key, a), resid(key, b)))

    teams: dict[tuple[str, str], list] = defaultdict(list)
    for key in z:
        teams[(key[0], context[key][1])].append(key)
    qb_rec, td_td = [], []
    for members in teams.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                for x, y in ((a, b), (b, a)):
                    if (
                        context[x][0] == "QB"
                        and "player_pass_yds" in z[x]
                        and "player_reception_yds" in z[y]
                    ):
                        qb_rec.append(
                            (resid(x, "player_pass_yds"), resid(y, "player_reception_yds"))
                        )
                if "player_anytime_td" in z[a] and "player_anytime_td" in z[b]:
                    td_td.append((resid(a, "player_anytime_td"), resid(b, "player_anytime_td")))

    return {
        "game_script": script,
        "same_player": {k: c for k, v in same.items() if (c := corr(v)) is not None},
        "qb_to_receiver": corr(qb_rec),
        "td_to_td": corr(td_td),
    }
