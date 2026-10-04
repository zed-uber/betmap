"""Store game-model probabilities for every line in the latest odds pull."""

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from betmap.data import nflverse
from betmap.models.game_model import GameModel, GamePrediction, ModelConfig
from betmap.models.prop_model import PropConfig, PropModel, StatDistribution, week_key
from betmap.odds.scan import latest_quotes
from betmap.tables import Event, Prediction, as_utc, utcnow
from betmap.teams import abbr
from betmap.tracking.grading import normalize_name

MODEL_NAME = "ratings-v1"
PROP_MODEL_NAME = "props-v1"
GAME_MARKETS = ("h2h", "spreads", "totals")


def side_probability(
    pred: GamePrediction, market_type: str, side: str, line: float | None, event: Event
) -> float | None:
    """Model probability that `side` wins at `line` (pushes excluded)."""
    is_home = side == event.home_team
    if market_type == "h2h":
        p_home = pred.home_win()
        return p_home if is_home else 1 - p_home
    if line is None:
        return None
    if market_type == "spreads":
        # A home line h is the away line -h; P(away covers a) = 1 - P(home covers -a).
        return pred.home_spread(line) if is_home else 1 - pred.home_spread(-line)
    if market_type == "totals":
        p_over = pred.over(line)
        return p_over if side == "Over" else 1 - p_over
    return None


def predict_upcoming(
    session: Session,
    schedule_rows: Iterable[dict],
    config: ModelConfig | None = None,
    now: datetime | None = None,
) -> int:
    """Fit on completed games and store predictions for upcoming games' quoted lines.

    Returns the number of predictions written. Replaces this model's earlier predictions
    for the same markets.
    """
    now = now or utcnow()
    rows = list(schedule_rows)
    # Link Odds API events to nflverse games. Only the latest season can have quoted
    # games; older seasons are training data and don't need event rows.
    latest = max(r["season"] for r in rows)
    nflverse.sync_games(session, [r for r in rows if r["season"] == latest])
    model = GameModel(config).fit([nflverse.model_game(r) for r in rows], as_of=now)
    neutral = {r["game_id"]: r.get("location") == "Neutral" for r in rows}

    predictions: dict[tuple[int, str, float | None], float] = {}
    for snap, market, event in latest_quotes(session):
        if market.market_type not in GAME_MARKETS or market.player:
            continue
        if event.kickoff is None or as_utc(event.kickoff) <= now:
            continue
        pred = model.predict(
            abbr(event.home_team),
            abbr(event.away_team),
            neutral.get(event.nflverse_game_id, False),
        )
        prob = side_probability(pred, market.market_type, snap.side, snap.line, event)
        if prob is not None:
            predictions[(market.id, snap.side, snap.line)] = prob

    return _store(session, MODEL_NAME, predictions)


def _store(
    session: Session, model: str, predictions: dict[tuple[int, str, float | None], float]
) -> int:
    """Replace `model`'s predictions for the markets in `predictions`."""
    market_ids = {m for m, _, _ in predictions}
    if market_ids:
        session.execute(
            delete(Prediction).where(
                Prediction.model == model, Prediction.market_id.in_(market_ids)
            )
        )
    session.add_all(
        Prediction(market_id=m, side=side, line=line, model=model, fair_prob=p)
        for (m, side, line), p in predictions.items()
    )
    session.flush()
    return len(predictions)


def prop_side_probability(dist: StatDistribution, side: str, line: float | None) -> float | None:
    side = side.lower()
    if side in ("yes", "no"):  # anytime TD style: at least one
        p_yes = dist.over(0.5)
        return p_yes if side == "yes" else 1 - p_yes
    if line is None or side not in ("over", "under"):
        return None
    p_over = dist.over(line)
    return p_over if side == "over" else 1 - p_over


def predict_props(
    session: Session,
    stat_rows: list[dict],
    config: PropConfig | None = None,
    now: datetime | None = None,
) -> tuple[int, list[str]]:
    """Store prop-model probabilities for player props in the latest pull.

    Returns (predictions written, player names that couldn't be matched to nflverse).
    """
    now = now or utcnow()
    model = PropModel(config).fit(
        stat_rows, before=max(week_key(r["season"], r["week"]) for r in stat_rows) + 1
    )
    latest: dict[str, dict] = {}
    for r in sorted(stat_rows, key=lambda r: week_key(r["season"], r["week"])):
        latest[r["player_id"]] = r
    by_name: dict[str, list[dict]] = {}
    for r in latest.values():
        by_name.setdefault(normalize_name(r["player_display_name"]), []).append(r)

    predictions: dict[tuple[int, str, float | None], float] = {}
    unmatched: set[str] = set()
    for snap, market, event in latest_quotes(session):
        if not market.player or event.kickoff is None or as_utc(event.kickoff) <= now:
            continue
        teams = {abbr(event.home_team), abbr(event.away_team)}
        # Same name on two teams happens; the player must be on one of this game's teams.
        candidates = [
            r for r in by_name.get(normalize_name(market.player), []) if r["team"] in teams
        ]
        if len(candidates) != 1:
            unmatched.add(market.player)
            continue
        player = candidates[0]
        opponent = next(iter(teams - {player["team"]}), None)
        dist = model.predict(player["player_id"], market.market_type, opponent)
        if dist is None:
            continue
        prob = prop_side_probability(dist, snap.side, snap.line)
        if prob is not None:
            predictions[(market.id, snap.side, snap.line)] = prob
    return _store(session, PROP_MODEL_NAME, predictions), sorted(unmatched)


def load_predictions(
    session: Session, model: str | None = None
) -> dict[tuple[int, str, float | None], float]:
    """Stored predictions keyed by (market id, side, line); all models unless `model`."""
    query = select(Prediction)
    if model:
        query = query.where(Prediction.model == model)
    return {(p.market_id, p.side, p.line): p.fair_prob for p in session.scalars(query)}
