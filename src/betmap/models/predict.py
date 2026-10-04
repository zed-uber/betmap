"""Store game-model probabilities for every line in the latest odds pull."""

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from betmap.data import nflverse
from betmap.models.game_model import GameModel, GamePrediction, ModelConfig
from betmap.odds.scan import latest_quotes
from betmap.tables import Event, Prediction, as_utc, utcnow
from betmap.teams import abbr

MODEL_NAME = "ratings-v1"
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

    market_ids = {m for m, _, _ in predictions}
    if market_ids:
        session.execute(
            delete(Prediction).where(
                Prediction.model == MODEL_NAME, Prediction.market_id.in_(market_ids)
            )
        )
    session.add_all(
        Prediction(market_id=m, side=side, line=line, model=MODEL_NAME, fair_prob=p)
        for (m, side, line), p in predictions.items()
    )
    session.flush()
    return len(predictions)


def load_predictions(
    session: Session, model: str = MODEL_NAME
) -> dict[tuple[int, str, float | None], float]:
    return {
        (p.market_id, p.side, p.line): p.fair_prob
        for p in session.scalars(select(Prediction).where(Prediction.model == model))
    }
