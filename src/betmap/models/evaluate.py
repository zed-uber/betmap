"""Forward test: stored pre-game predictions vs the closing market, on final games.

This is the test that matters for props, since there are no free historical prop lines:
every `model predict` before kickoff adds to it. For each predicted side and line whose
game is final, it compares the model's probability and the devigged closing consensus
against what happened. Lower Brier score is better; to have an edge, the model has to
beat the closing line, not just be decent.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.odds.scan import selection_label
from betmap.tables import BetStatus, Event, Market, Prediction
from betmap.tracking.clv import closing_quotes, consensus
from betmap.tracking.grading import Pick, grade_wager


@dataclass
class EvalGroup:
    model_probs: list[float] = field(default_factory=list)
    market_probs: list[float] = field(default_factory=list)
    hits: list[bool] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.hits)

    @property
    def brier_model(self) -> float:
        return fmean((p - h) ** 2 for p, h in zip(self.model_probs, self.hits, strict=True))

    @property
    def brier_market(self) -> float:
        return fmean((p - h) ** 2 for p, h in zip(self.market_probs, self.hits, strict=True))


@dataclass
class Evaluation:
    groups: dict[tuple[str, str], EvalGroup] = field(default_factory=lambda: defaultdict(EvalGroup))
    pending: int = 0  # game not final, or box score not in yet
    no_close: int = 0  # no closing consensus to compare against


def evaluate(session: Session) -> Evaluation:
    """Score every stored prediction whose outcome is known, grouped by (model, market)."""
    result = Evaluation()
    # One side per (model, market, line): the other side is its complement, and counting
    # both would double every observation.
    seen: set[tuple[str, int, float | None]] = set()
    for pred in session.scalars(select(Prediction).order_by(Prediction.id)):
        market = session.get(Market, pred.market_id)
        event: Event = market.event
        line_id = (pred.model, market.id, abs(pred.line) if pred.line is not None else None)
        if line_id in seen:
            continue
        pick = Pick(market.market_type, selection_label(market, pred.side, event), pred.line)
        outcome = grade_wager(session, pick, event)
        if outcome.status is None:
            result.pending += 1
            continue
        if outcome.status not in (BetStatus.WIN, BetStatus.LOSS):
            seen.add(line_id)  # a push says nothing about either side
            continue
        market_p, _ = consensus(
            closing_quotes(session, market, event), market, event, pred.side, pred.line
        )
        if market_p is None:
            result.no_close += 1
            continue
        seen.add(line_id)
        group = result.groups[(pred.model, market.market_type)]
        group.model_probs.append(pred.fair_prob)
        group.market_probs.append(market_p)
        group.hits.append(outcome.status == BetStatus.WIN)
    return result
