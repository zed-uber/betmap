"""One step to bring results up to date: sync nflverse, record closing lines, settle bets."""

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from betmap.data import nflverse
from betmap.tracking.clv import fill_closing_lines
from betmap.tracking.grading import GradeOutcome, settle_open_bets

Fetch = Callable[[list[int]], tuple[list[dict], list[dict]]]


@dataclass
class ResultsUpdate:
    games: int
    player_lines: int
    closing_lines: int
    outcomes: list[GradeOutcome] = field(default_factory=list)

    @property
    def graded(self) -> list[GradeOutcome]:
        return [o for o in self.outcomes if o.status is not None]

    @property
    def needs_manual(self) -> list[GradeOutcome]:
        return [o for o in self.outcomes if o.needs_manual]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for o in self.graded:
            counts[o.status] = counts.get(o.status, 0) + 1
        text = f"Synced {self.games} games. Settled {len(self.graded)} bet"
        text += "" if len(self.graded) == 1 else "s"
        if counts:
            text += " (" + ", ".join(f"{n} {s}" for s, n in counts.items()) + ")"
        text += f"; recorded {self.closing_lines} closing line"
        text += "" if self.closing_lines == 1 else "s"
        if self.needs_manual:
            text += ". Settle manually: " + "; ".join(
                f"#{o.bet.id} {o.reason}" for o in self.needs_manual
            )
        return text + "."


def update_results(
    session: Session,
    fetch: Fetch = nflverse.fetch,
    seasons: list[int] | None = None,
    dry_run: bool = False,
) -> ResultsUpdate:
    games, stats = fetch(seasons or [nflverse.current_season()])
    n_games = nflverse.sync_games(session, games)
    n_stats = nflverse.sync_player_stats(session, stats)
    n_close = 0 if dry_run else fill_closing_lines(session)
    outcomes = settle_open_bets(session, dry_run=dry_run)
    return ResultsUpdate(n_games, n_stats, n_close, outcomes)
