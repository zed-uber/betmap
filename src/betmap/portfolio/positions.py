"""Turn open bets and scan opportunities into positions the portfolio can reason about."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.models.prop_model import POSITION_GROUP
from betmap.odds.scan import Opportunity
from betmap.portfolio.correlation import Leg, joint_probability
from betmap.tables import Bet, BetLeg, BetStatus, Event, Market, PlayerGameStat
from betmap.teams import abbr, matchup
from betmap.tracking.grading import (
    find_event,
    normalize_name,
    parse_prop_selection,
    team_side,
)

# Without a fair probability, assume the price carries a typical two-way margin.
ASSUMED_OVERROUND = 1.045


@dataclass
class Position:
    label: str
    game_label: str
    leg: Leg
    price: float  # decimal
    prob: float  # fair win probability (pushes ignored)
    prob_source: str  # "fair", "closing", "scan", or "assumed"
    stake: float = 0.0  # already staked (open bets); 0 for candidates
    bet_id: int | None = None
    opportunity: Opportunity | None = None
    # Parlays: every (leg, fair prob) that must win; `leg` is the first, `prob` the joint.
    parts: list[tuple[Leg, float]] | None = None

    @property
    def components(self) -> list[tuple[Leg, float]]:
        return self.parts if self.parts else [(self.leg, self.prob)]


def _player_info(session: Session, player: str, event: Event) -> tuple[int, str | None]:
    """(team sign, position group) from the player's latest box score with either team."""
    teams = {abbr(event.home_team): 1, abbr(event.away_team): -1}
    wanted = normalize_name(player)
    rows = session.scalars(
        select(PlayerGameStat)
        .where(PlayerGameStat.team.in_(teams))
        .order_by(PlayerGameStat.season.desc(), PlayerGameStat.week.desc())
    )
    for row in rows:
        if normalize_name(row.player_name) == wanted:
            return teams[row.team], POSITION_GROUP.get(row.position or "")
    return 0, None


def make_leg(session: Session, event: Event, market: str, selection: str) -> Leg | None:
    """Interpret a selection ('BUF', 'Over', 'WAS Over', 'Josh Allen Over') as a Leg."""
    base = market.removeprefix("alternate_")
    words = selection.strip().split()
    if market.startswith("player_"):
        parsed = parse_prop_selection(selection)
        if parsed is None:
            return None
        player, side = parsed
        team_sign, group = _player_info(session, player, event)
        direction = 1 if side in ("over", "yes") else -1
        return Leg(event.id, market, direction, team_sign, normalize_name(player), group)
    if base.startswith("team_totals") and len(words) >= 2:
        team = team_side(" ".join(words[:-1]), event)
        if team is None or words[-1].lower() not in ("over", "under"):
            return None
        direction = 1 if words[-1].lower() == "over" else -1
        return Leg(event.id, market, direction, 1 if team == "home" else -1)
    if base.startswith("totals"):
        side = selection.strip().lower()
        if side not in ("over", "under"):
            return None
        return Leg(event.id, market, 1 if side == "over" else -1)
    team = team_side(selection, event)
    if team is None:
        return None
    return Leg(event.id, market, 1 if team == "home" else -1)


def open_positions(session: Session) -> tuple[list[Position], list[Bet]]:
    """Open bets as positions; also returns bets that couldn't be interpreted."""
    positions, skipped = [], []
    for bet in session.scalars(select(Bet).where(Bet.status == BetStatus.OPEN)):
        if bet.is_parlay:
            position = parlay_position(session, bet)
            if position is None:
                skipped.append(bet)
            else:
                positions.append(position)
            continue
        event = find_event(session, bet)
        leg = event and make_leg(session, event, bet.market_type, bet.selection)
        if leg is None:
            skipped.append(bet)
            continue
        if bet.closing_fair_prob is not None:
            prob, source = bet.closing_fair_prob, "closing"
        elif bet.fair_prob is not None:
            prob, source = bet.fair_prob, "fair"
        else:
            prob, source = 1 / bet.price / ASSUMED_OVERROUND, "assumed"
        positions.append(
            Position(
                label=f"#{bet.id} {bet.selection} {bet.market_type}"
                + ("" if bet.line is None else f" {bet.line:g}"),
                game_label=matchup(event.away_team, event.home_team),
                leg=leg,
                price=bet.price,
                prob=prob,
                prob_source=source,
                stake=bet.stake,
                bet_id=bet.id,
            )
        )
    return positions, skipped


def leg_probability(leg: BetLeg) -> float:
    if leg.closing_fair_prob is not None:
        return leg.closing_fair_prob
    if leg.fair_prob is not None:
        return leg.fair_prob
    return 1 / leg.price / ASSUMED_OVERROUND if leg.price else 0.5


def parlay_position(session: Session, bet: Bet) -> Position | None:
    """An open parlay as one position over its still-open legs (legs already won drop out)."""
    parts, games = [], []
    for leg in bet.legs:
        if leg.status != BetStatus.OPEN:
            continue
        event = session.get(Event, leg.event_id) if leg.event_id else None
        made = event and make_leg(session, event, leg.market_type, leg.selection)
        if made is None:
            return None
        parts.append((made, leg_probability(leg)))
        label = matchup(event.away_team, event.home_team)
        if label not in games:
            games.append(label)
    if not parts:
        return None
    sources = {
        "closing"
        if leg.closing_fair_prob is not None
        else "fair"
        if leg.fair_prob is not None
        else "assumed"
        for leg in bet.legs
    }
    return Position(
        label=f"#{bet.id} {bet.selection}",
        game_label=" / ".join(games),
        leg=parts[0][0],
        price=bet.price,
        prob=joint_probability(parts),
        prob_source="assumed" if "assumed" in sources else "fair",
        stake=bet.stake,
        bet_id=bet.id,
        parts=parts,
    )


def candidate_positions(session: Session, opportunities: list[Opportunity]) -> list[Position]:
    """Scan opportunities as candidates, keeping the best price per side and line."""
    best: dict[tuple[int, str, float | None], Opportunity] = {}
    for o in opportunities:
        key = (o.market_id, o.side, o.line)
        if key not in best or o.ev > best[key].ev:
            best[key] = o
    positions = []
    for o in best.values():
        market = session.get(Market, o.market_id)
        leg = make_leg(session, market.event, o.market_type, o.selection)
        if leg is None:
            continue
        line = "" if o.line is None else f" {o.line:g}"
        positions.append(
            Position(
                label=f"{o.selection} {o.market_type}{line} @ {o.book}",
                game_label=o.event_label,
                leg=leg,
                price=o.price,
                prob=o.fair_prob,
                prob_source="scan",
                opportunity=o,
            )
        )
    return positions
