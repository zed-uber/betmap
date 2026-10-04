"""Store Odds API responses as Event / Market / OddsSnapshot rows."""

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.data.nflverse import find_same_game
from betmap.odds.client import GAME_MARKETS, OddsApiClient
from betmap.tables import Event, Market, OddsSnapshot, utcnow


def _upsert_event(session: Session, data: dict) -> Event:
    kickoff = datetime.fromisoformat(data["commence_time"])
    event = session.scalars(select(Event).where(Event.odds_api_id == data["id"])).first()
    if event is None:
        # The game may already exist from an nflverse schedule sync.
        event = (
            find_same_game(session, data["home_team"], data["away_team"], kickoff, odds_api_id=None)
            or Event()
        )
        event.odds_api_id = data["id"]
        session.add(event)
    event.home_team = data["home_team"]
    event.away_team = data["away_team"]
    event.kickoff = kickoff
    return event


def _get_market(session: Session, event: Event, market_type: str, player: str | None) -> Market:
    query = select(Market).where(
        Market.event_id == event.id, Market.market_type == market_type, Market.player == player
    )
    market = session.scalars(query).first()
    if market is None:
        market = Market(event=event, market_type=market_type, player=player)
        session.add(market)
        session.flush()
    return market


def ingest_event(session: Session, data: dict, fetched_at: datetime | None = None) -> int:
    """Store one event's odds; returns the number of price snapshots written."""
    fetched_at = fetched_at or utcnow()
    event = _upsert_event(session, data)
    session.flush()
    count = 0
    for bookmaker in data.get("bookmakers", []):
        for market_data in bookmaker["markets"]:
            for outcome in market_data["outcomes"]:
                # Player props carry the player in `description`; sides are Over/Under.
                market = _get_market(session, event, market_data["key"], outcome.get("description"))
                session.add(
                    OddsSnapshot(
                        market_id=market.id,
                        book=bookmaker["key"],
                        side=outcome["name"],
                        line=outcome.get("point"),
                        price=outcome["price"],
                        fetched_at=fetched_at,
                    )
                )
                count += 1
    session.flush()
    return count


def pull_odds(
    session: Session,
    client: OddsApiClient,
    *,
    markets: tuple[str, ...] = GAME_MARKETS,
    props: tuple[str, ...] = (),
    regions: str = "us",
    days: float = 7,
) -> tuple[int, int]:
    """Fetch and store game lines, plus props per event if requested; returns (events, snapshots)."""
    fetched_at = utcnow()
    n_events, n_snaps = 0, 0
    if markets:
        n_events, n_snaps = ingest_events(
            session, client.game_odds(markets, regions, days), fetched_at
        )
    if props:
        events = client.events(days)
        for data in events:
            n_snaps += ingest_event(
                session, client.event_odds(data["id"], props, regions), fetched_at
            )
        n_events = max(n_events, len(events))
    return n_events, n_snaps


def ingest_events(
    session: Session, events: Iterable[dict], fetched_at: datetime | None = None
) -> tuple[int, int]:
    """Store a list of events; returns (events, snapshots)."""
    fetched_at = fetched_at or utcnow()
    n_events = n_snaps = 0
    for data in events:
        n_snaps += ingest_event(session, data, fetched_at)
        n_events += 1
    return n_events, n_snaps
