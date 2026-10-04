import httpx
import pytest
from sqlalchemy import func, select

from betmap.db import init_db, make_engine, session_scope
from betmap.odds.client import OddsApiClient, OddsApiError
from betmap.odds.ingest import ingest_events, pull_odds
from betmap.tables import Event, Market, OddsSnapshot

from .odds_fixtures import game, prop_event


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def fake_client(handler) -> OddsApiClient:
    http = httpx.Client(base_url="https://api.test/v4", transport=httpx.MockTransport(handler))
    return OddsApiClient("key", http=http)


def quota_headers(remaining: int, last: int) -> dict:
    return {
        "x-requests-remaining": str(remaining),
        "x-requests-used": "10",
        "x-requests-last": str(last),
    }


def test_ingest_creates_events_markets_snapshots(session):
    n_events, n_snaps = ingest_events(session, [game()])
    assert (n_events, n_snaps) == (1, 3 * 4 + 2)
    event = session.scalars(select(Event)).one()
    assert (event.home_team, event.odds_api_id) == ("Buffalo Bills", "evt1")
    assert {m.market_type for m in session.scalars(select(Market))} == {"h2h", "spreads"}


def test_repeat_ingest_reuses_event_and_markets(session):
    ingest_events(session, [game()])
    ingest_events(session, [game()])
    assert session.scalar(select(func.count(Event.id))) == 1
    assert session.scalar(select(func.count(Market.id))) == 2
    assert session.scalar(select(func.count(OddsSnapshot.id))) == 28


def test_props_are_keyed_by_player(session):
    ingest_events(session, [prop_event()])
    market = session.scalars(select(Market)).one()
    assert (market.market_type, market.player) == ("player_pass_yds", "Josh Allen")
    sides = {s.side for s in session.scalars(select(OddsSnapshot))}
    assert sides == {"Over", "Under"}


def test_pull_game_lines_and_props(session):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/events"):
            return httpx.Response(200, json=[{"id": "evt1"}], headers=quota_headers(400, 0))
        if path.endswith("/events/evt1/odds"):
            return httpx.Response(200, json=prop_event(), headers=quota_headers(399, 1))
        return httpx.Response(200, json=[game()], headers=quota_headers(397, 3))

    client = fake_client(handler)
    n_events, n_snaps = pull_odds(session, client, props=("player_pass_yds",))
    assert (n_events, n_snaps) == (1, 14 + 8)
    assert client.credits_spent == 4 and client.quota.remaining == 399
    odds_call = calls[0]
    assert odds_call.url.params["markets"] == "h2h,spreads,totals"
    assert odds_call.url.params["oddsFormat"] == "decimal"
    assert odds_call.url.params["apiKey"] == "key"


def test_api_error_surfaces_message(session):
    client = fake_client(lambda r: httpx.Response(401, json={"message": "Invalid api key"}))
    with pytest.raises(OddsApiError, match="401: Invalid api key"):
        pull_odds(session, client)


def test_missing_key():
    with pytest.raises(OddsApiError, match="not set"):
        OddsApiClient("")
