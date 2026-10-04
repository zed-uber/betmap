from html import unescape
from urllib.parse import parse_qsl

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from betmap.odds.client import OddsApiClient
from betmap.tables import Bet
from betmap.web.app import create_app

from .odds_fixtures import game


def shared_engine():
    # One shared in-memory connection so every request sees the same DB.
    return create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


@pytest.fixture
def client():
    return TestClient(create_app(shared_engine()))


def test_dashboard_empty(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "No open bets" in r.text


def test_deposit_bet_settle_flow(client):
    r = client.post("/bankroll", data={"kind": "deposit", "amount": "1000"})
    assert r.status_code == 200 and "Recorded deposit" in r.text

    r = client.post(
        "/bets",
        data={
            "event": "KC @ BUF",
            "market": "spreads",
            "selection": "BUF",
            "line": "-2.5",
            "odds": "-105",
            "stake": "50",
            "book": "dk",
            "fair_prob": "54",
        },
    )
    assert "Logged #1" in r.text and "EV +5.4%" in r.text
    assert "950.00" in r.text  # bankroll after stake is at risk

    r = client.post("/bets/1/settle", data={"result": "win", "closing_odds": "-120"})
    assert "Settled #1 as win: +47.62" in r.text

    r = client.get("/bets")
    assert "+6.5%" in r.text  # CLV column


def test_bad_input_shows_error(client):
    r = client.post(
        "/bets",
        data={
            "event": "x",
            "market": "h2h",
            "selection": "A",
            "odds": "abc",
            "stake": "10",
            "book": "dk",
        },
    )
    assert r.status_code == 200
    assert "Couldn&#39;t log bet" in r.text or "Couldn't log bet" in r.text

    r = client.post("/bets/99/settle", data={"result": "win"})
    assert "no bet with id 99" in r.text


def bet_form(**overrides: str) -> dict:
    form = {
        "event": "KC @ BUF",
        "market": "spreads",
        "selection": "BUF",
        "line": "-2.5",
        "odds": "-105",
        "stake": "50",
        "book": "dk",
        "fair_prob": "54",
        "notes": "rested starters",
    }
    return form | overrides


@pytest.mark.parametrize(
    ("field", "bad"),
    [("odds", "57%"), ("stake", "-5"), ("line", "abc"), ("fair_prob", "150")],
)
def test_bad_field_keeps_input_and_flags_field(client, field, bad):
    r = client.post("/bets", data=bet_form(**{field: bad}))
    assert r.url.fragment == "log"
    assert "Couldn" in r.text
    for name, value in bet_form(**{field: bad}).items():
        assert f'name="{name}" value="{value}"' in unescape(r.text)
    assert r.text.count('aria-invalid="true"') == 1
    assert f'name="{field}" value="{bad}" aria-invalid="true"' in unescape(r.text)
    assert "No open bets" in r.text  # nothing was logged


def test_fair_prob_accepts_percent_sign(client):
    r = client.post("/bets", data=bet_form(fair_prob="54%"))
    assert "Logged #1" in r.text and "EV +5.4%" in r.text


def test_bets_filter(client):
    client.post(
        "/bets",
        data={
            "event": "x",
            "market": "h2h",
            "selection": "A",
            "odds": "+120",
            "stake": "10",
            "book": "dk",
        },
    )
    assert "No bets with status" in client.get("/bets?status=win").text
    assert "+120" in client.get("/bets?status=open").text


def fake_odds_client() -> OddsApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"x-requests-remaining": "497", "x-requests-last": "3"}
        return httpx.Response(200, json=[game()], headers=headers)

    http = httpx.Client(base_url="https://api.test/v4", transport=httpx.MockTransport(handler))
    return OddsApiClient("key", http=http)


def test_scan_empty(client):
    r = client.get("/scan")
    assert r.status_code == 200 and "Pull odds to scan" in r.text


def test_pull_scan_log_flow():
    engine = shared_engine()
    client = TestClient(create_app(engine, odds_client=fake_odds_client))
    client.post("/bankroll", data={"kind": "deposit", "amount": "1000"})

    r = client.post("/odds/pull")
    assert "Stored 14 prices across 1 games (3 credits, 497 left)" in r.text
    assert "KC @ BUF" in r.text and "soft" in r.text and "+105" in r.text

    # Follow the "Log" link: the dashboard form comes back prefilled.
    query = unescape(r.text.split('href="/?', 1)[1].split("#", 1)[0])
    r = client.get("/?" + query)
    assert 'value="BUF"' in r.text and 'value="+105"' in r.text and 'value="-2.5"' in r.text

    r = client.post("/bets", data=dict(parse_qsl(query)))
    assert "Logged #1: BUF +105" in r.text and "EV +2.5%" in r.text
    with Session(engine) as s:
        assert s.scalars(select(Bet)).one().market_id is not None


def test_pull_error_is_flashed():
    client = TestClient(create_app(shared_engine(), odds_client=lambda: OddsApiClient("")))
    r = client.post("/odds/pull")
    assert "BETMAP_ODDS_API_KEY is not set" in r.text


def test_sync_results_settles_and_reports():
    from datetime import timedelta

    from betmap.tables import utcnow

    from .test_results import schedule_row

    kickoff = utcnow() - timedelta(hours=5)

    def fetch(seasons):
        return [schedule_row(kickoff=kickoff, score=(20, 24))], []

    client = TestClient(create_app(shared_engine(), results_fetch=fetch))
    client.post("/bets", data=bet_form(market="h2h", line="", notes=""))
    client.post("/bets", data=bet_form(market="player_pass_yds", selection="Josh Allen Over"))
    r = client.post("/results/sync")
    assert "Settled 1 bet (1 win)" in r.text
    # The prop is waiting on a box score: still open, and not flagged as a problem.
    assert "flash-error" not in r.text and "Josh Allen Over" in r.text
    assert 'class="status status-win"' in r.text


def test_sync_results_reports_missing_stats_extra():
    from betmap.data.nflverse import StatsUnavailable

    def fetch(seasons):
        raise StatsUnavailable("nflverse sync needs the stats extra")

    client = TestClient(create_app(shared_engine(), results_fetch=fetch))
    assert "Couldn" in client.post("/results/sync").text
