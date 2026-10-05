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


def test_portfolio_page():
    client = TestClient(create_app(shared_engine(), odds_client=fake_odds_client))
    r = client.get("/portfolio")
    assert r.status_code == 200 and "No scan prices" in r.text

    client.post("/bankroll", data={"kind": "deposit", "amount": "1000"})
    client.post("/odds/pull")
    r = client.get("/portfolio")
    assert "BUF spreads -2.5 @ soft" in r.text and 'href="/?' in r.text  # sized, with Log

    # An open BUF moneyline already carries that risk: the spread is sized to 0.
    client.post("/bets", data=bet_form(market="h2h", line="", fair_prob="0.55"))
    r = client.get("/portfolio")
    assert 'href="/?' not in r.text and '<span class="muted">0</span>' in r.text
    assert "KC @ BUF" in r.text and "50.00" in r.text  # at risk by game

    # Two open bets on the same side of one game are flagged as overlapping.
    client.post("/bets", data=bet_form(fair_prob="0.55"))
    r = client.get("/portfolio")
    assert "Overlapping bets" in r.text and "#1 BUF h2h" in r.text and "#2 BUF spreads" in r.text


def test_settle_requires_a_result(client):
    client.post("/bets", data=bet_form())
    r = client.get("/")
    assert '<option value="" selected disabled>result' in r.text
    r = client.post("/bets/1/settle", data={"result": ""})
    assert "pick win, loss, push, or void" in r.text
    assert "No open bets" not in r.text  # still open


def test_edit_bet_page_and_save(client):
    client.post(
        "/bets",
        data=bet_form(
            market="totals_h1",
            selection="Over",
            line="21.5",
            odds="57c",
            stake="100",
            book="kalshi",
        ),
    )
    client.post("/bets/1/settle", data={"result": "win"})  # the mistake

    r = client.get("/bets")
    assert 'href="/bets/1/edit"' in r.text
    r = client.get("/bets/1/edit")
    assert 'value="totals_h1"' in r.text and 'value="21.5"' in r.text
    assert '<option value="win" selected>' in r.text

    form = bet_form(
        market="totals_h1", selection="Over", line="21.5", odds="57c", stake="100", book="kalshi"
    ) | {"status": "loss"}
    r = client.post("/bets/1/edit", data=form)
    assert "Saved #1: Over (loss -100.00)" in r.text
    assert 'class="status status-loss"' in r.text

    r = client.post("/bets/1/edit", data=form | {"status": "open"})
    assert "Saved #1: Over (open)" in r.text


def test_edit_bet_bad_input_keeps_values(client):
    client.post("/bets", data=bet_form())
    form = bet_form(odds="57%", notes="typo here") | {"status": "open"}
    r = client.post("/bets/1/edit", data=form)
    assert r.url.path == "/bets/1/edit"
    assert "Couldn" in r.text and "odds must be" in r.text
    assert 'name="odds" value="57%" aria-invalid="true"' in unescape(r.text)
    assert 'value="typo here"' in r.text
    assert "-105" in client.get("/bets").text  # unchanged


def test_edit_missing_bet(client):
    assert "No bet #9" in client.get("/bets/9/edit").text


def test_parlay_shows_legs_on_bets_and_edit_pages():
    from sqlalchemy.orm import Session as OrmSession

    from betmap.tracking import ledger

    engine = shared_engine()
    client = TestClient(create_app(engine))
    with OrmSession(engine) as s:
        legs = [
            {"market_type": "h2h", "selection": "BUF", "price": 1.8},
            {"market_type": "totals", "selection": "Over", "line": 47.5, "price": 1.91},
        ]
        ledger.place_parlay(s, legs=legs, book="fanduel", price=3.44, stake=10)
        s.commit()
    r = client.get("/bets")
    assert '<details class="legs"><summary>2-leg: BUF + Over 47.5</summary>' in r.text
    assert "Over 47.5" in r.text and "status-open" in r.text
    r = client.get("/bets/1/edit")
    assert "Parlay legs (read-only" in r.text
