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

from .odds_fixtures import book, game


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


def builder_client():
    from betmap.db import session_scope
    from betmap.odds.ingest import ingest_events

    from .test_builder import second_game

    engine = shared_engine()
    client = TestClient(create_app(engine))
    with session_scope(engine) as s:
        ingest_events(s, [game(), second_game()])
    client.post("/bankroll", data={"kind": "deposit", "amount": "1000"})
    return client, engine


def board_form(client, selection_text, kind, parlay=""):
    """The hidden fields of the board row's add form for a selection, e.g. 'BUF -2.5'."""
    import re

    page = client.get("/builder").text
    rows = page.split("<tr>")
    row = next(r for r in rows if f"<td>{selection_text}</td>" in r)
    fields = dict(
        re.findall(
            r'name="(\w+)" value="([^"]*)"', row.split(f'value="{kind}"')[0].rsplit("<form", 1)[1]
        )
    )
    fields.update({"kind": kind, "parlay": parlay, "back": "/builder"})
    return unescape_fields(fields)


def unescape_fields(fields):
    return {k: unescape(v) for k, v in fields.items()}


def test_builder_flow():
    client, _ = builder_client()
    r = client.get("/builder")
    assert "No slate yet" in r.text and "BUF +2.5" not in r.text  # spreads use signed lines
    assert "BUF -2.5" in r.text and "NYJ" in r.text

    r = client.post("/slates", data={"name": "Sunday main", "back": "/builder"})
    assert "Created slate" in r.text and "Sunday main" in r.text and "+ Parlay" in r.text

    r = client.post("/slates/1/add", data=board_form(client, "BUF -2.5", "straight"))
    assert "Added BUF as a straight bet" in r.text and "soft" in r.text

    # Start a same-game parlay and add a second leg to it.
    r = client.post("/slates/1/add", data=board_form(client, "BUF", "parlay"))
    assert "parlay=2" in str(r.url) and "Building a parlay (1 leg)" in r.text
    r = client.post("/slates/1/add", data=board_form(client, "BUF -2.5", "parlay", parlay="2"))
    assert "(2 legs)" in r.text and "enter the price the book quotes" in r.text
    assert "Correlated legs" in r.text

    r = client.post("/slates/items/2", data={"stake": "10", "offered": "+160", "book": "FanDuel",
                                             "back": "/builder?slate=1"})  # fmt: skip
    assert "Saved" in r.text and 'value="+160"' in r.text and 'value="fanduel"' in r.text
    assert "enter the price" not in r.text

    r = client.post("/slates/items/2", data={"stake": "x", "back": "/builder?slate=1"})
    assert "Couldn" in r.text
    # The +EV SGP already holds BUF -2.5, so joint sizing suggests nothing more on the
    # straight; enter a stake to bet it anyway.
    assert 'placeholder="0.00 suggested"' in client.get("/builder?slate=1").text
    client.post("/slates/items/1", data={"stake": "15", "back": "/builder?slate=1"})

    # Duplicate, then compare both.
    r = client.post("/slates/1/duplicate", data={"back": "/builder?slate=1"})
    assert "Copied to" in r.text
    r = client.get("/slates?compare=1&compare=2")
    assert "Comparison" in r.text and r.text.count('href="/builder?slate=') >= 4
    assert "Growth" in r.text

    r = client.post("/slates/1/place", data={"back": "/builder?slate=1"})
    assert r.url.path == "/bets" and "Placed &#39;Sunday main&#39;: 2 bets" in r.text
    assert "2-leg: BUF + BUF -2.5" in r.text and "fanduel" in r.text
    assert r.text.count("· Sunday main") == 2  # each bet shows the slate it came from
    r = client.post("/slates/1/delete", data={"back": "/slates"})
    assert "Couldn&#39;t delete" in r.text and "already placed" in r.text
    r = client.post(
        "/slates/1/add", data=board_form(client, "NYJ", "straight") | {"back": "/builder?slate=1"}
    )
    assert "already placed" in r.text


def test_builder_errors():
    client, _ = builder_client()
    client.post("/slates", data={"name": "S", "back": "/builder"})
    gone = {
        "market_id": "999",
        "side": "Nobody",
        "line": "",
        "kind": "straight",
        "back": "/builder",
    }
    assert "no longer on the board" in client.post("/slates/1/add", data=gone).text
    assert "needs a name" in client.post("/slates", data={"name": " ", "back": "/builder"}).text
    r = client.post("/slates/1/place", data={"back": "/builder?slate=1"})
    assert "nothing to place" in r.text
    # The back link can't send you off-site.
    r = client.post("/slates/1/rename", data={"name": "T", "back": "https://evil.example/x"})
    assert r.url.path == "/builder" and "Renamed" in r.text
    r = client.post("/slates/1/delete", data={"back": "/slates"})
    assert "Deleted" in r.text and "No drafts" in r.text


def test_builder_shows_second_opinions():
    from betmap.db import session_scope
    from betmap.odds.ingest import ingest_events

    from .odds_fixtures import h2h

    engine = shared_engine()
    client = TestClient(create_app(engine))
    data = game()
    data["bookmakers"].append(book("pinnacle", {"h2h": h2h(1.70, 2.25)}))
    with session_scope(engine) as s:
        ingest_events(s, [data])
    page = client.get("/builder?market=h2h").text
    assert 'class="op' in page and "Pinnacle 57" in page and "Opinions" in page
