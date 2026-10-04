import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from betmap.web.app import create_app


@pytest.fixture
def client():
    # One shared in-memory connection so every request sees the same DB.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    return TestClient(create_app(engine))


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
