import pytest

from betmap.db import make_engine, session_scope
from betmap.tables import BetStatus
from betmap.tracking import ledger


@pytest.fixture
def session():
    with session_scope(make_engine(":memory:")) as s:
        yield s


def test_bankroll_lifecycle(session):
    ledger.record_transfer(session, "deposit", 1000)
    win = ledger.place_bet(
        session,
        event_label="KC @ BUF",
        market_type="h2h",
        selection="BUF",
        book="dk",
        price=2.0,
        stake=100,
        fair_prob=0.55,
    )
    loss = ledger.place_bet(
        session,
        event_label="KC @ BUF",
        market_type="totals",
        selection="Over",
        line=47.5,
        book="fd",
        price=1.91,
        stake=50,
    )
    push = ledger.place_bet(
        session,
        event_label="DAL @ PHI",
        market_type="spreads",
        selection="PHI",
        line=-3,
        book="mgm",
        price=1.91,
        stake=20,
    )
    ledger.place_bet(
        session,
        event_label="DAL @ PHI",
        market_type="h2h",
        selection="DAL",
        book="dk",
        price=3.0,
        stake=10,
    )

    summary = ledger.summarize(session)
    assert summary.open_bets == 4
    assert summary.bankroll == pytest.approx(820)
    assert summary.equity == pytest.approx(1000)

    ledger.settle_bet(session, win.id, BetStatus.WIN, closing_price=1.8)
    ledger.settle_bet(session, loss.id, BetStatus.LOSS)
    ledger.settle_bet(session, push.id, BetStatus.PUSH)

    summary = ledger.summarize(session)
    assert summary.settled_profit == pytest.approx(50)
    assert (summary.wins, summary.losses, summary.pushes) == (1, 1, 1)
    assert summary.open_exposure == pytest.approx(10)
    assert summary.bankroll == pytest.approx(1040)
    assert summary.roi == pytest.approx(50 / 170)


def test_cannot_settle_twice(session):
    bet = ledger.place_bet(
        session, event_label="x", market_type="h2h", selection="A", book="dk", price=2.0, stake=10
    )
    ledger.settle_bet(session, bet.id, BetStatus.LOSS)
    with pytest.raises(ValueError):
        ledger.settle_bet(session, bet.id, BetStatus.WIN)


def test_rejects_bad_inputs(session):
    with pytest.raises(ValueError):
        ledger.record_transfer(session, "deposit", -5)
    with pytest.raises(ValueError):
        ledger.place_bet(
            session,
            event_label="x",
            market_type="h2h",
            selection="A",
            book="dk",
            price=0.9,
            stake=10,
        )


def test_edit_bet_changes_result_and_recomputes_payout(session):
    bet = ledger.place_bet(
        session, event_label="IND @ WAS", market_type="totals_h1", selection="Over",
        line=21.5, book="kalshi", price=1 / 0.57, stake=100,
    )  # fmt: skip
    ledger.settle_bet(session, bet.id, BetStatus.WIN)
    settled_at = bet.settled_at
    assert bet.profit == pytest.approx(75.44, abs=0.01)

    ledger.edit_bet(session, bet.id, status=BetStatus.LOSS)
    assert (bet.status, bet.payout, bet.profit) == (BetStatus.LOSS, 0.0, -100)
    assert bet.settled_at == settled_at

    ledger.edit_bet(session, bet.id, stake=50, status=BetStatus.PUSH)
    assert bet.payout == 50 and bet.profit == 0

    ledger.edit_bet(session, bet.id, status=BetStatus.OPEN)
    assert bet.status == BetStatus.OPEN and bet.payout is None and bet.settled_at is None


def test_edit_bet_identity_change_clears_links(session):
    bet = ledger.place_bet(
        session, event_label="KC @ BUF", market_type="spreads", selection="BUF", line=-2.5,
        book="dk", price=1.91, stake=10,
    )  # fmt: skip
    bet.event_id, bet.closing_fair_prob, bet.closing_price = 7, 0.52, 1.87
    ledger.edit_bet(session, bet.id, notes="price only", price=1.95)
    assert bet.event_id == 7 and bet.closing_fair_prob == 0.52  # not an identity change
    ledger.edit_bet(session, bet.id, line=-3.0)
    assert bet.event_id is None and bet.closing_fair_prob is None and bet.closing_price is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"stake": 0}, "stake must be positive"),
        ({"price": 1.0}, "decimal odds"),
        ({"fair_prob": 1.5}, "between 0 and 1"),
        ({"status": "lost"}, "not a valid"),
        ({"placed_at": None}, "can't edit placed_at"),
        ({"source": "synergy-v1"}, "can't edit source"),  # fixed once placed
        ({"slate_id": None}, "can't edit slate_id"),
    ],
)
def test_edit_bet_rejects_bad_values(session, changes, message):
    bet = ledger.place_bet(
        session, event_label="x", market_type="h2h", selection="A", book="dk", price=2.0, stake=10
    )
    assert bet.source == "manual"
    with pytest.raises(ValueError, match=message):
        ledger.edit_bet(session, bet.id, **changes)
    with pytest.raises(ValueError, match="no bet with id 99"):
        ledger.edit_bet(session, 99, stake=5)


def test_cli_bet_edit(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from betmap.cli import app
    from betmap.config import get_settings

    monkeypatch.setenv("BETMAP_DB_PATH", str(tmp_path / "cli.db"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        runner.invoke(
            app,
            [
                "bet",
                "add",
                "--event",
                "KC @ BUF",
                "--market",
                "h2h",
                "--selection",
                "BUF",
                "--odds",
                "-110",
                "--stake",
                "50",
                "--book",
                "dk",
            ],
        )
        runner.invoke(app, ["bet", "settle", "1", "win"])
        r = runner.invoke(app, ["bet", "edit", "1", "--status", "loss", "--odds", "+100"])
        assert r.exit_code == 0 and "+100" in r.output and "loss, -50.00" in r.output
        r = runner.invoke(app, ["bet", "edit", "1"])
        assert r.exit_code == 1 and "Nothing to change" in r.output
        r = runner.invoke(app, ["bet", "edit", "1", "--stake", "-5"])
        assert r.exit_code == 1 and "stake must be positive" in r.output
    finally:
        get_settings.cache_clear()
