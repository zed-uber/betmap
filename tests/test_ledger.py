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
