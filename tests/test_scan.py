from datetime import timedelta

import pytest

from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import scan
from betmap.tables import utcnow

from .odds_fixtures import book, game, h2h, prop_event, spreads


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def test_soft_price_against_consensus(session):
    ingest_events(session, [game()])
    opps = scan(session)
    assert len(opps) == 1
    o = opps[0]
    assert (o.book, o.selection, o.line, o.market_type) == ("soft", "BUF", -2.5, "spreads")
    assert o.event_label == "KC @ BUF"
    # Leave-one-out: soft is scored against the other three, which are exactly 50/50.
    assert o.n_books == 3
    assert o.fair_prob == pytest.approx(0.5)
    assert o.ev == pytest.approx(o.fair_prob * 2.05 - 1)
    assert o.best_other == 1.91
    assert 0 < o.kelly <= 0.03


def test_spread_sides_pair_across_signs(session):
    # Away side quoted at +2.5 must pair with home at -2.5, not land in its own group.
    ingest_events(session, [game()])
    opps = scan(session, min_ev=-1)
    kc = [o for o in opps if o.selection == "KC" and o.market_type == "spreads"]
    assert {o.line for o in kc} == {2.5}
    assert len(kc) == 4 and all(o.n_books == 3 for o in kc)


def test_min_books_and_book_filter(session):
    ingest_events(session, [game()])
    assert scan(session, min_books=4) == []
    assert scan(session, books={"draftkings"}) == []
    assert [o.book for o in scan(session, books={"soft"})] == ["soft"]


def test_started_games_are_skipped(session):
    ingest_events(session, [game(hours=-1)])
    assert scan(session) == []
    assert scan(session, now=utcnow() - timedelta(hours=2))


def test_only_latest_pull_counts(session):
    ingest_events(session, [game()], fetched_at=utcnow() - timedelta(hours=1))
    fixed = [
        book(k, {"h2h": h2h(1.91, 1.91)}) for k in ("draftkings", "fanduel", "betmgm", "caesars")
    ]
    ingest_events(session, [game(bookmakers=fixed)])
    # The spreads market's latest pull is still the old one; h2h's is the new one.
    markets = {o.market_type for o in scan(session, min_ev=-1)}
    assert markets == {"h2h", "spreads"}
    assert not [o for o in scan(session) if o.market_type == "h2h"]


def test_props(session):
    ingest_events(session, [prop_event()])
    opps = scan(session)
    assert [(o.selection, o.book, o.line) for o in opps] == [("Josh Allen Over", "betmgm", 249.5)]


def test_exchange_prices_are_scored_after_fees(session):
    # Kalshi quotes BUF -2.5 at 2.05 (about 48.8c). Net of the 0.07 fee that's ~1.96.
    from betmap.odds.math import after_exchange_fee

    data = game()
    data["bookmakers"][-1]["key"] = "kalshi"  # the off-market book is now an exchange
    ingest_events(session, [data])
    [raw] = scan(session)
    assert raw.price == 2.05 and raw.ev == pytest.approx(0.025) and not raw.fee_adjusted
    # After the fee the edge is gone, so the default scan drops it.
    assert scan(session, fees={"kalshi": 0.07}) == []
    [net] = [
        o
        for o in scan(session, min_ev=-1, fees={"kalshi": 0.07})
        if o.book == "kalshi" and o.selection == "BUF"
    ]
    assert net.price == pytest.approx(after_exchange_fee(2.05, 0.07)) and net.fee_adjusted
    assert net.ev < 0 and net.fair_prob == pytest.approx(raw.fair_prob)


def test_pickem_books_are_not_scanned(session):
    from betmap.odds.scan import pickem_quotes

    data = game()
    # Underdog's pick looks like a huge edge, but it isn't a single bet you can make.
    data["bookmakers"].append(book("underdog", {"spreads": spreads(-2.5, 3.0, 1.3)}))
    ingest_events(session, [data])
    opps = scan(session, min_ev=-1)
    assert all(o.book != "underdog" for o in opps)
    assert all(o.n_books == 3 for o in opps)  # not part of the consensus either
    assert pickem_quotes(session) == 2


def test_unusable_exchange_quotes_stay_out_of_the_consensus(session):
    from betmap.odds.scan import usable_market

    assert usable_market([1.91, 1.91]) and not usable_market([1.0, 1.0])
    assert not usable_market([1.40, 1.50])  # ~38% margin: a thin order book, not a price
    data = game()
    data["bookmakers"] += [
        book("novig", {"spreads": spreads(-2.5, 1.0, 1.0)}),
        book("polymarket", {"spreads": spreads(-2.5, 1.40, 1.50)}),
    ]
    ingest_events(session, [data])
    opps = scan(session, min_ev=-1)  # used to crash devigging the 1.0 prices
    spread_opps = [o for o in opps if o.market_type == "spreads"]
    assert all(o.n_books == 3 for o in spread_opps if o.book in ("draftkings", "soft"))
    assert all(o.book != "novig" for o in spread_opps)  # a 1.0 price is never a bet
