from datetime import timedelta

import pytest

from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import scan
from betmap.tables import utcnow

from .odds_fixtures import book, game, h2h, prop_event


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
