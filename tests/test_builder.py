import pytest
from sqlalchemy import select

from betmap.builder.pricing import find_entry, price_parlay
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import board, scan
from betmap.tables import Event

from .odds_fixtures import book, game, h2h


def second_game(event_id="evt2", hours=50):
    """NE @ NYJ moneyline at the three main books; NYJ is a slight favorite."""
    data = game(event_id=event_id, hours=hours, bookmakers=[
        book(k, {"h2h": [
            {"name": "New York Jets", "price": 1.80},
            {"name": "New England Patriots", "price": 2.10},
        ]})
        for k in ("draftkings", "fanduel", "betmgm", "caesars")
    ])  # fmt: skip
    data["home_team"], data["away_team"] = "New York Jets", "New England Patriots"
    return data


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        ingest_events(s, [game(), second_game()])
        yield s


def entry(entries, selection, market, line=None):
    return next(
        e
        for e in entries
        if e.selection == selection and e.market_type == market and e.line == line
    )


# --- board ---------------------------------------------------------------------------------


def test_board_lists_every_side_with_best_price(session):
    entries = board(session)
    buf = entry(entries, "BUF", "spreads", -2.5)
    assert buf.best.book == "soft" and buf.best.price == 2.05
    assert buf.offers == {"draftkings": 1.91, "fanduel": 1.91, "betmgm": 1.91, "soft": 2.05}
    assert buf.fair_prob == pytest.approx(0.5) and buf.ev == pytest.approx(0.025)
    # Sides that aren't +EV are on the board too.
    kc = entry(entries, "KC", "spreads", 2.5)
    assert kc.ev < 0
    # Only three books price the moneyline: too thin for the scan, still on the board.
    assert entry(entries, "BUF", "h2h").n_books == 2
    assert not [
        o
        for o in scan(session, min_ev=None)
        if o.market_type == "h2h" and o.event_id == kc.event_id
    ]
    assert {(e.selection, e.market_type) for e in entries} >= {("NYJ", "h2h"), ("NE", "h2h")}


def test_scan_is_the_board_filtered_by_ev(session):
    on_board = {(e.market_id, e.side, e.line, e.best.book) for e in board(session)}
    for o in scan(session):
        assert (o.market_id, o.side, o.line, o.book) in on_board
    assert [o.book for o in scan(session)] == ["soft"]


def test_board_event_only_matches_upcoming_games(session):
    from datetime import timedelta

    from betmap.builder.pricing import board_event
    from betmap.data.nflverse import sync_games
    from betmap.tables import utcnow

    from .test_results import schedule_row

    # Last week's KC @ BUF exists too; pricing must pick the one on the board.
    sync_games(session, [schedule_row("2026_03_KC_BUF", kickoff=utcnow() - timedelta(hours=3))])
    entries = board(session)
    for label in ("KC @ BUF", "kc at buf", "Kansas City Chiefs @ Buffalo Bills", "BUF @ KC"):
        assert board_event(session, entries, label).odds_api_id == "evt1"
    assert board_event(session, entries, "SEA @ LA") is None


def test_find_entry_accepts_abbreviations_names_and_sides(session):
    entries = board(session)
    event = session.scalars(select(Event).where(Event.odds_api_id == "evt1")).one()
    for typed in ("BUF", "Buffalo Bills", "bills"):
        assert find_entry(entries, event.id, "spreads", typed, -2.5, event).selection == "BUF"
    assert find_entry(entries, event.id, "spreads", "BUF", -3.0, event) is None


# --- pricer --------------------------------------------------------------------------------


def test_cross_game_parlay_uses_one_books_prices(session):
    entries = board(session)
    legs = [entry(entries, "BUF", "h2h"), entry(entries, "NYJ", "h2h")]
    quote = price_parlay(session, legs)
    assert not quote.same_game and not quote.problems
    assert quote.fair_prob == pytest.approx(quote.naive_prob)  # different games: independent
    assert quote.book in {"draftkings", "fanduel", "betmgm"}  # 'soft' has no NYJ price
    assert quote.price == pytest.approx(1.91 * 1.80)
    assert quote.ev == pytest.approx(quote.fair_prob * quote.price - 1)
    soft = price_parlay(session, legs, book="soft")
    assert soft.price is None and "soft doesn't offer every leg" in soft.problems


def test_exchanges_are_not_used_for_automatic_parlay_prices(session):
    data = second_game()
    # Kalshi pays best on NYJ, but exchanges don't sell parlays.
    data["bookmakers"].append(book("kalshi", {"h2h": [
        {"name": "New York Jets", "price": 1.95},
        {"name": "New England Patriots", "price": 2.0},
    ]}))  # fmt: skip
    first = game()
    first["bookmakers"].append(book("kalshi", {"h2h": h2h(2.0, 1.95)}))
    ingest_events(session, [first, data])
    entries = board(session)
    legs = [entry(entries, "BUF", "h2h"), entry(entries, "NYJ", "h2h")]
    assert {e.best.book for e in legs} == {"kalshi"}
    assert price_parlay(session, legs).book != "kalshi"
    assert price_parlay(session, legs, book="kalshi").price == pytest.approx(2.0 * 1.95)


def test_same_game_parlay_needs_the_books_price_and_uses_correlation(session):
    entries = board(session)
    ml, spread = entry(entries, "BUF", "h2h"), entry(entries, "BUF", "spreads", -2.5)
    quote = price_parlay(session, [ml, spread])
    assert quote.same_game and quote.price is None
    assert "enter the price the book quotes" in quote.problems[0]
    # Covering -2.5 almost guarantees winning: the pair is worth far more than the product.
    assert quote.fair_prob > quote.naive_prob * 1.5
    priced = price_parlay(session, [ml, spread], offered=2.6)
    assert priced.price == 2.6 and priced.ev == pytest.approx(priced.fair_prob * 2.6 - 1)


def test_parlay_problems(session):
    entries = board(session)
    buf, kc = entry(entries, "BUF", "spreads", -2.5), entry(entries, "KC", "spreads", 2.5)
    assert "the same leg is in twice" in price_parlay(session, [buf, buf], offered=3.0).problems
    both = price_parlay(session, [buf, kc], offered=3.0)
    assert "both sides of one line are in the parlay" in both.problems
    assert both.fair_prob < 0.02
    assert "at least two legs" in price_parlay(session, [buf]).problems[0]


# --- CLI -----------------------------------------------------------------------------------


def test_cli_parlay_price(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from betmap.cli import app
    from betmap.config import get_settings

    db = tmp_path / "cli.db"
    monkeypatch.setenv("BETMAP_DB_PATH", str(db))
    monkeypatch.setenv("BETMAP_BOOKS", "")
    get_settings.cache_clear()
    try:
        engine = make_engine(db)
        init_db(engine)
        with session_scope(engine) as s:
            ingest_events(s, [game(), second_game()])
        runner = CliRunner()
        r = runner.invoke(app, ["parlay", "price", "--leg", "KC @ BUF|h2h|BUF",
                                "--leg", "NE @ NYJ|h2h|NYJ"])  # fmt: skip
        assert r.exit_code == 0, r.output
        assert "2-leg parlay" in r.output and "Price at" in r.output and "EV" in r.output
        r = runner.invoke(app, ["parlay", "price", "--leg", "KC @ BUF|h2h|BUF",
                                "--leg", "KC @ BUF|spreads|BUF|-2.5"])  # fmt: skip
        assert "same-game parlay: enter the price" in r.output
        r = runner.invoke(app, ["parlay", "price", "--leg", "KC @ BUF|h2h|BUF",
                                "--leg", "KC @ BUF|spreads|BUF|-2.5", "--odds", "+160"])  # fmt: skip
        assert "Price: +160" in r.output and "as if independent" in r.output
        r = runner.invoke(app, ["parlay", "price", "--leg", "KC @ BUF|totals|Over|99.5",
                                "--leg", "NE @ NYJ|h2h|NYJ"])  # fmt: skip
        assert r.exit_code == 1 and "no price in the latest pull" in r.output
        r = runner.invoke(app, ["parlay", "price", "--leg", "SEA @ LA|h2h|SEA",
                                "--leg", "NE @ NYJ|h2h|NYJ"])  # fmt: skip
        assert r.exit_code == 1 and "no upcoming game matches 'SEA @ LA'" in r.output
    finally:
        get_settings.cache_clear()
