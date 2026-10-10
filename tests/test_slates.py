from datetime import timedelta

import pytest
from sqlalchemy import select

from betmap.builder.slates import (
    add_straight,
    add_to_parlay,
    create_slate,
    delete_slate,
    draft_slates,
    duplicate_slate,
    evaluate_slate,
    place_slate,
    remove_leg,
    update_item,
)
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import board
from betmap.tables import Bet, BetKind, SlateStatus, utcnow

from .odds_fixtures import game
from .test_builder import entry, second_game


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        ingest_events(s, [game(), second_game()])
        yield s


def evaluate(session, slate, equity=1000.0):
    return evaluate_slate(session, slate, board(session), equity, 0.25, 0.03, 0.06)


# --- editing -------------------------------------------------------------------------------


def test_build_and_edit_a_slate(session):
    entries = board(session)
    slate = create_slate(session, "Sunday main")
    straight = add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))
    parlay = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), parlay)
    assert [i.kind for i in slate.items] == [BetKind.STRAIGHT, BetKind.PARLAY]
    assert straight.legs[0].book == "soft" and straight.legs[0].price_at_add == 2.05
    assert len(parlay.legs) == 2
    with pytest.raises(ValueError, match="already in this parlay"):
        add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), parlay)
    with pytest.raises(ValueError, match="not a parlay"):
        add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), straight)

    remove_leg(session, parlay.legs[1])
    assert len(parlay.legs) == 1
    remove_leg(session, straight.legs[0])  # the last leg removes the item
    assert slate.items == [parlay]
    with pytest.raises(ValueError, match="needs a name"):
        create_slate(session, "  ")
    with pytest.raises(ValueError, match="negative"):
        update_item(session, parlay, stake=-1)
    assert draft_slates(session) == [slate]


def test_duplicate_slate(session):
    entries = board(session)
    slate = create_slate(session, "A")
    item = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), item)
    update_item(session, item, stake=25)
    copy = duplicate_slate(session, slate)
    assert copy.name == "A (copy)" and copy.id != slate.id
    assert [len(i.legs) for i in copy.items] == [2] and copy.items[0].stake == 25
    remove_leg(session, copy.items[0].legs[0])
    assert len(slate.items[0].legs) == 2  # the original is untouched


# --- evaluating ----------------------------------------------------------------------------


def test_evaluate_prices_sizes_and_simulates(session):
    entries = board(session)
    slate = create_slate(session, "S")
    add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))  # +2.5% at soft
    add_straight(session, slate, entry(entries, "KC", "spreads", 2.5))  # -EV
    view = evaluate(session, slate)
    buf, kc = view.items
    assert (buf.book, buf.price) == ("soft", 2.05) and buf.ev == pytest.approx(0.025)
    assert buf.suggested > 0 and buf.stake == pytest.approx(round(buf.suggested * 1000, 2))
    assert kc.ev < 0 and kc.suggested == 0 and kc.stake == 0
    assert view.total_stake == buf.stake and view.risk.worst == -buf.stake
    assert view.growth is not None and not view.problems

    update_item(session, kc.item, stake=40)  # an entered stake overrides the suggestion
    view = evaluate(session, slate)
    assert view.items[1].stake == 40 and view.total_stake == pytest.approx(buf.stake + 40)
    # BUF -2.5 and KC +2.5 are opposite sides of one line.
    assert view.overlaps and view.overlaps[0].correlation < -0.9


def test_evaluate_parlays(session):
    entries = board(session)
    slate = create_slate(session, "P")
    cross = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), cross)
    sgp = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "BUF", "spreads", -2.5), sgp)
    add_to_parlay(session, slate, entry(entries, "NE", "h2h"))  # a one-leg parlay
    view = evaluate(session, slate)
    cross_v, sgp_v, single_v = view.items
    assert cross_v.price == pytest.approx(1.91 * 1.80) and cross_v.book != "soft"
    assert sgp_v.price is None and "enter the price the book quotes" in sgp_v.problems[0]
    assert single_v.problems == ["add at least one more leg"]

    update_item(session, sgp, offered_price=2.6, book="fanduel")
    sgp_v = evaluate(session, slate).items[1]
    assert sgp_v.price == 2.6 and sgp_v.book == "fanduel" and not sgp_v.problems
    assert sgp_v.fair_prob > sgp_v.quote.naive_prob


def test_moved_and_missing_legs(session):
    entries = board(session)
    slate = create_slate(session, "M")
    add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))
    add_straight(session, slate, entry(entries, "NYJ", "h2h"))
    moved = game()
    moved["bookmakers"][-1]["markets"][0]["outcomes"][0]["price"] = 2.10  # soft moves BUF
    gone = second_game(hours=-1)  # NE @ NYJ kicked off
    ingest_events(session, [moved, gone])
    view = evaluate(session, slate)
    assert view.items[0].legs[0].moved and view.items[0].price == 2.10
    assert "no longer offered" in view.items[1].problems[0] and view.items[1].stake == 0


# --- placing -------------------------------------------------------------------------------


def test_place_slate_logs_bets_and_parlays(session):
    entries = board(session)
    slate = create_slate(session, "Go")
    add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))
    parlay = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), parlay)
    update_item(session, parlay, stake=10)
    skipped = add_straight(session, slate, entry(entries, "KC", "spreads", 2.5))  # stake 0
    assert skipped

    bets = place_slate(session, evaluate(session, slate))
    assert slate.status == SlateStatus.PLACED and len(bets) == 2
    straight, par = bets
    assert (straight.selection, straight.book, straight.price) == ("BUF", "soft", 2.05)
    assert straight.market_id is not None and straight.event_id is not None
    assert par.kind == BetKind.PARLAY and par.stake == 10 and len(par.legs) == 2
    assert all(leg.event_id and leg.market_id and leg.fair_prob for leg in par.legs)
    assert par.notes is None
    assert all(b.slate is slate and b.source == "manual" for b in bets)
    assert [b.slate_item_id for b in bets] == [slate.items[0].id, parlay.id]
    assert len(session.scalars(select(Bet)).all()) == 2
    with pytest.raises(ValueError, match="already placed"):
        add_straight(session, slate, entry(entries, "NYJ", "h2h"))
    with pytest.raises(ValueError, match="already placed"):  # it's the record of its bets
        delete_slate(session, slate)


def test_placed_bets_take_the_slates_source(session):
    entries = board(session)
    slate = create_slate(session, "Suggested")
    slate.source = "synergy-v1"
    item = add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))
    update_item(session, item, stake=10)
    copy = duplicate_slate(session, slate)
    assert copy.source == "synergy-v1"  # a variation of the suggestion is still the model's
    [bet] = place_slate(session, evaluate(session, slate))
    assert bet.source == "synergy-v1" and bet.slate_item_id == item.id


def test_delete_draft_slate(session):
    slate = create_slate(session, "Scratch")
    add_straight(session, slate, entry(board(session), "BUF", "h2h"))
    delete_slate(session, slate)
    assert draft_slates(session) == []


def test_place_refuses_problems_and_empty_slates(session):
    entries = board(session)
    empty = create_slate(session, "Empty")
    with pytest.raises(ValueError, match="nothing to place"):
        place_slate(session, evaluate(session, empty))
    slate = create_slate(session, "SGP")
    sgp = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "BUF", "spreads", -2.5), sgp)
    update_item(session, sgp, stake=10)
    with pytest.raises(ValueError, match="fix these first"):
        place_slate(session, evaluate(session, slate))
    assert slate.status == SlateStatus.DRAFT


def test_placed_slate_parlay_grades_end_to_end(session):
    from betmap.data.nflverse import sync_games
    from betmap.tracking.grading import settle_open_bets

    from .test_results import schedule_row

    entries = board(session)
    slate = create_slate(session, "E2E")
    parlay = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), parlay)
    update_item(session, parlay, stake=10)
    [bet] = place_slate(session, evaluate(session, slate))
    kickoff = utcnow() - timedelta(hours=5)
    sync_games(session, [
        schedule_row("2026_05_KC_BUF", "KC", "BUF", kickoff + timedelta(hours=53), (17, 27)),
        schedule_row("2026_05_NE_NYJ", "NE", "NYJ", kickoff + timedelta(hours=55), (14, 20)),
    ])  # fmt: skip
    settle_open_bets(session)
    assert bet.status == "win" and bet.payout == pytest.approx(10 * bet.price)


def test_cli_slates(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from betmap.cli import app
    from betmap.config import get_settings
    from betmap.tracking import ledger

    db = tmp_path / "cli.db"
    monkeypatch.setenv("BETMAP_DB_PATH", str(db))
    monkeypatch.setenv("BETMAP_BOOKS", "")
    get_settings.cache_clear()
    try:
        engine = make_engine(db)
        init_db(engine)
        with session_scope(engine) as s:
            ingest_events(s, [game(), second_game()])
            ledger.record_transfer(s, "deposit", 1000)
            entries = board(s)
            a = create_slate(s, "Alpha")
            add_straight(s, a, entry(entries, "BUF", "spreads", -2.5))
            b = create_slate(s, "Beta")
            item = add_straight(s, b, entry(entries, "NYJ", "h2h"))
            update_item(s, item, stake=20)
        runner = CliRunner()
        assert "Alpha" in runner.invoke(app, ["slate", "list"]).output
        r = runner.invoke(app, ["slate", "show", "1"])
        assert r.exit_code == 0 and "BUF" in r.output and "suggested" in r.output
        r = runner.invoke(app, ["slate", "compare", "1", "2"])
        assert "Alpha" in r.output and "Beta" in r.output and "Growth" in r.output
        r = runner.invoke(app, ["slate", "place", "2"])
        assert r.exit_code == 0 and "Logged #1: NYJ for 20.00" in r.output
        assert runner.invoke(app, ["slate", "show", "9"]).exit_code == 1
    finally:
        get_settings.cache_clear()
