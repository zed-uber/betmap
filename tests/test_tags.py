import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from betmap.builder.slates import (
    add_straight,
    add_to_parlay,
    create_slate,
    duplicate_slate,
    place_slate,
    set_slate_tags,
    update_item,
)
from betmap.odds.scan import board
from betmap.tables import Bet, SlateTag
from betmap.tags import edit_tags, format_tags, parse_tags, set_tags, tag_options, tagged
from betmap.tracking import ledger

from .test_builder import entry
from .test_slates import evaluate, session  # noqa: F401
from .test_web import builder_client, shared_engine


def test_parse_tags():
    assert parse_tags(" Scenario=Shootout, hedge  window=sun-early,") == [
        ("scenario", "shootout"),
        ("hedge", ""),
        ("window", "sun-early"),
    ]
    assert parse_tags("scenario=a scenario=b scenario=a") == [("scenario", "a"), ("scenario", "b")]
    assert parse_tags("") == []
    for bad in ("scenario=", "9lives", "a=b=c", "=x"):
        with pytest.raises(ValueError, match="bad tag"):
            parse_tags(bad)


def test_edit_tags():
    now = [("scenario", "a"), ("scenario", "b"), ("hedge", "")]
    assert edit_tags(now, add=[("window", "late"), ("hedge", "")]) == [*now, ("window", "late")]
    assert edit_tags(now, remove=[("scenario", "a")]) == [("scenario", "b"), ("hedge", "")]
    assert edit_tags(now, remove=[("scenario", "")]) == [("hedge", "")]  # bare key: all values
    assert edit_tags(now, add=[("x", "")], clear=True) == [("x", "")]
    # Removing comes before adding, so a tag can be swapped in one go.
    assert edit_tags(now, add=[("scenario", "c")], remove=[("scenario", "")]) == [
        ("hedge", ""),
        ("scenario", "c"),
    ]


def test_set_tags_keeps_existing_rows(session):  # noqa: F811
    slate = create_slate(session, "S")
    set_tags(slate, parse_tags("scenario=a hedge"), SlateTag)
    session.flush()
    kept = slate.tags[1]
    set_tags(slate, parse_tags("hedge window=late"), SlateTag)
    session.flush()
    assert format_tags(slate.tags) == "hedge window=late" and slate.tags[0] is kept
    assert len(session.scalars(select(SlateTag)).all()) == 2  # the dropped tag is deleted
    assert {"hedge", "window=late", "scenario=", "experiment="} <= set(tag_options(session))


def test_placed_bets_get_slate_and_item_tags(session):  # noqa: F811
    entries = board(session)
    slate = create_slate(session, "Script")
    set_slate_tags(session, slate, parse_tags("window=sun-early"))
    straight = add_straight(session, slate, entry(entries, "BUF", "spreads", -2.5))
    parlay = add_to_parlay(session, slate, entry(entries, "BUF", "h2h"))
    add_to_parlay(session, slate, entry(entries, "NYJ", "h2h"), parlay)
    update_item(session, straight, stake=10, tags=parse_tags("scenario=chiefs-grind"))
    update_item(session, parlay, stake=5, tags=parse_tags("scenario=shootout window=sun-early"))

    copy = duplicate_slate(session, slate)
    assert format_tags(copy.tags) == "window=sun-early"
    assert format_tags(copy.items[0].tags) == "scenario=chiefs-grind"

    s_bet, p_bet = place_slate(session, evaluate(session, slate))
    assert format_tags(s_bet.tags) == "window=sun-early scenario=chiefs-grind"
    assert format_tags(p_bet.tags) == "window=sun-early scenario=shootout"  # no duplicate

    # The bets own their copies: the slate can't be retagged once placed anyway, and
    # editing the duplicate leaves them alone.
    with pytest.raises(ValueError, match="already placed"):
        set_slate_tags(session, slate, [])
    with pytest.raises(ValueError, match="already placed"):
        update_item(session, parlay, tags=[])
    set_slate_tags(session, copy, [])
    assert format_tags(s_bet.tags) == "window=sun-early scenario=chiefs-grind"

    def ids(tag):
        return [b.id for b in session.scalars(select(Bet).where(tagged(tag)).order_by(Bet.id))]

    assert ids("scenario=shootout") == [p_bet.id]
    assert ids("Scenario") == [s_bet.id, p_bet.id]  # a bare key matches any value
    assert ids("window=sun-late") == []


def test_ledger_tags(session):  # noqa: F811
    bet = ledger.place_bet(
        session, event_label="x", market_type="h2h", selection="A", book="dk", price=2.0,
        stake=10, tags=parse_tags("experiment=fade-public"),
    )  # fmt: skip
    assert format_tags(bet.tags) == "experiment=fade-public"
    ledger.edit_bet(session, bet.id, tags=parse_tags("hedge"))
    assert format_tags(bet.tags) == "hedge"


def test_web_tags():
    client, _ = builder_client()
    from .test_web import board_form

    client.post("/slates", data={"name": "Sunday", "back": "/builder"})
    r = client.post("/slates/1/tags", data={"tags": "Window=Sun-Early", "back": "/builder?slate=1"})
    assert "Saved tags" in r.text and 'value="window=sun-early"' in r.text
    assert "Couldn&#39;t tag" in client.post("/slates/1/tags", data={"tags": "a=b=c"}).text
    client.post("/slates/1/add", data=board_form(client, "BUF -2.5", "straight"))
    r = client.post(
        "/slates/items/1",
        data={"stake": "10", "tags": "scenario=shootout", "back": "/builder?slate=1"},
    )
    assert "Saved" in r.text and 'value="scenario=shootout"' in r.text
    assert '<option value="scenario=shootout">' in r.text  # autocomplete
    r = client.post("/slates/items/1", data={"stake": "10", "back": "/builder?slate=1"})
    assert 'value="scenario=shootout"' in r.text  # no tags field: tags left alone

    client.post("/slates/1/place", data={"back": "/builder?slate=1"})
    r = client.get("/bets")
    assert 'href="/bets?tag=scenario%3Dshootout"' in r.text and ">window=sun-early</a>" in r.text
    r = client.get("/bets?tag=scenario%3Dshootout")
    assert "Tagged" in r.text and "BUF" in r.text
    r = client.get("/bets?tag=scenario%3Dother")
    assert "tagged “scenario=other”" in r.text

    r = client.get("/bets/1/edit")
    assert 'value="window=sun-early scenario=shootout"' in r.text


def test_web_manual_bet_tags():
    from fastapi.testclient import TestClient

    from betmap.web.app import create_app

    client = TestClient(create_app(shared_engine()))
    form = {"event": "KC @ BUF", "market": "h2h", "selection": "BUF", "odds": "-110",
            "stake": "10", "book": "dk"}  # fmt: skip
    r = client.post("/bets", data=form | {"tags": "scenario=x=y"})
    assert (
        "Couldn&#39;t log bet: bad tag" in r.text and 'name="tags" value="scenario=x=y"' in r.text
    )
    client.post("/bets", data=form | {"tags": "hedge"})
    r = client.post(
        "/bets/1/edit",
        data=form | {"status": "open", "tags": "hedge experiment=a"},
    )
    assert "Saved #1" in r.text and ">experiment=a</a>" in r.text


def test_cli_tags(tmp_path, monkeypatch):
    from betmap.cli import app
    from betmap.config import get_settings

    monkeypatch.setenv("BETMAP_DB_PATH", str(tmp_path / "cli.db"))
    get_settings.cache_clear()
    try:
        runner = CliRunner(env={"COLUMNS": "200"})
        base = ["bet", "add", "--event", "KC @ BUF", "--market", "h2h", "--odds", "-110",
                "--stake", "10", "--book", "dk"]  # fmt: skip
        r = runner.invoke(app, [*base, "--selection", "BUF", "--tag", "scenario=shootout hedge"])
        assert r.exit_code == 0
        runner.invoke(app, [*base, "--selection", "KC", "--tag", "scenario=grind"])
        r = runner.invoke(app, ["bet", "list", "--tag", "scenario"])
        assert "BUF [scenario=shootout hedge]" in r.output and "KC [scenario=grind]" in r.output
        r = runner.invoke(app, ["bet", "list", "--tag", "scenario", "--tag", "hedge"])  # both
        assert r.output.count("│ h2h") == 1 and "BUF [" in r.output
        r = runner.invoke(app, [*base, "--selection", "KC", "--tag", "x=y=z"])
        assert r.exit_code == 1 and "bad tag" in r.output

        def edit(*args):
            return runner.invoke(app, ["bet", "edit", "1", *args])

        r = edit("--tag", "window=late", "--tag", "Experiment=A")  # adds, keeps the rest
        assert r.exit_code == 0
        assert "[scenario=shootout hedge window=late experiment=a]" in r.output
        assert "[hedge window=late experiment=a]" in edit("--untag", "scenario").output
        r = edit("--clear-tags", "--tag", "scenario=redo", "--stake", "12")
        assert "for 12.00 [scenario=redo]" in r.output
        r = edit("--clear-tags")
        assert r.exit_code == 0 and "[" not in r.output.split("for 12.00")[1]
        assert edit("--untag", "=bad").exit_code == 1
        r = runner.invoke(app, ["bet", "edit", "99", "--tag", "a"])
        assert r.exit_code == 1 and "no bet with id 99" in r.output
    finally:
        get_settings.cache_clear()


def test_cli_slate_tags(tmp_path, monkeypatch):
    from betmap.cli import app
    from betmap.config import get_settings
    from betmap.db import init_db, make_engine, session_scope
    from betmap.odds.ingest import ingest_events

    from .odds_fixtures import game

    db = tmp_path / "cli.db"
    monkeypatch.setenv("BETMAP_DB_PATH", str(db))
    monkeypatch.setenv("BETMAP_BOOKS", "")
    get_settings.cache_clear()
    try:
        engine = make_engine(db)
        init_db(engine)
        with session_scope(engine) as s:
            ingest_events(s, [game()])
            slate = create_slate(s, "Sunday")
            add_straight(s, slate, entry(board(s), "BUF", "h2h"))
            add_straight(s, slate, entry(board(s), "KC", "h2h"))
        runner = CliRunner(env={"COLUMNS": "200"})
        r = runner.invoke(app, ["slate", "tag", "1", "--tag", "window=sun-early"])
        assert r.exit_code == 0 and "Sunday: window=sun-early" in r.output
        r = runner.invoke(app, ["slate", "tag", "1", "--item", "2", "--tag", "scenario=kc"])
        assert "Sunday #2: scenario=kc" in r.output
        r = runner.invoke(app, ["slate", "tag", "1", "--item", "3", "--tag", "a"])
        assert r.exit_code == 1 and "has items 1-2" in r.output
        assert runner.invoke(app, ["slate", "tag", "1"]).exit_code == 1  # nothing to do
        assert "no slate #9" in runner.invoke(app, ["slate", "tag", "9", "--tag", "a"]).output

        r = runner.invoke(app, ["slate", "show", "1"])
        assert "Sunday (draft) [window=sun-early]" in r.output
        assert "│ 2 │ KC [scenario=kc]" in r.output
        assert "window=sun-early" in runner.invoke(app, ["slate", "list"]).output
        r = runner.invoke(app, ["slate", "tag", "1", "--item", "2", "--clear-tags"])
        assert "Sunday #2: no tags" in r.output
    finally:
        get_settings.cache_clear()
