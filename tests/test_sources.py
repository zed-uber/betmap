import os
import time

import httpx
import pytest
from sqlalchemy import select

from betmap.config import get_settings
from betmap.data.nflverse import sync_games
from betmap.db import init_db, make_engine, session_scope
from betmap.models.evaluate import evaluate
from betmap.odds.ingest import ingest_events
from betmap.odds.scan import board
from betmap.sources import (
    NfeloGame,
    load_nfelo,
    market_opinions,
    nfelo_opinions,
    opinions,
    parse_nfelo,
    record_opinions,
)
from betmap.tables import Event, Prediction

from .odds_fixtures import book, game, h2h, spreads
from .test_results import schedule_row

NFELO_CSV = (
    ",game_id,nfelo_home_line_close,nfelo_home_probability_close,other\n"
    "0,2026_05_KC_BUF,-3.0,0.62,x\n"
    "1,2026_05_NE_NYJ,,,x\n"  # not projected yet
)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with session_scope(engine) as s:
        data = game()
        data["bookmakers"] += [
            book("pinnacle", {"h2h": h2h(1.70, 2.25), "spreads": spreads(-2.5, 1.95, 1.95)}),
            book("kalshi", {"h2h": h2h(1.75, 2.20)}),
            book("novig", {"h2h": h2h(1.40, 1.50)}),  # thin book: unusable
        ]
        ingest_events(s, [data])
        event = s.scalars(select(Event)).one()
        sync_games(s, [schedule_row("2026_05_KC_BUF", kickoff=event.kickoff)])  # link nflverse id
        yield s


def entry(entries, selection, market, line=None):
    return next(
        e
        for e in entries
        if e.selection == selection and e.market_type == market and e.line == line
    )


def test_market_opinions(session):
    entries = board(session)
    ops = market_opinions(session)
    buf_ml = entry(entries, "BUF", "h2h")
    p_pin = ops["pinnacle"][buf_ml.key]
    assert p_pin == pytest.approx(0.57, abs=0.01)  # 1.70 / 2.25 devigged
    assert ops["pinnacle"][entry(entries, "KC", "h2h").key] == pytest.approx(1 - p_pin)
    assert ops["pinnacle"][entry(entries, "BUF", "spreads", -2.5).key] == pytest.approx(0.5)
    # Only Kalshi counts: Novig's 1.40/1.50 isn't a real market.
    assert ops["exchanges"][buf_ml.key] == pytest.approx(0.556, abs=0.01)


def test_parse_and_load_nfelo(tmp_path, monkeypatch):
    games = parse_nfelo(NFELO_CSV)
    assert games == {"2026_05_KC_BUF": NfeloGame(0.62, -3.0)}

    monkeypatch.setenv("BETMAP_NFELO_URL", "https://example.test/nfelo.csv")
    get_settings.cache_clear()
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, text=NFELO_CSV)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    assert load_nfelo(tmp_path, http) == games and len(calls) == 1
    assert load_nfelo(tmp_path, http) == games and len(calls) == 1  # fresh cache, no download

    stale = time.time() - 2 * 86400
    os.utime(tmp_path / "nfelo_games.csv", (stale, stale))
    offline = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert load_nfelo(tmp_path, offline) == games  # falls back to the old copy
    assert load_nfelo(tmp_path / "empty", offline) == {}  # nothing cached: no nfelo

    monkeypatch.setenv("BETMAP_NFELO_URL", "")
    get_settings.cache_clear()
    assert load_nfelo(tmp_path, http) == {}


def test_nfelo_opinions(session):
    entries = board(session)
    ops = nfelo_opinions(session, entries, parse_nfelo(NFELO_CSV))
    assert ops[entry(entries, "BUF", "h2h").key] == 0.62
    assert ops[entry(entries, "KC", "h2h").key] == pytest.approx(0.38)
    # nfelo has BUF by 3: -2.5 covers more often than not; KC +2.5 is the complement.
    buf = ops[entry(entries, "BUF", "spreads", -2.5).key]
    assert 0.5 < buf < 0.56
    assert ops[entry(entries, "KC", "spreads", 2.5).key] == pytest.approx(1 - buf)
    assert nfelo_opinions(session, entries, {}) == {}


def test_opinions_agreement_and_divergence(session):
    entries = board(session)
    session.add(Prediction(market_id=entry(entries, "BUF", "h2h").market_id, side="Buffalo Bills",
                           line=None, model="ratings-v1", fair_prob=0.70))  # fmt: skip
    session.flush()
    ops = opinions(session, entries, parse_nfelo(NFELO_CSV))
    buf = ops[entry(entries, "BUF", "h2h").key]
    assert set(buf.probs) == {"pinnacle", "exchanges", "nfelo", "model"}
    assert buf.breakeven == pytest.approx(1 / entry(entries, "BUF", "h2h").best.price)
    assert buf.agree == sum(p > buf.breakeven for p in buf.probs.values())
    assert "model" in buf.divergent  # 70% vs a ~52% consensus


def test_record_opinions_feed_the_forward_test(session):
    from datetime import timedelta

    entries = board(session)
    counts = record_opinions(session, entries, parse_nfelo(NFELO_CSV))
    assert counts["pinnacle"] == 4 and counts["exchanges"] == 2 and counts["nfelo"] == 4
    names = {p.model for p in session.scalars(select(Prediction))}
    assert names == {"pinnacle", "exchanges", "nfelo"}
    # Re-recording replaces rather than duplicates.
    record_opinions(session, entries, parse_nfelo(NFELO_CSV))
    assert len(session.scalars(select(Prediction)).all()) == 10
    # Once the game is final, evaluate groups by source (no close pulled, so not scored yet).
    event = session.scalars(select(Event)).one()
    event.kickoff = event.kickoff - timedelta(days=3)
    event.home_score, event.away_score = 24, 20
    result = evaluate(session)
    assert result.no_close > 0 and not result.groups
