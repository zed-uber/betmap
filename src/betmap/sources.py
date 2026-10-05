"""Second opinions on each board price: Pinnacle, exchanges, nfelo, and betmap's models.

These are shown next to the consensus fair price and recorded for `model evaluate`; they
don't change fair prices, EV, or stakes until the forward test shows they deserve to.
"""

import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import fmean

import httpx
from sqlalchemy.orm import Session

from betmap.config import get_settings
from betmap.models.game_model import side_prob
from betmap.models.predict import MODEL_NAME, PROP_MODEL_NAME, load_predictions, store_predictions
from betmap.odds.client import EXCHANGE_BOOKS
from betmap.odds.math import devig
from betmap.odds.scan import BoardEntry, latest_quotes, line_key, usable_market
from betmap.tables import Event, as_utc, utcnow

SOURCES = ("pinnacle", "exchanges", "nfelo", "model")
LABELS = {"pinnacle": "Pinnacle", "exchanges": "Exchanges", "nfelo": "nfelo", "model": "betmap"}
DIVERGENCE = 0.05  # flag a source this far from the consensus
NFELO_MARGIN_SD = 13.5
NFELO_MAX_AGE = timedelta(hours=24)

Key = tuple[int, str, float | None]  # (market id, side, line), as BoardEntry.key


@dataclass
class Opinions:
    probs: dict[str, float]  # source -> probability for this side at this line
    breakeven: float  # 1 / best price
    fair: float  # consensus

    @property
    def agree(self) -> int:
        """Sources that make the best price +EV."""
        return sum(p > self.breakeven for p in self.probs.values())

    @property
    def divergent(self) -> list[str]:
        return [s for s, p in self.probs.items() if abs(p - self.fair) > DIVERGENCE]


# --- market sources ------------------------------------------------------------------------


def _quotes_by_line(
    session: Session,
) -> dict[tuple[int, float | None], dict[str, dict[str, tuple[float, float | None]]]]:
    """Latest-pull prices grouped like the scan: (market, line key) -> book -> side -> (price, line)."""
    now = utcnow()
    groups: dict = defaultdict(lambda: defaultdict(dict))
    for snap, market, event in latest_quotes(session):
        if event.kickoff is None or as_utc(event.kickoff) <= now:
            continue
        key = line_key(market.market_type, snap.side, snap.line, event.home_team)
        groups[(market.id, key)][snap.book][snap.side] = (snap.price, snap.line)
    return groups


def market_opinions(session: Session) -> dict[str, dict[Key, float]]:
    """Devigged Pinnacle, and the average devigged exchange price, for every side/line."""
    out: dict[str, dict[Key, float]] = {"pinnacle": {}, "exchanges": {}}
    for (market_id, _), by_book in _quotes_by_line(session).items():
        sides = sorted({s for quotes in by_book.values() for s in quotes})
        if len(sides) < 2:
            continue
        fair: dict[str, list[float]] = {}
        for book, quotes in by_book.items():
            if not all(s in quotes for s in sides):
                continue
            prices = [quotes[s][0] for s in sides]
            if usable_market(prices):
                fair[book] = devig(prices)
        exchanges = [probs for book, probs in fair.items() if book in EXCHANGE_BOOKS]
        for i, side in enumerate(sides):
            line = next(q[side][1] for q in by_book.values() if side in q)
            key = (market_id, side, line)
            if "pinnacle" in fair:
                out["pinnacle"][key] = fair["pinnacle"][i]
            if exchanges:
                out["exchanges"][key] = fmean(p[i] for p in exchanges)
    return out


# --- nfelo ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class NfeloGame:
    home_win: float
    home_line: float  # nfelo's projected line for the home team (negative = favored)


def parse_nfelo(text: str) -> dict[str, NfeloGame]:
    games = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            games[row["game_id"]] = NfeloGame(
                float(row["nfelo_home_probability_close"]), float(row["nfelo_home_line_close"])
            )
        except (KeyError, ValueError):
            continue  # missing values for games nfelo hasn't projected
    return games


def load_nfelo(
    cache_dir: Path | None = None, http: httpx.Client | None = None
) -> dict[str, NfeloGame]:
    """nfelo's game predictions, downloaded at most once a day. Empty if unavailable."""
    settings = get_settings()
    if not settings.nfelo_url:
        return {}  # turned off
    path = (cache_dir or settings.cache_dir) / "nfelo_games.csv"
    fresh = (
        path.exists()
        and utcnow().timestamp() - path.stat().st_mtime < NFELO_MAX_AGE.total_seconds()
    )
    if not fresh:
        try:
            client = http or httpx.Client(timeout=30, follow_redirects=True)
            r = client.get(settings.nfelo_url)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
        except httpx.HTTPError:
            if not path.exists():
                return {}  # offline and never cached: nfelo is simply absent
    return parse_nfelo(path.read_text())


def nfelo_opinions(
    session: Session, entries: list[BoardEntry], games: dict[str, NfeloGame]
) -> dict[Key, float]:
    """Moneylines from nfelo's win probability; spreads at any line from its projected line."""
    out: dict[Key, float] = {}
    for e in entries:
        if e.market_type not in ("h2h", "spreads"):
            continue
        event = session.get(Event, e.event_id)
        game = event and event.nflverse_game_id and games.get(event.nflverse_game_id)
        if not game:
            continue
        is_home = e.side == event.home_team
        if e.market_type == "h2h":
            p_home = game.home_win
        else:
            # A home handicap h covers when margin > -h; nfelo's expected margin is -home_line.
            home_handicap = e.line if is_home else -e.line
            p_home = side_prob(-game.home_line, NFELO_MARGIN_SD, -home_handicap)
        out[e.key] = p_home if is_home else 1 - p_home
    return out


# --- together ------------------------------------------------------------------------------


def opinions(
    session: Session, entries: list[BoardEntry], nfelo: dict[str, NfeloGame] | None = None
) -> dict[Key, Opinions]:
    by_source = market_opinions(session)
    by_source["nfelo"] = nfelo_opinions(session, entries, nfelo or {})
    models = load_predictions(session, MODEL_NAME) | load_predictions(session, PROP_MODEL_NAME)
    by_source["model"] = models
    out = {}
    for e in entries:
        probs = {s: by_source[s][e.key] for s in SOURCES if e.key in by_source[s]}
        if probs:
            out[e.key] = Opinions(probs, 1 / e.best.price, e.fair_prob)
    return out


def record_opinions(
    session: Session, entries: list[BoardEntry], nfelo: dict[str, NfeloGame] | None = None
) -> dict[str, int]:
    """Store Pinnacle, exchange, and nfelo probabilities as predictions for the forward test."""
    by_source = market_opinions(session)
    by_source["nfelo"] = nfelo_opinions(session, entries, nfelo or {})
    return {
        name: store_predictions(session, name, by_source[name])
        for name in ("pinnacle", "exchanges", "nfelo")
    }
