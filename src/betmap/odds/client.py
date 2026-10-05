"""Minimal client for The Odds API v4 (https://the-odds-api.com/liveapi/guides/v4/).

Credit cost per call is markets x regions; for per-event prop calls that applies per event.
Listing bookmakers instead of regions costs one region per 10 books, which is how one
pull can cover sportsbooks, exchanges (Kalshi), and pick'em sites (Underdog) at the price
of a single region. The events listing is free.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from betmap.tables import utcnow

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
GAME_MARKETS = ("h2h", "spreads", "totals")
# Pick'em (DFS) sites pay multipliers on multi-pick entries, so a single pick's listed
# price isn't a bet you can make. Stored, but kept out of the consensus and the scan.
PICKEM_BOOKS = frozenset({"underdog", "prizepicks", "dabble_us_dfs", "pick6"})


class OddsApiError(Exception):
    pass


@dataclass
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> "Quota":
        def num(name: str) -> int | None:
            value = headers.get(name)
            return None if value is None else int(float(value))

        return cls(num("x-requests-remaining"), num("x-requests-used"), num("x-requests-last"))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class OddsApiClient:
    def __init__(self, api_key: str, *, http: httpx.Client | None = None):
        if not api_key:
            raise OddsApiError("BETMAP_ODDS_API_KEY is not set")
        self.api_key = api_key
        self.http = http or httpx.Client(base_url=BASE_URL, timeout=20)
        self.quota = Quota()
        self.credits_spent = 0

    def _get(self, path: str, **params: str) -> list | dict:
        r = self.http.get(path, params={"apiKey": self.api_key, **params})
        self.quota = Quota.from_headers(r.headers)
        self.credits_spent += self.quota.last_cost or 0
        if r.status_code != 200:
            try:
                detail = r.json().get("message", r.text)
            except ValueError:
                detail = r.text
            raise OddsApiError(f"Odds API {r.status_code}: {detail}")
        return r.json()

    @staticmethod
    def _window(days: float) -> dict[str, str]:
        now = utcnow()
        return {"commenceTimeFrom": _iso(now), "commenceTimeTo": _iso(now + timedelta(days=days))}

    def events(self, days: float = 7) -> list[dict]:
        """Upcoming events without odds (free)."""
        return self._get(f"/sports/{SPORT}/events", **self._window(days))

    @staticmethod
    def _venues(regions: str, bookmakers: tuple[str, ...]) -> dict[str, str]:
        # The API uses `bookmakers` instead of `regions` when given.
        return {"bookmakers": ",".join(bookmakers)} if bookmakers else {"regions": regions}

    def game_odds(
        self,
        markets: tuple[str, ...] = GAME_MARKETS,
        regions: str = "us",
        days: float = 7,
        bookmakers: tuple[str, ...] = (),
    ) -> list[dict]:
        return self._get(
            f"/sports/{SPORT}/odds",
            markets=",".join(markets),
            oddsFormat="decimal",
            **self._venues(regions, bookmakers),
            **self._window(days),
        )

    def event_odds(
        self,
        event_id: str,
        markets: tuple[str, ...],
        regions: str = "us",
        bookmakers: tuple[str, ...] = (),
    ) -> dict:
        """Odds for one event; the only way to get player props."""
        return self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            markets=",".join(markets),
            oddsFormat="decimal",
            **self._venues(regions, bookmakers),
        )
