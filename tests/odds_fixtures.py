"""Fake Odds API payloads shaped like the real v4 responses."""

from datetime import timedelta

from betmap.tables import utcnow


def iso_in(hours: float) -> str:
    return (utcnow() + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def book(key: str, markets: dict[str, list[dict]]) -> dict:
    return {
        "key": key,
        "title": key,
        "last_update": iso_in(-0.1),
        "markets": [{"key": k, "outcomes": v} for k, v in markets.items()],
    }


def h2h(home: float, away: float) -> list[dict]:
    return [
        {"name": "Buffalo Bills", "price": home},
        {"name": "Kansas City Chiefs", "price": away},
    ]


def spreads(home_point: float, home: float, away: float) -> list[dict]:
    return [
        {"name": "Buffalo Bills", "price": home, "point": home_point},
        {"name": "Kansas City Chiefs", "price": away, "point": -home_point},
    ]


def game(event_id: str = "evt1", hours: float = 48, bookmakers: list[dict] | None = None) -> dict:
    """KC @ BUF. Three books agree on -110/-110 BUF -2.5; 'soft' hangs +105 on BUF."""
    if bookmakers is None:
        bookmakers = [
            book("draftkings", {"h2h": h2h(1.91, 1.91), "spreads": spreads(-2.5, 1.91, 1.91)}),
            book("fanduel", {"h2h": h2h(1.91, 1.91), "spreads": spreads(-2.5, 1.91, 1.91)}),
            book("betmgm", {"h2h": h2h(1.91, 1.91), "spreads": spreads(-2.5, 1.91, 1.91)}),
            book("soft", {"spreads": spreads(-2.5, 2.05, 1.80)}),
        ]
    return {
        "id": event_id,
        "sport_key": "americanfootball_nfl",
        "commence_time": iso_in(hours),
        "home_team": "Buffalo Bills",
        "away_team": "Kansas City Chiefs",
        "bookmakers": bookmakers,
    }


def prop_event(event_id: str = "evt1") -> dict:
    def ou(over: float, under: float) -> list[dict]:
        return [
            {"name": "Over", "description": "Josh Allen", "price": over, "point": 249.5},
            {"name": "Under", "description": "Josh Allen", "price": under, "point": 249.5},
        ]

    data = game(event_id, bookmakers=[])
    data["bookmakers"] = [
        book(k, {"player_pass_yds": ou(o, u)})
        for k, o, u in [
            ("draftkings", 1.87, 1.95),
            ("fanduel", 1.87, 1.95),
            ("caesars", 1.87, 1.95),
            ("betmgm", 2.10, 1.75),
        ]
    ]
    return data
