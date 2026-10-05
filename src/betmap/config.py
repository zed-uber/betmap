from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BETMAP_", env_file=".env", extra="ignore")

    odds_api_key: str = ""
    db_path: Path = Path("data/betmap.db")
    cache_dir: Path = Path("data/cache")

    # Sizing defaults, as fractions of bankroll.
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.03
    max_game_fraction: float = 0.06

    # Comma-separated Odds API book keys you can bet at (e.g. "draftkings,fanduel").
    # Empty means report every book. All books still feed the consensus.
    books: str = ""

    # Books to pull, as Odds API keys. Every 10 cost one region's worth of credits, so this
    # list costs the same as pulling just the "us" region. Empty means pull the "us" region.
    pull_books: str = (
        "fanduel,draftkings,betmgm,betrivers,lowvig,betonlineag,kalshi,polymarket,novig,underdog"
    )

    # Exchange taker fees as book:rate, applied as rate x price x (1 - price) per $1 contract
    # (Kalshi's formula; some Kalshi sports series use half the rate).
    exchange_fees: str = "kalshi:0.07"

    @property
    def book_set(self) -> set[str]:
        return {b.strip() for b in self.books.split(",") if b.strip()}

    @property
    def pull_book_list(self) -> tuple[str, ...]:
        return tuple(b.strip() for b in self.pull_books.split(",") if b.strip())

    @property
    def fee_rates(self) -> dict[str, float]:
        rates = {}
        for item in self.exchange_fees.split(","):
            book, _, rate = item.partition(":")
            if book.strip():
                rates[book.strip()] = float(rate or 0)
        return rates


@lru_cache
def get_settings() -> Settings:
    return Settings()
