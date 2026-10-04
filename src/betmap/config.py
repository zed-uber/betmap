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


@lru_cache
def get_settings() -> Settings:
    return Settings()
