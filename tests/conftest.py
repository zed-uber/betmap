import os

import pytest

# CLI output is checked as plain text; a shell's FORCE_COLOR would add escape codes. This
# runs before betmap.cli creates its console.
os.environ.pop("FORCE_COLOR", None)

from betmap.config import get_settings


@pytest.fixture(autouse=True)
def no_network_sources(monkeypatch):
    """Tests never download nfelo (or read the developer's cache)."""
    monkeypatch.setenv("BETMAP_NFELO_URL", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
