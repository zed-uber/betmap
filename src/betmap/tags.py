"""Key-value tags on slates, slate items, and bets, for grouping and filtering.

Tags are typed as `key=value` (or a bare `key`), separated by spaces or commas:
`scenario=shootout window=sun-early`. Keys and values are lowercased so the same tag always
groups together. A key can carry several values (one parlay can play two scenarios).
Anything the code acts on gets a real column instead; tags are only for grouping.
"""

import re

from sqlalchemy import select, union
from sqlalchemy.orm import Session

from betmap.tables import Bet, BetTag, SlateItemTag, SlateTag

# Suggested in the tag inputs; any other key is fine too.
KNOWN_KEYS = ("scenario", "window", "experiment")

KEY = re.compile(r"[a-z][a-z0-9_-]*")
VALUE = re.compile(r"[a-z0-9][a-z0-9_.:+/-]*")

Pair = tuple[str, str]


def parse_tags(text: str) -> list[Pair]:
    """`"Scenario=Shootout, hedge"` -> `[("scenario", "shootout"), ("hedge", "")]`."""
    pairs: list[Pair] = []
    for token in re.split(r"[\s,]+", text.strip().lower()):
        if not token:
            continue
        key, _, value = token.partition("=")
        if not KEY.fullmatch(key):
            raise ValueError(f"bad tag '{token}': keys are letters, digits, - and _")
        if "=" in token and not VALUE.fullmatch(value):
            raise ValueError(f"bad tag '{token}': give a value after =, without spaces")
        if (key, value) not in pairs:
            pairs.append((key, value))
    return pairs


def label(key: str, value: str) -> str:
    return f"{key}={value}" if value else key


def format_tags(tags) -> str:
    """Tag rows (or pairs) back into the text form, for editing."""
    return " ".join(label(*_pair(t)) for t in tags)


def _pair(tag) -> Pair:
    return tag if isinstance(tag, tuple) else (tag.key, tag.value)


def edit_tags(
    current: list[Pair], add: list[Pair] = (), remove: list[Pair] = (), clear: bool = False
) -> list[Pair]:
    """Tags after clearing (optional), removing, then adding. Removing a bare key drops
    every value of it; `key=value` drops just that one."""
    kept = [] if clear else list(current)
    for key, value in remove:
        kept = [(k, v) for k, v in kept if not (k == key and (not value or v == value))]
    return list(dict.fromkeys(kept + list(add)))


def set_tags(owner, pairs: list[Pair], model) -> None:
    """Make `owner.tags` exactly `pairs`, keeping rows that are already there."""
    keep = {_pair(t): t for t in owner.tags}
    owner.tags = [keep.get(p) or model(key=p[0], value=p[1]) for p in dict.fromkeys(pairs)]


def tagged(text: str):
    """A filter for bets with this tag: a bare key matches any of its values."""
    key, has_value, value = text.strip().lower().partition("=")
    match = BetTag.key == key
    if has_value:
        match &= BetTag.value == value
    return Bet.tags.any(match)


def tag_options(session: Session) -> list[str]:
    """Tags already in use plus the known keys, for autocomplete."""
    used = union(
        *(select(m.key, m.value) for m in (BetTag, SlateTag, SlateItemTag))
    )  # fmt: skip
    seen = {label(k, v) for k, v in session.execute(used)}
    return sorted(seen | {f"{k}=" for k in KNOWN_KEYS})
