"""Identifiers: human-readable node slugs and opaque fact/disclosure ids."""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from collections import defaultdict
from typing import Protocol

# Device names Windows refuses as file names, with or without an extension. Node ids
# become vault file names, so "con.md" would be unwritable on Windows.
_RESERVED = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(10)}


def slugify(name: str, *, max_len: int = 48) -> str:
    """Lowercase ASCII slug made of ``[a-z0-9]`` runs joined by single dashes.

    Diacritics are folded ("Núñez" -> "nunez"). Names with no ASCII letters or digits
    (for example a name written only in CJK characters) get a stable hash-based id.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    if not slug:
        return "x-" + hashlib.blake2b(name.encode("utf-8"), digest_size=4).hexdigest()
    if slug in _RESERVED:
        return slug + "-node"
    return slug


class IdFactory(Protocol):
    def new(self, prefix: str) -> str: ...


class RandomIds:
    """``f-3fa94c0b1e2d``: 48 random bits, plenty for one assistant's lifetime of facts."""

    def new(self, prefix: str) -> str:
        return f"{prefix}-{secrets.token_hex(6)}"


class SequentialIds:
    """``f-000001``, ``f-000002``...: reproducible ids for tests and demos."""

    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def new(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]:06d}"
