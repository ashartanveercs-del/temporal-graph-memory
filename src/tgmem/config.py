"""Settings from environment variables (optionally loaded from a ``.env`` file)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from .memory import Memory
from .timeutil import Clock

EMBEDDERS = ("auto", "none", "hash", "openai")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("memory.db")
    vault_dir: Path = Path("vault")
    embedder: str = "auto"  # auto: OpenAI-compatible when an API key is set, else none
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str | None = field(default=None, repr=False)
    embed_model: str = "text-embedding-3-small"
    embed_dimensions: int | None = None
    embed_timeout: float = 10.0
    hash_dim: int = 256
    default_country_code: str | None = None
    timezone: str = "UTC"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(name, "").strip()
            return value or None

        embedder = (get("TGMEM_EMBEDDER") or "auto").lower()
        if embedder not in EMBEDDERS:
            raise ValueError(f"TGMEM_EMBEDDER must be one of {', '.join(EMBEDDERS)}")
        country = get("TGMEM_DEFAULT_COUNTRY_CODE")
        if country is not None and not re.fullmatch(r"\d{1,3}", country):
            raise ValueError("TGMEM_DEFAULT_COUNTRY_CODE must be 1-3 digits, e.g. 1 or 44")
        settings = cls(
            db_path=Path(get("TGMEM_DB") or "memory.db"),
            vault_dir=Path(get("TGMEM_VAULT") or "vault"),
            embedder=embedder,
            embed_base_url=get("TGMEM_EMBED_BASE_URL") or "https://api.openai.com/v1",
            embed_api_key=get("TGMEM_EMBED_API_KEY") or get("OPENAI_API_KEY"),
            embed_model=get("TGMEM_EMBED_MODEL") or "text-embedding-3-small",
            embed_dimensions=_positive(get("TGMEM_EMBED_DIMENSIONS"),
                                       "TGMEM_EMBED_DIMENSIONS", int),
            embed_timeout=_positive(get("TGMEM_EMBED_TIMEOUT"), "TGMEM_EMBED_TIMEOUT",
                                    float) or 10.0,
            hash_dim=_positive(get("TGMEM_HASH_DIM"), "TGMEM_HASH_DIM", int) or 256,
            default_country_code=country,
            timezone=get("TGMEM_TIMEZONE") or "UTC",
        )
        settings.tz()  # fail fast on an unknown zone
        return settings

    def tz(self) -> tzinfo:
        if self.timezone.upper() == "UTC":
            return UTC
        try:
            return ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"TGMEM_TIMEZONE {self.timezone!r} is not a known time zone; "
                             "use an IANA name such as Europe/London") from None

    def build_embedder(self) -> Embedder | None:
        if self.embedder == "none" or (self.embedder == "auto" and not self.embed_api_key):
            return None
        if self.embedder == "hash":
            return HashingEmbedder(self.hash_dim)
        if not self.embed_api_key:
            raise ValueError("TGMEM_EMBEDDER=openai needs an API key: set TGMEM_EMBED_API_KEY "
                             "or OPENAI_API_KEY")
        return OpenAICompatibleEmbedder(api_key=self.embed_api_key, model=self.embed_model,
                                        base_url=self.embed_base_url,
                                        dimensions=self.embed_dimensions,
                                        timeout=self.embed_timeout)

    def open_memory(self, *, clock: Clock | None = None) -> Memory:
        return Memory.open(self.db_path, clock=clock, embedder=self.build_embedder(),
                           default_country_code=self.default_country_code)


def _positive(raw: str | None, name: str, kind: type) -> int | float | None:
    if raw is None:
        return None
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number, not {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, not {raw!r}")
    return value


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines: ``#`` comment lines, blank lines, an optional ``export``
    prefix and matching surrounding quotes are understood. Nothing else is interpreted."""
    values: dict[str, str] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not _KEY.match(key):
            raise ValueError(f"{path}: line {number}: expected KEY=VALUE")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values
