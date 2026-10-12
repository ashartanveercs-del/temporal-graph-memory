"""Timestamps, clocks and duration parsing.

SQLite has no datetime type, so every instant is stored as text. Text only sorts like time
when every value uses the same zone and the same width, so the storage form is always UTC
with six fractional digits: ``2026-03-01T09:30:00.000000Z``. Comparisons such as
``valid_from <= ?`` then work as plain string comparisons and can use indexes.

Naive datetimes are rejected everywhere in the library: a naive value silently means
"local time of whichever machine ran this", which is how point-in-time queries drift by
hours between a laptop and a server.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Protocol

_DB_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_DURATION = re.compile(r"^(\d+)([smhdw])$")
_RELATIVE = re.compile(r"^([+-])(\d+)([smhdw])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """A clock that only moves when told to. Used by tests, demos and simulations."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_aware(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta | None = None, **kwargs: float) -> datetime:
        self._now += (delta or timedelta()) + timedelta(**kwargs)
        return self._now

    def set(self, value: datetime) -> None:
        self._now = ensure_aware(value)


def ensure_aware(value: datetime) -> datetime:
    """Return ``value`` converted to UTC; refuse naive datetimes."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"datetime {value!r} has no timezone; pass an aware datetime")
    return value.astimezone(UTC)


def to_db(value: datetime) -> str:
    """Fixed-width UTC text: sorts exactly like the instants it represents."""
    return ensure_aware(value).strftime(_DB_FORMAT)


def from_db(text: str) -> datetime:
    # fromisoformat is implemented in C and understands the trailing "Z"; it is roughly
    # ten times faster than strptime, which matters when a query materializes many rows.
    return datetime.fromisoformat(text)


def to_text(value: datetime) -> str:
    """Canonical human-facing form: the shortest exact ISO-8601 UTC rendering."""
    value = ensure_aware(value)
    if value.microsecond:
        return value.strftime(_DB_FORMAT)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(text: str) -> timedelta:
    """``45s``, ``90m``, ``36h``, ``7d`` or ``2w``."""
    match = _DURATION.match(text.strip())
    if not match:
        raise ValueError(f"invalid duration {text!r}; use e.g. 90m, 36h, 7d or 2w")
    amount, unit = match.groups()
    return timedelta(**{_UNITS[unit]: int(amount)})


def parse_time(text: str, *, tz: tzinfo = UTC, now: datetime | None = None) -> datetime:
    """Parse user input: ISO dates/datetimes, ``now``, or offsets like ``+3d`` / ``-90m``.

    Input without an offset is interpreted in ``tz``. The result is always UTC.
    """
    raw = text.strip()
    if raw == "now" or _RELATIVE.match(raw):
        if now is None:
            raise ValueError(f"{raw!r} is relative; a reference time is required")
        base = ensure_aware(now)
        if raw == "now":
            return base
        sign, amount, unit = _RELATIVE.match(raw).groups()  # type: ignore[union-attr]
        delta = timedelta(**{_UNITS[unit]: int(amount)})
        return base + delta if sign == "+" else base - delta
    candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(candidate), datetime.min.time())
        except ValueError:
            raise ValueError(f"cannot parse time {text!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(UTC)
