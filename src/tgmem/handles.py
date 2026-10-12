"""Sender handles: the phone number or email address a message arrived from.

Scoping is only as good as identity resolution, so the rules are deliberately strict and
symmetric: the same function canonicalizes a handle when a contact is registered and when
a message arrives, and lookups are exact matches on the canonical form.

There is intentionally no fuzzy "ends with the same digits" matching. Two different people
can share a ten-digit tail under different country codes, and a fuzzy match would hand one
of them the other's private memory. Anything that does not canonicalize cleanly is treated
as an unknown sender, who only ever sees public facts.
"""

from __future__ import annotations

import re

_SCHEME = re.compile(r"^(tel|sms|whatsapp|mailto):", re.IGNORECASE)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_PUNCTUATION = re.compile(r"[\s\-.()/]")


def canonical_handle(raw: str, *, default_country_code: str | None = None) -> str | None:
    """Return ``+<digits>`` (E.164) for phone numbers, a lowercase address for emails,
    or ``None`` when the input is not a usable handle.

    National-format numbers (no ``+`` or ``00`` prefix) are only accepted when a
    ``default_country_code`` such as ``"1"`` or ``"44"`` is configured; a single leading
    trunk ``0`` (used by many national dialling plans) is dropped first.
    """
    value = _SCHEME.sub("", raw.strip()).strip()
    if not value:
        return None
    if "@" in value:
        email = value.lower()
        return email if _EMAIL.match(email) else None

    compact = _PHONE_PUNCTUATION.sub("", value)
    if compact.startswith("+"):
        digits = compact[1:]
    elif compact.startswith("00"):
        digits = compact[2:]
    else:
        digits = _national_to_international(compact, default_country_code)
        if digits is None:
            return None
    if not digits.isdigit() or not 8 <= len(digits) <= 15:
        return None
    return "+" + digits


def _national_to_international(digits: str, country_code: str | None) -> str | None:
    if not digits.isdigit() or not country_code or not country_code.isdigit():
        return None
    if digits.startswith("0"):
        return country_code + digits[1:]
    if digits.startswith(country_code) and len(digits) >= 11:
        return digits  # already carries the country code, just without the "+"
    return country_code + digits
