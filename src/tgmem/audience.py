"""The audience ledger: what was told to whom, and what that implies.

Disclosures are recorded per *fact version*, not per topic, which is what makes the
interesting questions answerable:

* Repetition: was this contact already told this value? A newer row carrying the same
  value (a remainder, or a fact consolidation folded into another) counts as told.
* Corrections owed: was this contact told something that has since changed, been
  corrected or been retracted, without being told the current value?
* Leaks: does an outgoing reply contain a distinctive value from a fact the recipient
  is not allowed to see?

Mention and leak detection are deterministic token matching, deliberately conservative:
a value is only matched when it is distinctive (several words, a long word, or a number
with three or more digits), so short generic values never trigger false alarms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .model import (
    ACTIVE,
    BELIEVED,
    CORRECTED,
    DUPLICATE,
    EXPIRED,
    MERGED,
    ONE,
    RETRACTED,
    SUPERSEDED,
    Disclosure,
    Fact,
    Viewer,
)
from .textindex import tokenize
from .timeutil import to_text

if TYPE_CHECKING:
    from .memory import Memory

_NUMBER = re.compile(r"\d{3,}")
_YEAR = re.compile(r"^(19|20)\d\d$")


@dataclass(frozen=True)
class Correction:
    told: Fact  # what the contact was told
    told_at: datetime
    current: Fact | None  # what is true now, if anything
    reason: str  # "changed", "corrected" or "retracted"


@dataclass(frozen=True)
class ContextFact:
    fact: Fact
    told_at: datetime | None  # when this contact last heard this value, if ever
    attribution: str | None  # who claimed it, when that is not the owner


@dataclass(frozen=True)
class ContextPack:
    """Everything an assistant needs for one reply, already scoped to the sender."""

    viewer: Viewer
    facts: tuple[ContextFact, ...]
    corrections: tuple[Correction, ...]
    mode: str
    degraded: str | None
    text: str  # ready to paste into a system prompt

    @property
    def fresh(self) -> list[Fact]:
        return [cf.fact for cf in self.facts if cf.told_at is None]


# -- repetition ----------------------------------------------------------------------------

def told_at(memory: Memory, contact: str, fact: Fact,
            since: datetime | None = None) -> datetime | None:
    """When ``contact`` was last told ``fact``'s value, or ``None``."""
    family = memory.store.query_facts(subject=fact.subject, predicate=fact.predicate,
                                      scopes=[fact.scope])
    equivalent = {fact.id} | {f.id for f in family
                              if (f.value, f.object) == (fact.value, fact.object)}
    grew = True
    while grew:  # facts consolidation folded into an equivalent one count as well
        before = len(equivalent)
        equivalent |= {f.id for f in family
                       if f.status in (DUPLICATE, MERGED) and f.superseded_by in equivalent}
        grew = len(equivalent) > before
    disclosures = memory.store.disclosures(contact=contact, fact_ids=sorted(equivalent))
    times = [d.told_at for d in disclosures if since is None or d.told_at >= since]
    return max(times, default=None)


# -- corrections ---------------------------------------------------------------------------

def corrections_owed(memory: Memory, contact: str) -> list[Correction]:
    now = memory.now()
    latest_per_key: dict[tuple[str, str, str], tuple[Disclosure, Fact]] = {}
    latest_per_fact: dict[str, tuple[Disclosure, Fact]] = {}
    for disclosure in memory.store.disclosures(contact=contact):
        fact = memory.store.get_fact(disclosure.fact_id)
        if fact is None:
            continue
        if memory.cardinality(fact.predicate) == ONE:
            latest_per_key[fact.key] = (disclosure, fact)
        else:
            latest_per_fact[fact.id] = (disclosure, fact)

    owed: list[Correction] = []
    for disclosure, told in latest_per_key.values():
        if told.status in (ACTIVE, EXPIRED) and not told.valid_at(now):
            continue  # it lapsed on schedule (a promo ended); nothing was wrong
        current = memory.store.query_facts(subject=told.subject, predicate=told.predicate,
                                           scopes=[told.scope], statuses=BELIEVED, valid_at=now)
        if not current:
            if told.status in (CORRECTED, RETRACTED):
                owed.append(Correction(told, disclosure.told_at, None, told.status))
            continue
        if (current[0].value, current[0].object) == (told.value, told.object):
            continue
        reason = "corrected" if told.status == CORRECTED else "changed"
        owed.append(Correction(told, disclosure.told_at, current[0], reason))

    for disclosure, told in latest_per_fact.values():
        if told.status == RETRACTED:
            owed.append(Correction(told, disclosure.told_at, None, "retracted"))
        elif told.status in (CORRECTED, SUPERSEDED):
            successor = _successor(memory, told, now)
            if successor is None and told.status == CORRECTED:
                owed.append(Correction(told, disclosure.told_at, None, "corrected"))
            elif successor is not None and told_at(memory, contact, successor) is None:
                reason = "corrected" if told.status == CORRECTED else "changed"
                owed.append(Correction(told, disclosure.told_at, successor, reason))
    return sorted(owed, key=lambda c: (c.told_at, c.told.id))


def _successor(memory: Memory, fact: Fact, now: datetime) -> Fact | None:
    seen, current = set(), fact
    while current.superseded_by and current.superseded_by not in seen:
        seen.add(current.id)
        nxt = memory.store.get_fact(current.superseded_by)
        if nxt is None:
            return None
        if nxt.believed and nxt.valid_at(now):
            return nxt
        current = nxt
    return None


# -- mentions and leaks --------------------------------------------------------------------

MAX_PHRASE = 32  # long values are matched by their first 32 words


def secrets_of(text: str) -> list[tuple[str, ...]]:
    """Token sequences that would reveal a value if they appeared in a reply: the value's
    opening phrase (when distinctive) and each number of three or more digits."""
    words = tuple(tokenize(text))
    found: list[tuple[str, ...]] = []
    if len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 8):
        found.append(words[:MAX_PHRASE])
    for word in words:
        if _NUMBER.search(word) and not _YEAR.match(word) and (word,) not in found:
            found.append((word,))
    return found


def _contains(haystack: list[str], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return any(tuple(haystack[i:i + n]) == needle for i in range(len(haystack) - n + 1))


def mentions(memory: Memory, text: str, facts: list[Fact]) -> list[Fact]:
    """Facts among ``facts`` whose value appears in ``text``."""
    tokens = tokenize(text)
    return [f for f in facts
            if any(_contains(tokens, s) for s in secrets_of(value_text(memory, f)))]


class SecretIndex:
    """Phrase -> facts index over every fact's value, rebuilt only when facts or node names
    change (in any process). A reply is then checked by looking up its word n-grams, which
    costs the same whether the memory holds a hundred facts or a hundred thousand."""

    def __init__(self) -> None:
        self._version: tuple[int, int] | None = None
        self._by_phrase: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        self._longest = 0

    def refresh(self, memory: Memory) -> None:
        version = (memory.store.facts_version(), memory.store.nodes_version())
        if version == self._version:
            return
        names = {node.id: node.name for node in memory.store.nodes()}
        by_phrase: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        for fact_id, scope, value, obj in memory.store.fact_values():
            text = value if value is not None else names.get(obj or "", "")
            for secret in secrets_of(text):
                by_phrase.setdefault(secret, []).append((fact_id, scope))
        self._by_phrase = by_phrase
        self._longest = max((len(p) for p in by_phrase), default=0)
        self._version = version

    def hits(self, tokens: list[str]) -> dict[tuple[str, ...], list[tuple[str, str]]]:
        found: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        for i in range(len(tokens)):
            for n in range(1, min(self._longest, len(tokens) - i) + 1):
                phrase = tuple(tokens[i:i + n])
                if phrase in self._by_phrase:
                    found[phrase] = self._by_phrase[phrase]
        return found


def leaks(memory: Memory, viewer: Viewer, text: str,
          index: SecretIndex | None = None) -> list[Fact]:
    """Facts ``viewer`` may not see whose distinctive values appear in ``text``.

    A phrase the viewer can legitimately know (because a fact visible to them contains
    it) is not a leak, even if a hidden fact contains it too.
    """
    if viewer.scopes() is None:
        return []
    index = index or SecretIndex()
    index.refresh(memory)
    leaked: list[str] = []
    for phrase, owners in index.hits(tokenize(text)).items():
        hidden = [fid for fid, scope in owners if not viewer.can_see(scope)]
        if not hidden or _visible_elsewhere(memory, viewer, phrase):
            continue
        leaked += [fid for fid in hidden if fid not in leaked]
    return memory.store.query_facts(ids=leaked) if leaked else []


def _visible_elsewhere(memory: Memory, viewer: Viewer, phrase: tuple[str, ...]) -> bool:
    candidates = memory.store.texts_with_phrase(list(phrase), scopes=viewer.scopes())
    return any(_contains(tokenize(text), phrase) for text in candidates)


# -- the context pack ----------------------------------------------------------------------

def build_context(memory: Memory, viewer: Viewer, message: str, *, k: int = 8,
                  repeat_after: timedelta | None = None) -> ContextPack:
    result = memory.search(message, viewer=viewer, k=k)
    contact = viewer.node_id if viewer.kind == "contact" else None
    since = memory.now() - repeat_after if repeat_after is not None else None
    facts = tuple(
        ContextFact(hit.fact, told_at(memory, contact, hit.fact, since) if contact else None,
                    attribution(memory, hit.fact))
        for hit in result.hits)
    corrections = tuple(corrections_owed(memory, contact)) if contact else ()
    text = _render(memory, facts, corrections)
    return ContextPack(viewer, facts, corrections, result.mode, result.degraded, text)


def attribution(memory: Memory, fact: Fact) -> str | None:
    if fact.asserted_by in ("owner", "system", "import"):
        return None
    node = memory.store.get_node(fact.asserted_by)
    if node is not None and node.role == "owner":
        return None
    return node.name if node is not None else fact.asserted_by


def _render(memory: Memory, facts: tuple[ContextFact, ...],
            corrections: tuple[Correction, ...]) -> str:
    lines: list[str] = []
    if corrections:
        lines.append("Corrections to make first (this person was told something outdated):")
        for c in corrections:
            was = value_text(memory, c.told)
            if c.current is None:
                lines.append(f'- On {to_text(c.told_at)} you said "{was}"; that is no '
                             f"longer true ({c.reason}).")
            else:
                lines.append(f'- On {to_text(c.told_at)} you said "{was}"; it is now '
                             f'"{value_text(memory, c.current)}" ({c.reason}).')
        lines.append("")
    as_of = to_text(memory.now())
    if not facts:
        lines.append(f"Facts: none of the stored facts match this message (as of {as_of}).")
        return "\n".join(lines)
    lines.append(f"Facts you may use with this person (as of {as_of}):")
    for cf in facts:
        notes = []
        if cf.attribution:
            notes.append(f"according to {cf.attribution}")
        if cf.told_at is not None:
            notes.append(f"already told {to_text(cf.told_at)}; don't repeat unless asked")
        suffix = f" [{'; '.join(notes)}]" if notes else ""
        lines.append(f"- {cf.fact.text}{suffix}")
    return "\n".join(lines)


def value_text(memory: Memory, fact: Fact) -> str:
    return fact.value if fact.value is not None else memory.node(fact.object or "").name
