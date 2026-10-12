"""Nightly consolidation: expire lapsed facts, fold duplicates, merge near-duplicates, and
backfill embeddings that failed earlier.

Every step is a status change on existing rows (nothing is deleted), runs inside one write
transaction, and is idempotent: a second run right after the first finds nothing to do.
Embedding happens afterwards, outside the transaction, so a slow provider never holds the
database's write lock.

Merging is deliberately narrow. It only touches many-valued predicates (notes, likes) and
only compares facts on the same timeline (``Fact.key``). Nothing is ever merged across
scopes, which would move private text into a shared fact, or between a contact's claims
and the owner's notes. It keeps the most informative wording, and it never touches a note
that contains a negation: word overlap cannot tell "allergic to latex gloves" from "not
allergic to latex gloves", or "needs latex gloves" from "needs latex-free gloves", and
folding one into the other would erase what was said. Exact duplicates are still folded.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .model import ACTIVE, DUPLICATE, EXPIRED, MANY, MERGED, Fact
from .textindex import STOPWORDS, tokenize

if TYPE_CHECKING:
    from .memory import Memory

DEFAULT_MERGE_THRESHOLD = 0.85

# Words that turn a statement around, including contractions typed without an apostrophe
# ("dont", "cant") and the "non-" / "-free" of "non-latex" and "latex-free".
NEGATIONS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor",
    "without", "cannot", "non", "free", "aint", "arent", "cant", "couldnt", "didnt",
    "doesnt", "dont", "hadnt", "hasnt", "havent", "isnt", "mustnt", "neednt", "shouldnt",
    "wasnt", "werent", "wont", "wouldnt",
})
_CONTRACTED = re.compile(r"n['‘’ʼ]t\b", re.IGNORECASE)  # isn't, don’t, can't, won't


def negated(text: str) -> bool:
    """Whether ``text`` contains a negation, so its words alone do not say what it means."""
    if _CONTRACTED.search(unicodedata.normalize("NFKC", text)):
        return True
    return any(word in NEGATIONS for word in tokenize(text))


@dataclass(frozen=True)
class ConsolidationReport:
    ran_at: datetime
    expired: list[str]
    duplicates: list[tuple[str, str]]  # (folded fact, fact it was folded into)
    merged: list[tuple[str, str]]  # (merged fact, fact it was merged into)
    embedded: int
    pending_embeddings: int
    dry_run: bool = False
    embed_error: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.expired or self.duplicates or self.merged or self.embedded)

    def summary(self) -> str:
        prefix = "dry run: " if self.dry_run else ""
        text = (f"{prefix}expired {len(self.expired)}, duplicates {len(self.duplicates)}, "
                f"merged {len(self.merged)}, embedded {self.embedded}, "
                f"pending embeddings {self.pending_embeddings}")
        return text + (f" (embedding stopped: {self.embed_error})" if self.embed_error else "")


class _Rollback(Exception):
    pass


def consolidate(memory: Memory, *, dry_run: bool = False,
                merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
                reindex: bool = True) -> ConsolidationReport:
    store = memory.store
    expired: list[str] = []
    duplicates: list[tuple[str, str]] = []
    merged: list[tuple[str, str]] = []
    try:
        with store.transaction():
            now = store.next_tx_time(memory.clock.now())
            for fact in store.query_facts(statuses=[ACTIVE]):
                if fact.valid_to is not None and fact.valid_to <= now:
                    store.end_fact(fact.id, status=EXPIRED, ended_at=now)
                    expired.append(fact.id)
            by_key: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
            for fact in store.query_facts(statuses=[ACTIVE]):
                by_key[fact.key].append(fact)
            for key, facts in by_key.items():
                remaining = _fold_duplicates(memory, facts, now, duplicates)
                if memory.cardinality(key[1]) == MANY:
                    _merge_similar(memory, remaining, now, merged, merge_threshold)
            if dry_run:
                raise _Rollback
    except _Rollback:
        pass
    embedded, error = 0, None
    if reindex and not dry_run and memory.embedder is not None:
        result = memory.reindex()
        embedded, error = result.embedded, result.error
    pending = int(memory.stats()["pending_embeddings"])  # type: ignore[call-overload]
    return ConsolidationReport(now, expired, duplicates, merged, embedded, pending, dry_run,
                               error)


def _content_key(fact: Fact) -> tuple[str, str]:
    if fact.object is not None:
        return ("node", fact.object)
    text = re.sub(r"\s+", " ", (fact.value or "").casefold()).strip()
    return ("text", text.rstrip(".!?,;: "))


def _overlaps(a: Fact, b: Fact) -> bool:
    return (b.valid_to is None or a.valid_from < b.valid_to) and (
        a.valid_to is None or b.valid_from < a.valid_to)


def _fold_duplicates(memory: Memory, facts: list[Fact], now: datetime,
                     out: list[tuple[str, str]]) -> list[Fact]:
    """Same content, overlapping validity: keep the oldest (its provenance is earliest)."""
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in sorted(facts, key=lambda f: (f.recorded_at, f.id)):
        groups[_content_key(fact)].append(fact)
    remaining: list[Fact] = []
    for group in groups.values():
        kept: list[Fact] = []
        for fact in group:
            target = next((k for k in kept if _overlaps(k, fact)), None)
            if target is None:
                kept.append(fact)
                continue
            memory.store.end_fact(fact.id, status=DUPLICATE, ended_at=now,
                                  superseded_by=target.id)
            out.append((fact.id, target.id))
        remaining += kept
    return remaining


def _merge_similar(memory: Memory, facts: list[Fact], now: datetime,
                   out: list[tuple[str, str]], threshold: float) -> None:
    """Fold a note into a longer one that says the same thing (or more). Notes with a
    negation take no part, as either side."""
    words = {f.id: frozenset(t for t in tokenize(f.value) if t not in STOPWORDS)
             for f in facts if f.value is not None and not negated(f.value)}
    order = sorted((f for f in facts if f.id in words),
                   key=lambda f: (-len(words[f.id]), f.recorded_at, f.id))
    kept: list[Fact] = []
    for fact in order:
        target = next((k for k in kept if _overlaps(k, fact)
                       and _similar(words[fact.id], words[k.id], threshold)), None)
        if target is None:
            kept.append(fact)
            continue
        memory.store.end_fact(fact.id, status=MERGED, ended_at=now, superseded_by=target.id)
        out.append((fact.id, target.id))


def _similar(candidate: frozenset[str], keeper: frozenset[str], threshold: float) -> bool:
    if not candidate or not keeper:
        return False
    jaccard = len(candidate & keeper) / len(candidate | keeper)
    subsumed = len(candidate) >= 3 and candidate <= keeper
    return jaccard >= threshold or subsumed
