"""The memory facade: temporal fact rules, contacts and per-sender views.

Writing a fact for a single-valued predicate is interval arithmetic, not an overwrite.
When a new fact ``N`` valid over ``[s, e)`` arrives, every believed fact ``X`` for the same
key whose interval overlaps it is ended, and the parts of ``X`` that ``N`` does not cover
are re-recorded as *remainder* rows carrying X's value:

* ``X = [a, ∞)``, ``N = [s, ∞)`` with ``a < s``: a change. Remainder ``[a, s)``.
* ``X = [a, ∞)``, ``N = [s, ∞)`` with ``s <= a``: a correction. No remainder; X was wrong.
* ``X = [a, ∞)``, ``N = [s, e)`` inside it: a temporary override (holiday hours).
  Remainders ``[a, s)`` and ``[e, ∞)``, so the regular value resumes on its own.

Rows are never edited apart from being ended once, which is what lets ``known_at`` queries
reconstruct exactly what the memory believed at any earlier moment.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from . import audience, consolidation, vault
from .embeddings import CircuitBreaker, Embedder, EmbeddingError, VectorIndex, to_blob
from .handles import canonical_handle
from .ids import IdFactory, RandomIds, slugify
from .model import (
    ACTIVE,
    BELIEVED,
    CORRECTED,
    DEFAULT_CARDINALITY,
    EXPIRED,
    MANY,
    ONE,
    PUBLIC,
    RETRACTED,
    SUPERSEDED,
    Disclosure,
    Fact,
    Node,
    Viewer,
    normalize_predicate,
    normalize_value,
    private_scope,
    render_fact_text,
    scope_audience,
    validate_node_id,
    validate_scope,
)
from .retrieval import EntityMatcher, Hit, SearchResult, mentions_self, rrf
from .store import GraphStore, SQLiteGraphStore, text_hash
from .textindex import query_terms
from .timeutil import Clock, SystemClock, ensure_aware

log = logging.getLogger("tgmem")

_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,63}$")
ROLES = ("owner", "contact")
CANDIDATE_POOL = 50  # per-signal list length fed into rank fusion


@dataclass(frozen=True)
class ReindexReport:
    embedded: int  # texts embedded by this run
    pending: int  # believed texts still without a vector for the current model
    error: str | None = None  # why the run stopped early, if it did


class NotFound(KeyError):
    """Unknown node or fact. Facts a viewer may not see raise this too, with the same
    message as a missing one, so errors never confirm that something exists."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "not found"


class ScopeError(PermissionError):
    """The viewer is not allowed to perform this write."""


class Memory:
    """Long-term memory over a ``GraphStore``.

    ``embedder`` is optional. Without one, search is keyword + graph; with one, vectors
    join in, and any embedder failure degrades to keyword search instead of raising.
    """

    def __init__(self, store: GraphStore, *, clock: Clock | None = None,
                 ids: IdFactory | None = None, embedder: Embedder | None = None,
                 breaker: CircuitBreaker | None = None,
                 default_country_code: str | None = None) -> None:
        self.store = store
        self.clock = clock or SystemClock()
        self.ids = ids or RandomIds()
        self.embedder = embedder
        self.breaker = breaker or CircuitBreaker(clock=self.clock)
        self.default_country_code = default_country_code
        self._index = VectorIndex(store, embedder.model_id) if embedder is not None else None
        self._entities = EntityMatcher()
        self._secrets = audience.SecretIndex()
        self._bulk_depth = 0

    @classmethod
    def open(cls, path: str | Path, *, fts: bool | None = None, **kwargs) -> Memory:
        return cls(SQLiteGraphStore(path, fts=fts), **kwargs)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def now(self) -> datetime:
        """The current time, never earlier than the latest recorded write."""
        wall = ensure_aware(self.clock.now())
        last = self.store.last_tx_time()
        return max(wall, last) if last is not None else wall

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """Run many writes in one transaction: all of them are stored, or none. Embedding
        is deferred to the end of the outermost batch, outside the transaction."""
        self._bulk_depth += 1
        try:
            with self.store.transaction():
                yield
        finally:
            self._bulk_depth -= 1
        if not self._bulk_depth and self.embedder is not None:
            self.reindex()

    # -- predicates ---------------------------------------------------------------------

    def define_predicate(self, name: str, cardinality: str) -> None:
        self.store.put_predicate(normalize_predicate(name), cardinality)

    def cardinality(self, predicate: str) -> str:
        name = normalize_predicate(predicate)
        return self.store.predicates().get(name) or DEFAULT_CARDINALITY.get(name, ONE)

    # -- nodes --------------------------------------------------------------------------

    def node(self, node_id: str) -> Node:
        found = self.store.get_node(node_id)
        if found is None:
            raise NotFound(f"unknown node {node_id!r}")
        return found

    def nodes(self) -> list[Node]:
        return self.store.nodes()

    def add_entity(self, name: str, kind: str = "thing", *,
                   aliases: Sequence[str] = ()) -> Node:
        """Create a new node with an id derived from its name."""
        with self.store.transaction():
            node = Node(id=self._unused_id(name), name=_clean_name(name), kind=_kind(kind),
                        created_at=self.store.next_tx_time(self.clock.now()),
                        aliases=_clean_aliases(aliases))
            self.store.put_node(node)
        return node

    def upsert_entity(self, node_id: str, name: str, kind: str = "thing", *,
                      aliases: Sequence[str] = ()) -> Node:
        """Create the node, or update its name, kind and aliases (handles and role stay)."""
        validate_node_id(node_id)
        renamed = False
        with self.store.transaction():
            existing = self.store.get_node(node_id)
            if existing is None:
                node = Node(id=node_id, name=_clean_name(name), kind=_kind(kind),
                            created_at=self.store.next_tx_time(self.clock.now()),
                            aliases=_clean_aliases(aliases))
                self.store.put_node(node)
                return node
            node = replace(existing, name=_clean_name(name), kind=_kind(kind),
                           aliases=_clean_aliases(aliases))
            self.store.put_node(node)
            if (node.name, node.aliases) != (existing.name, existing.aliases):
                self._refresh_texts(node_id)
                renamed = True
        if renamed:
            self._backfill_quietly()
        return node

    def add_contact(self, name: str, handles: Sequence[str], *, role: str = "contact",
                    id: str | None = None, aliases: Sequence[str] = (),
                    kind: str = "person") -> Node:
        """Register someone the assistant talks to. Handles are canonicalized exactly the
        way incoming senders are, and each handle can belong to only one node."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        canonical = []
        for raw in handles:
            handle = canonical_handle(raw, default_country_code=self.default_country_code)
            if handle is None:
                raise ValueError(f"{raw!r} is not a usable phone number or email address")
            canonical.append(handle)
        with self.store.transaction():
            if id is not None:
                validate_node_id(id)
                if self.store.get_node(id) is not None:
                    raise ValueError(f"node {id!r} already exists")
            node = Node(id=id or self._unused_id(name), name=_clean_name(name), kind=_kind(kind),
                        created_at=self.store.next_tx_time(self.clock.now()),
                        aliases=_clean_aliases(aliases), handles=tuple(sorted(set(canonical))),
                        role=role)
            self.store.put_node(node)
        return node

    def _unused_id(self, name: str) -> str:
        base = slugify(name)
        candidate, n = base, 1
        while self.store.get_node(candidate) is not None:
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    def _refresh_texts(self, node_id: str) -> None:
        for fact in self.store.query_facts(node=node_id):
            text = self._render_text(fact.subject, fact.predicate, fact.value, fact.object)
            if text != fact.text:
                self.store.set_fact_text(fact.id, text)

    def _render_text(self, subject: str, predicate: str, value: str | None,
                     obj: str | None) -> str:
        """The sentence that is keyword-indexed and embedded for a fact."""
        return render_fact_text(self.node(subject), predicate, value,
                                self.node(obj) if obj is not None else None)

    # -- identity -----------------------------------------------------------------------

    def viewer_for(self, handle: str) -> Viewer:
        """Map a transport-verified sender handle to a viewer. Anything unrecognized gets
        the anonymous (public-only) view; there is no fuzzy matching."""
        canonical = canonical_handle(handle, default_country_code=self.default_country_code)
        node = self.store.node_for_handle(canonical) if canonical else None
        if node is None:
            return Viewer.anonymous()
        if node.role == "owner":
            return Viewer.owner(node.id)
        if node.role == "contact":
            return Viewer.contact(node.id)
        return Viewer.anonymous()

    def for_sender(self, handle: str) -> ScopedMemory:
        return ScopedMemory(self, self.viewer_for(handle))

    def for_viewer(self, viewer: Viewer) -> ScopedMemory:
        return ScopedMemory(self, viewer)

    # -- writing facts ------------------------------------------------------------------

    def remember(self, subject: str, predicate: str, value: str | None = None, *,
                 object: str | None = None, scope: str | None = None,
                 valid_from: datetime | None = None, valid_to: datetime | None = None,
                 ttl: timedelta | None = None, asserted_by: str = "owner",
                 source: str | None = None) -> Fact:
        """Record ``subject --predicate--> value`` (or ``--> object``).

        Without ``scope``, facts about a contact are private to that contact and other
        facts are public. For single-valued predicates the new fact supersedes whatever it
        overlaps (see the module docstring); repeating a value already in force is a no-op.
        """
        predicate = normalize_predicate(predicate)
        value = normalize_value(value) if value is not None else None
        if (value is None) == (object is None):
            raise ValueError("pass exactly one of value or object")
        if valid_to is not None and ttl is not None:
            raise ValueError("pass valid_to or ttl, not both")
        asserted_by = _tag(asserted_by, "asserted_by")
        source = _tag(source, "source") if source is not None else None
        with self.store.transaction():
            subject_node = self.node(subject)
            if object is not None:
                self.node(object)
            scope = self._check_scope(scope or _default_scope(subject_node))
            now = self.store.next_tx_time(self.clock.now())
            start = ensure_aware(valid_from) if valid_from is not None else now
            end = ensure_aware(valid_to) if valid_to is not None else (
                start + ttl if ttl is not None else None)
            if end is not None and end <= start:
                raise ValueError("valid_to must be later than valid_from")
            current = self.store.query_facts(subject=subject, predicate=predicate,
                                             scopes=[scope], statuses=BELIEVED)
            for existing in current:
                if (existing.value, existing.object) == (value, object) and _covers(
                        existing, start, end):
                    return existing
            targets = [] if self.cardinality(predicate) == MANY else [
                f for f in current if _overlaps(f, start, end)]
            fact = self._new_fact(subject, predicate, value, object, scope, start, end, now,
                                  asserted_by, source)
            self.store.insert_fact(fact)
            self._override(targets, fact, now)
        self._embed_quietly({text_hash(fact.text): fact.text})
        return fact

    def correct(self, fact_id: str, value: str | None = None, *, object: str | None = None,
                valid_from: datetime | None = None, asserted_by: str = "owner",
                source: str | None = None) -> Fact:
        """Replace a fact's value. By default the replacement covers the fact's whole
        interval (it was wrong); with a later ``valid_from`` it is a change from then on."""
        value = normalize_value(value) if value is not None else None
        if (value is None) == (object is None):
            raise ValueError("pass exactly one of value or object")
        asserted_by = _tag(asserted_by, "asserted_by")
        source = _tag(source, "source") if source is not None else None
        with self.store.transaction():
            old = self._believed(fact_id)
            if object is not None:
                self.node(object)
            now = self.store.next_tx_time(self.clock.now())
            start = ensure_aware(valid_from) if valid_from is not None else old.valid_from
            end = old.valid_to
            if end is not None and end <= start:
                raise ValueError("the correction would start after the fact ends")
            if (old.value, old.object) == (value, object) and _covers(old, start, end):
                return old
            if self.cardinality(old.predicate) == MANY:
                targets = [old]
            else:
                current = self.store.query_facts(subject=old.subject, predicate=old.predicate,
                                                 scopes=[old.scope], statuses=BELIEVED)
                targets = [f for f in current if _overlaps(f, start, end)]
            fact = self._new_fact(old.subject, old.predicate, value, object, old.scope, start,
                                  end, now, asserted_by, source)
            self.store.insert_fact(fact)
            self._override(targets, fact, now)
        self._embed_quietly({text_hash(fact.text): fact.text})
        return fact

    def retract(self, fact_id: str) -> Fact:
        """Withdraw a fact with no replacement. History keeps it; queries stop seeing it."""
        with self.store.transaction():
            self._believed(fact_id)
            now = self.store.next_tx_time(self.clock.now())
            self.store.end_fact(fact_id, status=RETRACTED, ended_at=now)
        return self.fact(fact_id)

    def purge(self, fact_id: str) -> list[str]:
        """Erase a value completely: the fact, every row derived from the same assertion,
        their disclosures, and cached embeddings no other fact uses. Unlike ``retract``
        this destroys history; use it for erasure requests."""
        with self.store.transaction():
            start = self.fact(fact_id)
            family = {f.id: f for f in self.store.query_facts(
                subject=start.subject, predicate=start.predicate, scopes=[start.scope])}
            children: dict[str, list[str]] = {}
            for f in family.values():
                if f.derived_from:
                    children.setdefault(f.derived_from, []).append(f.id)
            lineage: set[str] = set()
            queue = [start.id]
            while queue:
                current = queue.pop()
                if current in lineage or current not in family:
                    continue
                lineage.add(current)
                parent = family[current].derived_from
                if parent:
                    queue.append(parent)
                queue.extend(children.get(current, []))
            hashes = {text_hash(family[i].text) for i in lineage}
            for purged in sorted(lineage):
                self.store.delete_fact(purged)
            unused = {digest for digest in hashes if not self.store.hash_in_use(digest)}
            for digest in unused:
                self.store.delete_embeddings(digest)
        if self._index is not None:
            self._index.forget(unused)
        return sorted(lineage)

    def _believed(self, fact_id: str) -> Fact:
        fact = self.fact(fact_id)
        if not fact.believed:
            raise ValueError(f"fact {fact_id!r} is {fact.status}; only believed facts change")
        return fact

    def _new_fact(self, subject: str, predicate: str, value: str | None, obj: str | None,
                  scope: str, start: datetime, end: datetime | None, now: datetime,
                  asserted_by: str, source: str | None) -> Fact:
        lapsed = end is not None and end <= now
        return Fact(id=self.ids.new("f"), subject=subject, predicate=predicate, value=value,
                    object=obj, scope=scope, valid_from=start, valid_to=end, recorded_at=now,
                    asserted_by=asserted_by, status=EXPIRED if lapsed else ACTIVE,
                    ended_at=now if lapsed else None, source=source,
                    text=self._render_text(subject, predicate, value, obj))

    def _override(self, targets: Sequence[Fact], new: Fact, now: datetime) -> None:
        for old in targets:
            pieces: list[tuple[datetime, datetime | None]] = []
            if old.valid_from < new.valid_from:
                pieces.append((old.valid_from, new.valid_from))
            if new.valid_to is not None and (old.valid_to is None or new.valid_to < old.valid_to):
                pieces.append((new.valid_to, old.valid_to))
            self.store.end_fact(old.id, status=SUPERSEDED if pieces else CORRECTED,
                                ended_at=now, superseded_by=new.id)
            for lo, hi in pieces:
                lapsed = hi is not None and hi <= now
                self.store.insert_fact(replace(
                    old, id=self.ids.new("f"), valid_from=lo, valid_to=hi, recorded_at=now,
                    status=EXPIRED if lapsed else ACTIVE, ended_at=now if lapsed else None,
                    superseded_by=None, derived_from=old.id))

    def _check_scope(self, scope: str) -> str:
        scope = validate_scope(scope)
        audience = scope_audience(scope)
        if audience is not None and not self.node(audience).is_contact:
            raise ValueError(f"{audience!r} is not a contact; private scopes need a contact")
        return scope

    # -- reading facts ------------------------------------------------------------------

    def fact(self, fact_id: str) -> Fact:
        found = self.store.get_fact(fact_id)
        if found is None:
            raise NotFound(f"unknown fact {fact_id!r}")
        return found

    def facts(self, subject: str | None = None, predicate: str | None = None, *,
              viewer: Viewer | None = None, as_of: datetime | None = None,
              known_at: datetime | None = None) -> list[Fact]:
        """Facts valid at ``as_of`` (default: now) as currently believed, or as believed at
        ``known_at``."""
        viewer = viewer or Viewer.owner()
        predicate = normalize_predicate(predicate) if predicate else None
        when = ensure_aware(as_of) if as_of is not None else self.now()
        if known_at is not None:
            return self.store.facts_known_at(ensure_aware(known_at), valid_at=when,
                                             subject=subject, predicate=predicate,
                                             scopes=viewer.scopes())
        return self.store.query_facts(subject=subject, predicate=predicate,
                                      scopes=viewer.scopes(), statuses=BELIEVED, valid_at=when)

    def timeline(self, subject: str, predicate: str | None = None, *,
                 viewer: Viewer | None = None) -> list[Fact]:
        """Everything currently believed about ``subject`` across all of valid time."""
        viewer = viewer or Viewer.owner()
        return self.store.query_facts(
            subject=subject, predicate=normalize_predicate(predicate) if predicate else None,
            scopes=viewer.scopes(), statuses=BELIEVED)

    def history(self, subject: str, predicate: str | None = None) -> list[Fact]:
        """Every row ever recorded about ``subject``, including ended ones (owner only)."""
        return self.store.query_facts(
            subject=subject, predicate=normalize_predicate(predicate) if predicate else None)

    def neighbors(self, node_id: str, *, viewer: Viewer | None = None,
                  as_of: datetime | None = None) -> list[tuple[Fact, Node]]:
        """Believed node-to-node facts touching ``node_id``, with the node at the other end."""
        viewer = viewer or Viewer.owner()
        when = ensure_aware(as_of) if as_of is not None else self.now()
        out = []
        for fact in self.store.query_facts(node=node_id, scopes=viewer.scopes(),
                                           statuses=BELIEVED, valid_at=when):
            if fact.object is None:
                continue
            other = fact.object if fact.subject == node_id else fact.subject
            out.append((fact, self.node(other)))
        return out

    # -- retrieval ----------------------------------------------------------------------

    def search(self, query: str, *, viewer: Viewer | None = None,
               as_of: datetime | None = None, k: int = 8) -> SearchResult:
        """Hybrid search over the facts ``viewer`` may see that are valid at ``as_of``.

        Scoping is applied inside every signal before ranking, so a viewer's top-k is
        never computed over facts they cannot see.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        viewer = viewer or Viewer.owner()
        when = ensure_aware(as_of) if as_of is not None else self.now()
        scopes = viewer.scopes()
        ranked: dict[str, list[str]] = {
            "keyword": [fid for fid, _ in self.store.keyword_search(
                query_terms(query), scopes=scopes, valid_at=when, limit=CANDIDATE_POOL)]}
        mode, degraded = "keyword", None
        if self.embedder is not None and self._index is not None:
            if not self.breaker.allow():
                degraded = "vector search paused after repeated embedder failures"
            else:
                try:
                    [vector] = self._call_embedder([query])
                except Exception as exc:  # any embedder failure degrades, never raises
                    self.breaker.record_failure()
                    degraded = f"vector search unavailable: {_describe(exc)}"
                else:
                    self.breaker.record_success()
                    candidates = self.store.search_candidates(scopes=scopes, valid_at=when)
                    ranked["vector"] = [fid for fid, _ in self._index.top(
                        vector, candidates, CANDIDATE_POOL)]
                    mode = "hybrid"
        graph = self._graph_ranking(query, viewer, when)
        if graph:
            ranked["graph"] = graph
        fused = rrf({name: ids for name, ids in ranked.items() if ids})[:k]
        found = {f.id: f for f in self.store.query_facts(ids=[fid for fid, _, _ in fused])}
        hits = []
        for fact_id, score, signals in fused:
            fact = found.get(fact_id)
            if fact is None:
                continue
            if not viewer.can_see(fact.scope):  # defence in depth; should be unreachable
                log.error("dropped out-of-scope search hit %s", fact_id)
                continue
            hits.append(Hit(fact, score, signals))
        return SearchResult(tuple(hits), mode, degraded)

    def _graph_ranking(self, query: str, viewer: Viewer, when: datetime) -> list[str]:
        """Facts about entities the query names (and, for "my"/"me", the sender), then
        facts about entities one visible relationship away."""
        self._entities.refresh(self.store.nodes_version(), self.store.nodes)
        anchors = self._entities.mentions(query)
        if viewer.kind == "contact" and viewer.node_id and mentions_self(query) and (
                viewer.node_id not in anchors):
            anchors.insert(0, viewer.node_id)
        if not anchors:
            return []
        scopes = viewer.scopes()
        direct: list[Fact] = []
        neighbours: list[str] = []
        for anchor in anchors:
            for fact in self.store.query_facts(node=anchor, scopes=scopes, statuses=BELIEVED,
                                               valid_at=when):
                direct.append(fact)
                other = fact.object if fact.subject == anchor else fact.subject
                if fact.object is not None and other not in anchors and other not in neighbours:
                    neighbours.append(other)
        hop: list[Fact] = []
        for node_id in neighbours:
            hop += self.store.query_facts(subject=node_id, scopes=scopes, statuses=BELIEVED,
                                          valid_at=when)

        def recency(fact: Fact) -> tuple:
            return (fact.subject not in anchors, -fact.recorded_at.timestamp(), fact.id)

        ordered = sorted(direct, key=recency) + sorted(hop, key=recency)
        return list(dict.fromkeys(f.id for f in ordered))[:CANDIDATE_POOL]

    def reindex(self, *, batch_size: int = 64) -> ReindexReport:
        """Embed every believed fact text that has no vector for the current model yet
        (facts written during an outage, after a rename, or under a previous model)."""
        if self.embedder is None or self._index is None:
            return ReindexReport(0, 0, "no embedder configured")
        model = self.embedder.model_id
        embedded, error, attempted = 0, None, set()
        while True:
            batch = [(digest, text) for digest, text in
                     self.store.texts_missing_embedding(model, limit=batch_size)
                     if digest not in attempted]
            if not batch:
                break
            attempted.update(digest for digest, _ in batch)
            if not self.breaker.allow():
                error = "embedder paused after repeated failures"
                break
            try:
                vectors = self._call_embedder([text for _, text in batch])
            except Exception as exc:
                self.breaker.record_failure()
                error = _describe(exc)
                break
            self.breaker.record_success()
            self._store_vectors([digest for digest, _ in batch], vectors)
            embedded += len(batch)
        pending = len(self.store.texts_missing_embedding(model))
        return ReindexReport(embedded, pending, error)

    def consolidate(self, *, dry_run: bool = False,
                    merge_threshold: float = consolidation.DEFAULT_MERGE_THRESHOLD,
                    reindex: bool = True) -> consolidation.ConsolidationReport:
        """The nightly job: expire, fold duplicates, merge near-duplicates, backfill
        embeddings. Safe to run at any time and idempotent."""
        return consolidation.consolidate(self, dry_run=dry_run,
                                         merge_threshold=merge_threshold, reindex=reindex)

    def export_vault(self, directory: str | Path) -> vault.ExportReport:
        """Write the Markdown mirror (see ``tgmem.vault``). Unchanged pages are left alone."""
        return vault.export_vault(self.store, directory)

    def stats(self) -> dict[str, object]:
        out: dict[str, object] = dict(self.store.counts())
        model = self.embedder.model_id if self.embedder is not None else None
        out["pending_embeddings"] = len(self.store.texts_missing_embedding(model)) if model else 0
        out["embedder"] = model
        out["keyword_index"] = "fts5" if self.store.fts_enabled else "like"
        return out

    def _embed_quietly(self, items: dict[str, str]) -> int:
        """Best-effort embedding right after a write. Failures leave the text pending."""
        if self.embedder is None or self._index is None or self._bulk_depth:
            return 0
        self._index.refresh()
        missing = {digest: text for digest, text in items.items() if not self._index.has(digest)}
        if not missing or not self.breaker.allow():
            return 0
        try:
            vectors = self._call_embedder(list(missing.values()))
        except Exception as exc:
            self.breaker.record_failure()
            log.warning("embedding failed (%s); %d text(s) stay keyword-only until reindex",
                        _describe(exc), len(missing))
            return 0
        self.breaker.record_success()
        self._store_vectors(list(missing), vectors)
        return len(missing)

    def _backfill_quietly(self) -> None:
        if self.embedder is not None and not self._bulk_depth:
            report = self.reindex()
            if report.error:
                log.warning("embedding backfill stopped: %s", report.error)

    def _call_embedder(self, texts: list[str]) -> list[list[float]]:
        assert self.embedder is not None
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"embedder returned {len(vectors)} vectors for {len(texts)}")
        for vector in vectors:
            if not vector or not all(math.isfinite(x) for x in vector):
                raise EmbeddingError("embedder returned an empty or non-finite vector")
        return vectors

    def _store_vectors(self, digests: list[str], vectors: list[list[float]]) -> None:
        assert self.embedder is not None and self._index is not None
        rows = []
        for digest, vector in zip(digests, vectors, strict=True):
            blob, dim = to_blob(vector)
            rows.append((digest, blob, dim))
        self.store.put_embeddings(self.embedder.model_id, rows)
        self._index.add_texts(dict(zip(digests, vectors, strict=True)))

    # -- audience ledger ----------------------------------------------------------------

    def record_disclosure(self, contact: str, fact_ids: Sequence[str], *,
                          channel: str | None = None,
                          at: datetime | None = None) -> list[Disclosure]:
        """Record that ``contact`` was told these facts. The ledger refuses any fact the
        contact is not allowed to see, so it can never record a cross-person leak."""
        channel = _tag(channel, "channel") if channel is not None else None
        with self.store.transaction():
            node = self.node(contact)
            if node.role != "contact":
                raise ValueError(f"{contact!r} is not a contact; disclosures track contacts")
            viewer = Viewer.contact(contact)
            facts = [self.fact(fact_id) for fact_id in fact_ids]
            for fact in facts:
                if not viewer.can_see(fact.scope):
                    raise ScopeError(f"fact {fact.id!r} is not visible to {contact!r}")
            when = ensure_aware(at) if at is not None else self.store.next_tx_time(
                self.clock.now())
            recorded = [Disclosure(self.ids.new("d"), contact, fact.id, when, channel)
                        for fact in facts]
            for disclosure in recorded:
                self.store.insert_disclosure(disclosure)
        return recorded

    def told(self, contact: str) -> list[Disclosure]:
        return self.store.disclosures(contact=contact)

    def told_at(self, contact: str, fact: Fact, *,
                since: datetime | None = None) -> datetime | None:
        return audience.told_at(self, contact, fact, since)

    def corrections_owed(self, contact: str) -> list[audience.Correction]:
        return audience.corrections_owed(self, contact)

    def mentions(self, text: str, facts: Sequence[Fact]) -> list[Fact]:
        return audience.mentions(self, text, list(facts))

    def leaks(self, viewer: Viewer, text: str) -> list[Fact]:
        return audience.leaks(self, viewer, text, self._secrets)


class ScopedMemory:
    """The memory as one sender may use it. Build it from the transport-verified sender
    (``Memory.for_sender``) and hand only this object to the model's tools: every read is
    filtered to the viewer's scopes, and writes are confined to what the viewer may write.
    """

    def __init__(self, memory: Memory, viewer: Viewer) -> None:
        self.memory = memory
        self.viewer = viewer

    def fact(self, fact_id: str) -> Fact:
        found = self.memory.store.get_fact(fact_id)
        if found is None or not self.viewer.can_see(found.scope):
            raise NotFound(f"unknown fact {fact_id!r}")
        return found

    def facts(self, subject: str | None = None, predicate: str | None = None, *,
              as_of: datetime | None = None) -> list[Fact]:
        return self.memory.facts(subject, predicate, viewer=self.viewer, as_of=as_of)

    def remember(self, predicate: str, value: str | None = None, *, object: str | None = None,
                 subject: str | None = None, scope: str | None = None,
                 valid_from: datetime | None = None, valid_to: datetime | None = None,
                 ttl: timedelta | None = None, source: str | None = None) -> Fact:
        viewer = self.viewer
        if viewer.kind == "anonymous":
            raise ScopeError("unknown senders cannot store facts")
        if viewer.kind == "contact":
            own = private_scope(viewer.node_id or "")
            if scope not in (None, own):
                raise ScopeError("contacts can only store facts private to themselves")
            scope, subject, asserted_by = own, subject or viewer.node_id, viewer.node_id
        else:
            if subject is None:
                raise ValueError("say which node the fact is about (subject=...)")
            asserted_by = viewer.node_id or "owner"
        return self.memory.remember(subject or "", predicate, value, object=object, scope=scope,
                                    valid_from=valid_from, valid_to=valid_to, ttl=ttl,
                                    asserted_by=asserted_by or "owner", source=source)

    def search(self, query: str, *, as_of: datetime | None = None, k: int = 8) -> SearchResult:
        return self.memory.search(query, viewer=self.viewer, as_of=as_of, k=k)

    def mark_told(self, fact_ids: Sequence[str], *,
                  channel: str | None = None) -> list[Disclosure]:
        if self.viewer.kind != "contact" or not self.viewer.node_id:
            raise ScopeError("disclosures are tracked for known contacts only")
        facts = [self.fact(fact_id) for fact_id in fact_ids]  # NotFound if not visible
        return self.memory.record_disclosure(self.viewer.node_id, [f.id for f in facts],
                                             channel=channel)

    def corrections_owed(self) -> list[audience.Correction]:
        if self.viewer.kind != "contact" or not self.viewer.node_id:
            return []
        return self.memory.corrections_owed(self.viewer.node_id)

    def guard(self, reply: str) -> list[Fact]:
        """Facts this viewer may not see whose values appear in ``reply``. Send nothing
        that returns a non-empty list."""
        return self.memory.leaks(self.viewer, reply)

    def context(self, message: str, *, k: int = 8,
                repeat_after: timedelta | None = None) -> audience.ContextPack:
        """Scoped facts for a reply, each marked with when this sender last heard it,
        plus any corrections owed, rendered as a prompt block."""
        return audience.build_context(self.memory, self.viewer, message, k=k,
                                      repeat_after=repeat_after)

    def record_reply(self, reply: str, pack: audience.ContextPack, *,
                     channel: str | None = None) -> list[Disclosure]:
        """After sending ``reply``, mark the facts it actually stated as told."""
        if self.viewer.kind != "contact":
            return []
        offered = [cf.fact for cf in pack.facts] + [
            c.current for c in pack.corrections if c.current is not None]
        said = self.memory.mentions(reply, list({f.id: f for f in offered}.values()))
        return self.mark_told([f.id for f in said], channel=channel) if said else []


def _describe(exc: Exception) -> str:
    """A safe one-line reason: embedder messages are written to exclude credentials;
    anything unexpected is reported by type only."""
    return str(exc) if isinstance(exc, EmbeddingError) else type(exc).__name__


def _default_scope(subject: Node) -> str:
    return private_scope(subject.id) if subject.is_contact else PUBLIC


def _covers(fact: Fact, start: datetime, end: datetime | None) -> bool:
    if fact.valid_from > start:
        return False
    return fact.valid_to is None or (end is not None and end <= fact.valid_to)


def _overlaps(fact: Fact, start: datetime, end: datetime | None) -> bool:
    return (end is None or fact.valid_from < end) and (
        fact.valid_to is None or start < fact.valid_to)


def _tag(value: str, what: str) -> str:
    if not _TAG.match(value):
        raise ValueError(f"invalid {what} {value!r}: use letters, digits and ._:/@+-")
    return value


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("a name cannot be empty")
    return cleaned


def _clean_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(a.split()) for a in aliases if a.strip()}))


def _kind(kind: str) -> str:
    if not re.match(r"^[a-z][a-z0-9_]{0,31}$", kind):
        raise ValueError(f"invalid kind {kind!r}")
    return kind
