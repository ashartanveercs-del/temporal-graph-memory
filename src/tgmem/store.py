"""Graph storage: the ``GraphStore`` contract and its SQLite implementation.

The temporal rules (supersession, corrections, remainders) live in ``memory.py`` and only
use the primitives below, so a different backend has a small, well-defined job: implement
this protocol and pass ``tests/test_store_contract.py``.

SQLite layout: ``nodes`` + ``handles`` for entities and contacts, ``edges`` for facts (each
with ``valid_from``/``valid_to`` and ``recorded_at``/``ended_at``), ``disclosures`` for the
audience ledger, ``embeddings`` as a content-addressed vector cache, and an FTS5 index over
each fact's text.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from .model import BELIEVED, MANY, ONE, STATUSES, Disclosure, Fact, Node
from .textindex import fts_expression, like_pattern
from .timeutil import ensure_aware, from_db, to_db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    aliases    TEXT NOT NULL DEFAULT '[]',
    role       TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS handles (
    handle  TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS handles_node ON handles(node_id);
CREATE TABLE IF NOT EXISTS predicates (
    name        TEXT PRIMARY KEY,
    cardinality TEXT NOT NULL CHECK (cardinality IN ('one', 'many'))
);
CREATE TABLE IF NOT EXISTS edges (
    seq           INTEGER PRIMARY KEY,
    id            TEXT NOT NULL UNIQUE,
    subject       TEXT NOT NULL REFERENCES nodes(id),
    predicate     TEXT NOT NULL,
    value         TEXT,
    object        TEXT REFERENCES nodes(id),
    scope         TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL,
    asserted_by   TEXT NOT NULL,
    status        TEXT NOT NULL,
    ended_at      TEXT,
    superseded_by TEXT REFERENCES edges(id) ON DELETE SET NULL,
    derived_from  TEXT REFERENCES edges(id) ON DELETE SET NULL,
    source        TEXT,
    text          TEXT NOT NULL,
    text_hash     TEXT NOT NULL,
    CHECK ((value IS NULL) <> (object IS NULL)),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);
CREATE INDEX IF NOT EXISTS edges_key ON edges(subject, predicate, scope, status);
CREATE INDEX IF NOT EXISTS edges_object ON edges(object);
CREATE INDEX IF NOT EXISTS edges_status ON edges(status, scope);
CREATE INDEX IF NOT EXISTS edges_hash ON edges(text_hash);
CREATE TABLE IF NOT EXISTS disclosures (
    id      TEXT PRIMARY KEY,
    contact TEXT NOT NULL REFERENCES nodes(id),
    fact_id TEXT NOT NULL REFERENCES edges(id) ON DELETE CASCADE,
    told_at TEXT NOT NULL,
    channel TEXT
);
CREATE INDEX IF NOT EXISTS disclosures_contact ON disclosures(contact, told_at);
CREATE INDEX IF NOT EXISTS disclosures_fact ON disclosures(fact_id);
CREATE TABLE IF NOT EXISTS embeddings (
    seq       INTEGER PRIMARY KEY,
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    UNIQUE (model, text_hash)
);
CREATE INDEX IF NOT EXISTS embeddings_seq ON embeddings(model, seq);
"""

_FTS_TABLE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5("
    "text, tokenize = 'porter unicode61 remove_diacritics 2')"
)

_FACT_COLUMNS = (
    "id, subject, predicate, value, object, scope, valid_from, valid_to, recorded_at, "
    "asserted_by, status, ended_at, superseded_by, derived_from, source, text"
)
_ORDER = "ORDER BY subject, predicate, scope, valid_from, recorded_at, id"
_BELIEVED_SQL = "(" + ", ".join(f"'{s}'" for s in BELIEVED) + ")"


class HandleConflict(ValueError):
    """A handle (phone or email) is already registered to another node."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GraphStore(Protocol):
    """Storage primitives the memory layer is written against."""

    fts_enabled: bool

    def close(self) -> None: ...
    def transaction(self) -> AbstractContextManager[None]: ...
    def next_tx_time(self, candidate: datetime) -> datetime: ...
    def last_tx_time(self) -> datetime | None: ...
    def restore_tx_time(self, stamp: datetime) -> None: ...
    def defer_foreign_keys(self) -> None: ...

    def put_predicate(self, name: str, cardinality: str) -> None: ...
    def predicates(self) -> dict[str, str]: ...

    def put_node(self, node: Node) -> None: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def node_for_handle(self, handle: str) -> Node | None: ...
    def nodes(self) -> list[Node]: ...
    def nodes_version(self) -> int: ...

    def insert_fact(self, fact: Fact) -> None: ...
    def get_fact(self, fact_id: str) -> Fact | None: ...
    def end_fact(self, fact_id: str, *, status: str, ended_at: datetime,
                 superseded_by: str | None = None) -> None: ...
    def set_fact_text(self, fact_id: str, text: str) -> None: ...
    def delete_fact(self, fact_id: str) -> None: ...
    def facts_version(self) -> int: ...
    def fact_values(self) -> list[tuple[str, str, str | None, str | None]]: ...
    def texts_with_phrase(self, tokens: Sequence[str], *, scopes: Sequence[str] | None,
                          limit: int = 50) -> list[str]: ...
    def query_facts(self, *, subject: str | None = None, predicate: str | None = None,
                    node: str | None = None, scopes: Sequence[str] | None = None,
                    statuses: Sequence[str] | None = None, valid_at: datetime | None = None,
                    ids: Sequence[str] | None = None) -> list[Fact]: ...
    def facts_known_at(self, known_at: datetime, *, valid_at: datetime,
                       subject: str | None = None, predicate: str | None = None,
                       scopes: Sequence[str] | None = None) -> list[Fact]: ...

    def insert_disclosure(self, disclosure: Disclosure) -> None: ...
    def disclosures(self, *, contact: str | None = None,
                    fact_ids: Sequence[str] | None = None) -> list[Disclosure]: ...

    def keyword_search(self, terms: Sequence[str], *, scopes: Sequence[str] | None,
                       valid_at: datetime, limit: int) -> list[tuple[str, float]]: ...
    def search_candidates(self, *, scopes: Sequence[str] | None,
                          valid_at: datetime) -> list[tuple[str, str]]: ...

    def put_embeddings(self, model: str, items: Sequence[tuple[str, bytes, int]]) -> None: ...
    def embeddings_after(self, model: str, after: int) -> list[tuple[int, str, int, bytes]]: ...
    def texts_missing_embedding(self, model: str,
                                limit: int | None = None) -> list[tuple[str, str]]: ...
    def delete_embeddings(self, text_hash: str) -> None: ...
    def hash_in_use(self, text_hash: str) -> bool: ...

    def counts(self) -> dict[str, int]: ...


class SQLiteGraphStore:
    """``GraphStore`` on a single SQLite file (or ``":memory:"``).

    Every write primitive runs inside ``transaction()``, which issues ``BEGIN IMMEDIATE``.
    Read-modify-write sequences (look up the current fact, then supersede it) must take the
    write lock *before* reading: two deferred transactions that both read and then both try
    to write deadlock, and SQLite resolves that by failing one immediately, without waiting
    for the busy timeout.
    """

    def __init__(self, path: str | Path, *, fts: bool | None = None,
                 busy_timeout: float = 10.0) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None, timeout=busy_timeout)
        self._depth = 0
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self.fts_enabled = self._setup_fts(fts)

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def _setup_fts(self, requested: bool | None) -> bool:
        if requested is False:
            return False
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'edges_fts'").fetchone() is not None
        try:
            self._conn.execute(_FTS_TABLE)
        except sqlite3.OperationalError:
            if requested:
                raise RuntimeError("this SQLite build has no FTS5 module") from None
            return False
        if not exists:
            # The index was created after facts already existed (e.g. a database first
            # used on a build without FTS5): index what is there.
            self._conn.execute("INSERT INTO edges_fts(rowid, text) SELECT seq, text FROM edges")
        return True

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._depth = 0
            self._conn.execute("ROLLBACK")
            raise
        self._depth = 0
        self._conn.execute("COMMIT")

    # -- meta & transaction time ----------------------------------------------------------

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def last_tx_time(self) -> datetime | None:
        value = self._get_meta("last_tx")
        return from_db(value) if value else None

    def next_tx_time(self, candidate: datetime) -> datetime:
        """A timestamp strictly later than every one handed out before, even if the wall
        clock stalls or jumps backwards. Must be called inside ``transaction()`` so that
        concurrent writers are serialized and timestamps follow commit order."""
        if not self._depth:
            raise RuntimeError("next_tx_time() must be called inside transaction()")
        stamp = ensure_aware(candidate)
        last = self.last_tx_time()
        if last is not None and stamp <= last:
            stamp = last + timedelta(microseconds=1)
        self._set_meta("last_tx", to_db(stamp))
        return stamp

    def nodes_version(self) -> int:
        return int(self._get_meta("nodes_version") or 0)

    def facts_version(self) -> int:
        """Changes whenever any fact's content is inserted, re-rendered or deleted, in
        this process or another one; caches derived from fact text key on it."""
        return int(self._get_meta("facts_version") or 0)

    def _bump_facts_version(self) -> None:
        self._set_meta("facts_version", str(self.facts_version() + 1))

    def restore_tx_time(self, stamp: datetime) -> None:
        """Raise the transaction clock to at least ``stamp`` (used after a rebuild)."""
        with self.transaction():
            last = self.last_tx_time()
            if last is None or ensure_aware(stamp) > last:
                self._set_meta("last_tx", to_db(stamp))

    def defer_foreign_keys(self) -> None:
        """Check foreign keys at commit instead of per statement, so a bulk load can
        insert rows that reference each other in any order."""
        if not self._depth:
            raise RuntimeError("defer_foreign_keys() must be called inside transaction()")
        self._conn.execute("PRAGMA defer_foreign_keys = ON")

    # -- predicates ---------------------------------------------------------------------

    def put_predicate(self, name: str, cardinality: str) -> None:
        if cardinality not in (ONE, MANY):
            raise ValueError(f"cardinality must be {ONE!r} or {MANY!r}")
        with self.transaction():
            self._conn.execute(
                "INSERT INTO predicates(name, cardinality) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET cardinality = excluded.cardinality",
                (name, cardinality))

    def predicates(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT name, cardinality FROM predicates ORDER BY name")
        return dict(rows.fetchall())

    # -- nodes --------------------------------------------------------------------------

    def put_node(self, node: Node) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT INTO nodes(id, kind, name, aliases, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "kind = excluded.kind, name = excluded.name, aliases = excluded.aliases, "
                "role = excluded.role",
                (node.id, node.kind, node.name, json.dumps(list(node.aliases)), node.role,
                 to_db(node.created_at)))
            self._conn.execute("DELETE FROM handles WHERE node_id = ?", (node.id,))
            for handle in node.handles:
                owner = self._conn.execute(
                    "SELECT node_id FROM handles WHERE handle = ?", (handle,)).fetchone()
                if owner is not None:
                    raise HandleConflict(f"handle {handle!r} already belongs to {owner[0]!r}")
                self._conn.execute(
                    "INSERT INTO handles(handle, node_id) VALUES (?, ?)", (handle, node.id))
            self._set_meta("nodes_version", str(self.nodes_version() + 1))

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute(
            "SELECT id, kind, name, aliases, role, created_at FROM nodes WHERE id = ?",
            (node_id,)).fetchone()
        return self._node(row, self._handles_of(node_id)) if row else None

    def node_for_handle(self, handle: str) -> Node | None:
        row = self._conn.execute(
            "SELECT node_id FROM handles WHERE handle = ?", (handle,)).fetchone()
        return self.get_node(row[0]) if row else None

    def nodes(self) -> list[Node]:
        handles: dict[str, list[str]] = {}
        for handle, node_id in self._conn.execute(
                "SELECT handle, node_id FROM handles ORDER BY handle"):
            handles.setdefault(node_id, []).append(handle)
        rows = self._conn.execute(
            "SELECT id, kind, name, aliases, role, created_at FROM nodes ORDER BY id")
        return [self._node(row, handles.get(row[0], [])) for row in rows]

    def _handles_of(self, node_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT handle FROM handles WHERE node_id = ? ORDER BY handle", (node_id,))
        return [r[0] for r in rows]

    @staticmethod
    def _node(row: tuple, handles: list[str]) -> Node:
        node_id, kind, name, aliases, role, created_at = row
        return Node(id=node_id, name=name, kind=kind, created_at=from_db(created_at),
                    aliases=tuple(json.loads(aliases)), handles=tuple(handles), role=role)

    # -- facts --------------------------------------------------------------------------

    def insert_fact(self, fact: Fact) -> None:
        values = (
            fact.id, fact.subject, fact.predicate, fact.value, fact.object, fact.scope,
            to_db(fact.valid_from), _opt(fact.valid_to), to_db(fact.recorded_at),
            fact.asserted_by, fact.status, _opt(fact.ended_at), fact.superseded_by,
            fact.derived_from, fact.source, fact.text, text_hash(fact.text))
        with self.transaction():
            try:
                cursor = self._conn.execute(
                    f"INSERT INTO edges({_FACT_COLUMNS}, text_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"cannot store fact {fact.id!r}: {exc}") from None
            if self.fts_enabled:
                self._conn.execute("INSERT INTO edges_fts(rowid, text) VALUES (?, ?)",
                                   (cursor.lastrowid, fact.text))
            self._bump_facts_version()

    def get_fact(self, fact_id: str) -> Fact | None:
        found = self.query_facts(ids=[fact_id])
        return found[0] if found else None

    def end_fact(self, fact_id: str, *, status: str, ended_at: datetime,
                 superseded_by: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self.transaction():
            self._conn.execute(
                "UPDATE edges SET status = ?, ended_at = ?, superseded_by = ? WHERE id = ?",
                (status, to_db(ended_at), superseded_by, fact_id))

    def set_fact_text(self, fact_id: str, text: str) -> None:
        with self.transaction():
            self._conn.execute("UPDATE edges SET text = ?, text_hash = ? WHERE id = ?",
                               (text, text_hash(text), fact_id))
            if self.fts_enabled:
                self._conn.execute(
                    "UPDATE edges_fts SET text = ? "
                    "WHERE rowid = (SELECT seq FROM edges WHERE id = ?)", (text, fact_id))
            self._bump_facts_version()

    def delete_fact(self, fact_id: str) -> None:
        with self.transaction():
            row = self._conn.execute("SELECT seq FROM edges WHERE id = ?",
                                     (fact_id,)).fetchone()
            if row is None:
                return
            if self.fts_enabled:
                self._conn.execute("DELETE FROM edges_fts WHERE rowid = ?", (row[0],))
            self._conn.execute("DELETE FROM edges WHERE id = ?", (fact_id,))
            self._bump_facts_version()

    def query_facts(self, *, subject: str | None = None, predicate: str | None = None,
                    node: str | None = None, scopes: Sequence[str] | None = None,
                    statuses: Sequence[str] | None = None, valid_at: datetime | None = None,
                    ids: Sequence[str] | None = None) -> list[Fact]:
        where: list[str] = []
        params: list[object] = []
        if subject is not None:
            where.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            where.append("predicate = ?")
            params.append(predicate)
        if node is not None:
            where.append("(subject = ? OR object = ?)")
            params += [node, node]
        if ids is not None:
            where.append(_in("id", ids, params))
        if statuses is not None:
            where.append(_in("status", statuses, params))
        if scopes is not None:
            where.append(_in("scope", scopes, params))
        if valid_at is not None:
            where.append(_valid_clause(valid_at, params))
        sql = f"SELECT {_FACT_COLUMNS} FROM edges"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [_fact(row) for row in self._conn.execute(f"{sql} {_ORDER}", params)]

    def fact_values(self) -> list[tuple[str, str, str | None, str | None]]:
        """``(id, scope, value, object)`` for every fact, without building ``Fact``s."""
        return list(self._conn.execute("SELECT id, scope, value, object FROM edges ORDER BY id"))

    def texts_with_phrase(self, tokens: Sequence[str], *, scopes: Sequence[str] | None,
                          limit: int = 50) -> list[str]:
        """Texts of facts (any status) in ``scopes`` that contain all of ``tokens``,
        as a phrase when FTS5 is available. Callers re-check exact word order."""
        if not tokens:
            return []
        params: list[object] = []
        scope_filter = f" AND {_in('e.scope', scopes, params)}" if scopes is not None else ""
        if self.fts_enabled:
            phrase = '"' + " ".join(tokens) + '"'
            sql = ("SELECT e.text FROM edges_fts JOIN edges e ON e.seq = edges_fts.rowid "
                   f"WHERE edges_fts MATCH ?{scope_filter} ORDER BY e.id LIMIT ?")
            return [row[0] for row in self._conn.execute(sql, [phrase, *params, limit])]
        likes = " AND ".join("e.text LIKE ? ESCAPE '\\'" for _ in tokens)
        sql = f"SELECT e.text FROM edges e WHERE {likes}{scope_filter} ORDER BY e.id LIMIT ?"
        return [row[0] for row in self._conn.execute(
            sql, [*(like_pattern(t) for t in tokens), *params, limit])]

    def facts_known_at(self, known_at: datetime, *, valid_at: datetime,
                       subject: str | None = None, predicate: str | None = None,
                       scopes: Sequence[str] | None = None) -> list[Fact]:
        """What the memory believed at ``known_at`` about the world at ``valid_at``.

        A row was believed from ``recorded_at`` until ``ended_at``. Expired rows are the
        exception: expiry means "this interval is over", not "this was wrong", so they
        stay part of the belief after being marked.
        """
        k = to_db(known_at)
        params: list[object] = [k, k]
        where = ["recorded_at <= ?",
                 "(ended_at IS NULL OR ended_at > ? OR status = 'expired')",
                 _valid_clause(valid_at, params)]
        if subject is not None:
            where.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            where.append("predicate = ?")
            params.append(predicate)
        if scopes is not None:
            where.append(_in("scope", scopes, params))
        sql = f"SELECT {_FACT_COLUMNS} FROM edges WHERE {' AND '.join(where)} {_ORDER}"
        return [_fact(row) for row in self._conn.execute(sql, params)]

    # -- disclosures --------------------------------------------------------------------

    def insert_disclosure(self, disclosure: Disclosure) -> None:
        with self.transaction():
            try:
                self._conn.execute(
                    "INSERT INTO disclosures(id, contact, fact_id, told_at, channel) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (disclosure.id, disclosure.contact, disclosure.fact_id,
                     to_db(disclosure.told_at), disclosure.channel))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"cannot store disclosure {disclosure.id!r}: {exc}") from None

    def disclosures(self, *, contact: str | None = None,
                    fact_ids: Sequence[str] | None = None) -> list[Disclosure]:
        where: list[str] = []
        params: list[object] = []
        if contact is not None:
            where.append("contact = ?")
            params.append(contact)
        if fact_ids is not None:
            where.append(_in("fact_id", fact_ids, params))
        sql = "SELECT id, contact, fact_id, told_at, channel FROM disclosures"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._conn.execute(sql + " ORDER BY told_at, id", params)
        return [Disclosure(i, c, f, from_db(t), ch) for i, c, f, t, ch in rows]

    # -- search -------------------------------------------------------------------------

    def keyword_search(self, terms: Sequence[str], *, scopes: Sequence[str] | None,
                       valid_at: datetime, limit: int) -> list[tuple[str, float]]:
        """Best keyword matches among believed facts valid at ``valid_at`` that the scopes
        allow. Filtering happens inside the query, before ranking and ``LIMIT``, so a
        viewer's top-k can never be computed over facts they are not allowed to see."""
        if not terms:
            return []
        params: list[object] = []
        filters = [f"e.status IN {_BELIEVED_SQL}", _valid_clause(valid_at, params, "e.")]
        if scopes is not None:
            filters.append(_in("e.scope", scopes, params))
        if self.fts_enabled:
            sql = (
                "SELECT e.id, bm25(edges_fts) AS rank FROM edges_fts "
                "JOIN edges e ON e.seq = edges_fts.rowid "
                f"WHERE edges_fts MATCH ? AND {' AND '.join(filters)} "
                "ORDER BY rank, e.id LIMIT ?")
            rows = self._conn.execute(sql, [fts_expression(list(terms)), *params, limit])
            return [(fact_id, -rank) for fact_id, rank in rows]
        hits = " + ".join("(e.text LIKE ? ESCAPE '\\')" for _ in terms)
        sql = (
            f"SELECT id, hits FROM (SELECT e.id AS id, e.seq AS seq, {hits} AS hits "
            f"FROM edges e WHERE {' AND '.join(filters)}) "
            "WHERE hits > 0 ORDER BY hits DESC, id LIMIT ?")
        rows = self._conn.execute(sql, [*(like_pattern(t) for t in terms), *params, limit])
        return [(fact_id, float(count)) for fact_id, count in rows]

    def search_candidates(self, *, scopes: Sequence[str] | None,
                          valid_at: datetime) -> list[tuple[str, str]]:
        params: list[object] = []
        filters = [f"status IN {_BELIEVED_SQL}", _valid_clause(valid_at, params)]
        if scopes is not None:
            filters.append(_in("scope", scopes, params))
        sql = f"SELECT id, text_hash FROM edges WHERE {' AND '.join(filters)} ORDER BY id"
        return list(self._conn.execute(sql, params))

    # -- embeddings ---------------------------------------------------------------------

    def put_embeddings(self, model: str, items: Sequence[tuple[str, bytes, int]]) -> None:
        with self.transaction():
            self._conn.executemany(
                "INSERT OR IGNORE INTO embeddings(model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)", [(model, h, dim, blob) for h, blob, dim in items])

    def embeddings_after(self, model: str, after: int) -> list[tuple[int, str, int, bytes]]:
        rows = self._conn.execute(
            "SELECT seq, text_hash, dim, vector FROM embeddings "
            "WHERE model = ? AND seq > ? ORDER BY seq", (model, after))
        return list(rows)

    def texts_missing_embedding(self, model: str,
                                limit: int | None = None) -> list[tuple[str, str]]:
        sql = (
            "SELECT e.text_hash, MIN(e.text) FROM edges e "
            f"WHERE e.status IN {_BELIEVED_SQL} AND NOT EXISTS ("
            "SELECT 1 FROM embeddings m WHERE m.model = ? AND m.text_hash = e.text_hash) "
            "GROUP BY e.text_hash ORDER BY MIN(e.seq)")
        params: list[object] = [model]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return list(self._conn.execute(sql, params))

    def delete_embeddings(self, text_hash: str) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM embeddings WHERE text_hash = ?", (text_hash,))

    def hash_in_use(self, text_hash: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM edges WHERE text_hash = ? LIMIT 1",
                                 (text_hash,)).fetchone()
        return row is not None

    # -- stats --------------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        def scalar(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])

        return {
            "nodes": scalar("SELECT COUNT(*) FROM nodes"),
            "facts": scalar("SELECT COUNT(*) FROM edges"),
            "active_facts": scalar("SELECT COUNT(*) FROM edges WHERE status = 'active'"),
            "disclosures": scalar("SELECT COUNT(*) FROM disclosures"),
            "embeddings": scalar("SELECT COUNT(*) FROM embeddings"),
            "believed_texts": scalar(
                f"SELECT COUNT(DISTINCT text_hash) FROM edges WHERE status IN {_BELIEVED_SQL}"),
        }


def _opt(value: datetime | None) -> str | None:
    return to_db(value) if value is not None else None


def _in(column: str, values: Sequence[str], params: list[object]) -> str:
    values = list(values)
    if not values:
        return "0"  # an empty IN-list matches nothing
    params.extend(values)
    return f"{column} IN ({', '.join('?' for _ in values)})"


def _valid_clause(when: datetime, params: list[object], prefix: str = "") -> str:
    stamp = to_db(when)
    params += [stamp, stamp]
    return f"{prefix}valid_from <= ? AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)"


def _fact(row: tuple) -> Fact:
    (fact_id, subject, predicate, value, obj, scope, valid_from, valid_to, recorded_at,
     asserted_by, status, ended_at, superseded_by, derived_from, source, text) = row
    return Fact(
        id=fact_id, subject=subject, predicate=predicate, value=value, object=obj,
        scope=scope, valid_from=from_db(valid_from),
        valid_to=from_db(valid_to) if valid_to else None, recorded_at=from_db(recorded_at),
        asserted_by=asserted_by, status=status,
        ended_at=from_db(ended_at) if ended_at else None, superseded_by=superseded_by,
        derived_from=derived_from, source=source, text=text)
