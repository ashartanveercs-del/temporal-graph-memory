"""The Markdown vault: a readable, diffable mirror that the graph is rebuilt from.

Layout::

    vault/
      index.md               format version, predicate definitions, list of pages
      entities/<node-id>.md  one page per node

A page is JSON-valued front matter (valid YAML, so Obsidian shows it as properties)
followed by one line per fact::

    - opening_hours: 9am-6pm  {public; valid 2026-03-03T09:02:00Z -> open; ...}  ^f-000002

The trailing ``^f-...`` is an Obsidian block id, so a "Told" line on a contact's page can
link to the exact fact it refers to: ``[[shop#^f-000002|opening_hours]]``.

Rendering is a pure function of the stored rows: fixed ordering, fixed field order,
canonical timestamps, ``\\n`` line endings on every OS. Exporting twice writes nothing the
second time, and export -> rebuild -> export reproduces the vault byte for byte. Parsing
is strict and reports ``file:line`` for anything it does not understand, because a vault
that silently loses a fact is worse than one that refuses to load.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .model import (
    ACTIVE,
    DEFAULT_CARDINALITY,
    ENDED,
    MANY,
    ONE,
    Disclosure,
    Fact,
    Node,
    Viewer,
    normalize_value,
    render_fact_text,
    scope_audience,
    validate_scope,
)
from .store import GraphStore, SQLiteGraphStore
from .timeutil import UTC, to_text

FORMAT_VERSION = 1

_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?Z$")
_FACT_ID = r"f-[0-9a-z]+"
_FACT_LINE = re.compile(
    r"^- (?P<predicate>[a-z][a-z0-9_]*): (?P<value>.*)  \{(?P<meta>[^{}]*)\}  "
    rf"\^(?P<id>{_FACT_ID})$")
_TOLD_LINE = re.compile(
    rf"^- \[\[(?P<page>[a-z0-9-]+)#\^(?P<fact>{_FACT_ID})\|(?P<label>[a-z][a-z0-9_]*)\]\]  "
    r"\{(?P<meta>[^{}]*)\}  \^(?P<id>d-[0-9a-z]+)$")
_OBJECT = re.compile(r"^\[\[(?P<node>[a-z0-9]+(?:-[a-z0-9]+)*)\]\]$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,63}$")
_SECTIONS = {"## Facts": "facts", "## History": "history", "## Told": "told"}
_ESCAPES = {"\\": "\\", "n": "\n", "r": "\r", "t": "\t", "[": "["}


class VaultError(ValueError):
    """The vault cannot be read. The message starts with ``file:line:`` when known."""

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None):
        where = f"{path}:{line}: " if path and line else f"{path}: " if path else ""
        super().__init__(where + message)


@dataclass(frozen=True)
class ExportReport:
    written: list[str]  # pages created or changed (vault-relative, "/" separated)
    unchanged: int
    removed: list[str]  # stale pages of nodes that no longer exist


@dataclass(frozen=True)
class VaultData:
    nodes: list[Node]
    facts: list[Fact]
    disclosures: list[Disclosure]
    predicates: dict[str, str]


@dataclass(frozen=True)
class RebuildReport:
    nodes: int
    facts: int
    disclosures: int
    embeddings_kept: int  # cached vectors carried over from the previous database


# -- export --------------------------------------------------------------------------------

def export_vault(store: GraphStore, directory: str | Path) -> ExportReport:
    root = Path(directory)
    (root / "entities").mkdir(parents=True, exist_ok=True)
    nodes = store.nodes()
    facts_by_subject: dict[str, list[Fact]] = defaultdict(list)
    facts_by_id: dict[str, Fact] = {}
    for fact in store.query_facts():
        facts_by_subject[fact.subject].append(fact)
        facts_by_id[fact.id] = fact
    pages = {"index.md": render_index(nodes, store.predicates())}
    for node in nodes:
        told = [(d, facts_by_id[d.fact_id]) for d in store.disclosures(contact=node.id)]
        pages[f"entities/{node.id}.md"] = render_page(node, facts_by_subject[node.id], told)

    written, unchanged = [], 0
    for relative, content in pages.items():
        path = root / relative
        data = content.encode("utf-8")
        if path.exists() and path.read_bytes() == data:
            unchanged += 1
            continue
        _atomic_write(path, data)
        written.append(relative)
    removed = []
    for page in sorted((root / "entities").glob("*.md")):
        relative = f"entities/{page.name}"
        if relative not in pages and _managed_page(page):
            page.unlink()
            removed.append(relative)
    return ExportReport(written, unchanged, removed)


def render_index(nodes: Iterable[Node], predicates: dict[str, str]) -> str:
    lines = [
        "---",
        f"tgmem-vault: {FORMAT_VERSION}",
        f"predicates: {json.dumps(predicates, sort_keys=True)}",
        "---",
        "",
        "# Memory vault",
        "",
        "A readable mirror of a temporal-graph-memory database, and the source it is rebuilt",
        "from (`tgmem vault rebuild`). Each page in `entities/` is one node. Each line under",
        "**Facts** or **History** is one fact; **Told** lists what a contact has been told.",
        "Pages are regenerated on every export: change memory through the API or CLI, or",
        "edit fact lines here and rebuild.",
        "",
        "## Entities",
        "",
    ]
    lines += [f"- [[{node.id}]] {node.name} ({node.kind})" for node in nodes]
    return "\n".join(lines) + "\n"


def render_page(node: Node, facts: list[Fact], told: list[tuple[Disclosure, Fact]]) -> str:
    lines = [
        "---",
        f"id: {_json(node.id)}",
        f"name: {_json(node.name)}",
        f"kind: {_json(node.kind)}",
        f"role: {_json(node.role)}",
        f"handles: {_json(list(node.handles))}",
        f"aliases: {_json(list(node.aliases))}",
        f"created: {_json(to_text(node.created_at))}",
        "---",
        "",
        f"# {node.name}",
    ]
    for title, group in (("## Facts", [f for f in facts if f.status == ACTIVE]),
                         ("## History", [f for f in facts if f.status != ACTIVE])):
        if group:
            lines += ["", title, ""] + [render_fact(f) for f in group]
    if told:
        lines += ["", "## Told", ""] + [render_told(d, f) for d, f in told]
    return "\n".join(lines) + "\n"


def render_fact(fact: Fact) -> str:
    value = f"[[{fact.object}]]" if fact.object is not None else escape_value(fact.value or "")
    until = to_text(fact.valid_to) if fact.valid_to is not None else "open"
    meta = [fact.scope, f"valid {to_text(fact.valid_from)} -> {until}",
            f"recorded {to_text(fact.recorded_at)}", f"by {fact.asserted_by}"]
    if fact.source:
        meta.append(f"source {fact.source}")
    if fact.status != ACTIVE and fact.ended_at is not None:
        end = f"{fact.status} {to_text(fact.ended_at)}"
        meta.append(end + (f" by {fact.superseded_by}" if fact.superseded_by else ""))
    if fact.derived_from:
        meta.append(f"from {fact.derived_from}")
    return f"- {fact.predicate}: {value}  {{{'; '.join(meta)}}}  ^{fact.id}"


def render_told(disclosure: Disclosure, fact: Fact) -> str:
    meta = [f"at {to_text(disclosure.told_at)}"]
    if disclosure.channel:
        meta.append(f"via {disclosure.channel}")
    return (f"- [[{fact.subject}#^{fact.id}|{fact.predicate}]]  "
            f"{{{'; '.join(meta)}}}  ^{disclosure.id}")


def escape_value(value: str) -> str:
    """Backslash-escape so a value always fits on one line and never reads as a link."""
    out = (value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
           .replace("\t", "\\t"))
    return "\\" + out if out.startswith("[[") else out


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes (never text mode: Windows would turn \\n into \\r\\n) via a temporary
    file and an atomic rename, so a crash never leaves half a page behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _managed_page(path: Path) -> bool:
    try:
        front, _ = _front_matter(path.read_text(encoding="utf-8").split("\n"), path.name)
    except (VaultError, UnicodeDecodeError):
        return False
    return front.get("id") == path.stem


# -- parsing -------------------------------------------------------------------------------

def read_vault(directory: str | Path) -> VaultData:
    """Parse and validate a whole vault. Raises ``VaultError`` on the first problem."""
    root = Path(directory)
    index_path = root / "index.md"
    if not index_path.exists():
        raise VaultError("index.md not found; is this a vault directory?", path=str(root))
    front, _ = _front_matter(_lines(index_path), "index.md")
    if front.get("tgmem-vault") != FORMAT_VERSION:
        raise VaultError(f"unsupported vault version {front.get('tgmem-vault')!r}; "
                         f"this build reads version {FORMAT_VERSION}", path="index.md")
    predicates = front.get("predicates", {})
    if not isinstance(predicates, dict) or any(
            not isinstance(k, str) or v not in (ONE, MANY) for k, v in predicates.items()):
        raise VaultError("predicates must map names to \"one\" or \"many\"", path="index.md")

    nodes: list[Node] = []
    facts: list[tuple[Fact, str, int]] = []
    told: list[tuple[Disclosure, str, str, str, int]] = []
    for page in sorted((root / "entities").glob("*.md")):
        relative = f"entities/{page.name}"
        lines = _lines(page)
        if not lines or lines[0] != "---":
            continue  # not a managed page (personal notes are left alone)
        node, page_facts, page_told = _parse_page(lines, relative, page.stem)
        nodes.append(node)
        facts += page_facts
        told += page_told
    _validate(nodes, facts, told, predicates)
    return VaultData(nodes, [f for f, _, _ in facts], [d for d, *_ in told], dict(predicates))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _front_matter(lines: list[str], path: str) -> tuple[dict, int]:
    if not lines or lines[0] != "---":
        raise VaultError("missing front matter", path=path, line=1)
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise VaultError("front matter is not closed with ---", path=path, line=1) from None
    data = {}
    for number, raw in enumerate(lines[1:end], start=2):
        key, sep, value = raw.partition(": ")
        if not sep:
            raise VaultError("front matter lines look like 'key: value'", path=path,
                             line=number)
        try:
            data[key] = json.loads(value)
        except ValueError:
            raise VaultError(f"value of {key!r} is not valid JSON", path=path,
                             line=number) from None
    return data, end


def _parse_page(lines: list[str], path: str,
                stem: str) -> tuple[Node, list, list]:
    front, end = _front_matter(lines, path)
    if front.get("id") != stem:
        raise VaultError(f"front matter id {front.get('id')!r} does not match the file name",
                         path=path, line=2)
    node = _node(front, path)
    facts, told, section = [], [], None
    for number, line in enumerate(lines[end + 1:], start=end + 2):
        if line.startswith("## "):
            section = _SECTIONS.get(line.strip())
            continue
        if section is None or not line.strip():
            continue
        if section == "told":
            told.append(_parse_told(line, node.id, path, number))
            continue
        fact = _parse_fact(line, node.id, path, number)
        if (section == "facts") != (fact.status == ACTIVE):
            where = "Facts" if section == "facts" else "History"
            raise VaultError(f"a {fact.status} fact cannot be listed under ## {where}",
                             path=path, line=number)
        facts.append((fact, path, number))
    return node, facts, told


def _node(front: dict, path: str) -> Node:
    try:
        name, kind, role = front["name"], front["kind"], front.get("role")
        handles, aliases = front.get("handles", []), front.get("aliases", [])
        created = _time(front["created"], path, None)
    except KeyError as missing:
        raise VaultError(f"front matter lacks {missing}", path=path) from None
    ok = (isinstance(name, str) and name and isinstance(kind, str)
          and role in (None, "owner", "contact")
          and all(isinstance(h, str) for h in handles) and all(isinstance(a, str) for a in aliases))
    if not ok:
        raise VaultError("front matter has a field of the wrong type", path=path)
    return Node(id=front["id"], name=name, kind=kind, created_at=created,
                aliases=tuple(aliases), handles=tuple(handles), role=role)


def _parse_fact(line: str, subject: str, path: str, number: int) -> Fact:
    match = _FACT_LINE.match(line)
    if match is None:
        raise VaultError("not a valid fact line", path=path, line=number)
    raw = match["value"]
    obj_match = _OBJECT.match(raw)
    if obj_match:
        value, obj = None, obj_match["node"]
    else:
        try:
            value, obj = normalize_value(_unescape(raw, path, number)), None
        except ValueError as exc:
            raise VaultError(str(exc), path=path, line=number) from None
    meta = _fact_meta(match["meta"], path, number)
    try:
        return Fact(id=match["id"], subject=subject, predicate=match["predicate"], value=value,
                    object=obj, text="", **meta)
    except ValueError as exc:
        raise VaultError(str(exc), path=path, line=number) from None


def _fact_meta(meta: str, path: str, number: int) -> dict:
    parts = meta.split("; ")
    try:
        out: dict = {"scope": validate_scope(parts[0])}
    except ValueError as exc:
        raise VaultError(str(exc), path=path, line=number) from None
    for part in parts[1:]:
        word, _, rest = part.partition(" ")
        if word == "valid":
            start, sep, until = rest.partition(" -> ")
            if not sep:
                raise VaultError("valid needs 'FROM -> TO'", path=path, line=number)
            out["valid_from"] = _time(start, path, number)
            out["valid_to"] = None if until == "open" else _time(until, path, number)
        elif word == "recorded":
            out["recorded_at"] = _time(rest, path, number)
        elif word in ("by", "source"):
            if not _TAG.match(rest):
                raise VaultError(f"invalid {word} {rest!r}", path=path, line=number)
            out["asserted_by" if word == "by" else "source"] = rest
        elif word == "from":
            out["derived_from"] = _fact_ref(rest, path, number)
        elif word in ENDED:
            stamp, _, successor = rest.partition(" by ")
            out["status"], out["ended_at"] = word, _time(stamp, path, number)
            out["superseded_by"] = _fact_ref(successor, path, number) if successor else None
        else:
            raise VaultError(f"unknown field {word!r} in fact metadata", path=path, line=number)
    for required in ("valid_from", "recorded_at", "asserted_by"):
        if required not in out:
            raise VaultError(f"fact metadata lacks {required}", path=path, line=number)
    return out


def _parse_told(line: str, contact: str, path: str,
                number: int) -> tuple[Disclosure, str, str, str, int]:
    match = _TOLD_LINE.match(line)
    if match is None:
        raise VaultError("not a valid told line", path=path, line=number)
    fields: dict = {}
    for part in match["meta"].split("; "):
        word, _, rest = part.partition(" ")
        if word == "at":
            fields["told_at"] = _time(rest, path, number)
        elif word == "via" and _TAG.match(rest):
            fields["channel"] = rest
        else:
            raise VaultError(f"invalid told line: unknown field {word!r}", path=path,
                             line=number)
    if "told_at" not in fields:
        raise VaultError("invalid told line: missing 'at'", path=path, line=number)
    disclosure = Disclosure(match["id"], contact, match["fact"], fields["told_at"],
                            fields.get("channel"))
    return disclosure, match["page"], match["label"], path, number


def _fact_ref(text: str, path: str, number: int) -> str:
    if not re.fullmatch(_FACT_ID, text):
        raise VaultError(f"invalid fact id {text!r}", path=path, line=number)
    return text


def _time(text: object, path: str, number: int | None) -> datetime:
    if not isinstance(text, str) or not _TIME.match(text):
        raise VaultError(f"invalid timestamp {text!r}; expected e.g. 2026-03-02T09:00:00Z",
                         path=path, line=number)
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in text else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.strptime(text, fmt).replace(tzinfo=UTC)


def _unescape(text: str, path: str, number: int) -> str:
    out, i = [], 0
    while i < len(text):
        char = text[i]
        if char != "\\":
            out.append(char)
            i += 1
            continue
        mapped = _ESCAPES.get(text[i + 1]) if i + 1 < len(text) else None
        if mapped is None:
            raise VaultError("invalid backslash escape in value", path=path, line=number)
        out.append(mapped)
        i += 2
    return "".join(out)


def _validate(nodes: list[Node], facts: list[tuple[Fact, str, int]],
              told: list[tuple[Disclosure, str, str, str, int]],
              predicates: dict[str, str]) -> None:
    by_node = {n.id: n for n in nodes}
    handle_owner: dict[str, str] = {}
    for node in nodes:
        for handle in node.handles:
            if handle in handle_owner:
                raise VaultError(f"handle {handle!r} appears on two pages",
                                 path=f"entities/{node.id}.md")
            handle_owner[handle] = node.id

    by_fact: dict[str, Fact] = {}
    for fact, path, number in facts:
        if fact.id in by_fact:
            raise VaultError(f"duplicate fact id {fact.id!r}", path=path, line=number)
        by_fact[fact.id] = fact
    for fact, path, number in facts:
        if fact.object is not None and fact.object not in by_node:
            raise VaultError(f"unknown node {fact.object!r}", path=path, line=number)
        audience = scope_audience(fact.scope)
        if audience is not None:
            if audience not in by_node:
                raise VaultError(f"unknown node {audience!r}", path=path, line=number)
            if by_node[audience].role not in ("owner", "contact"):
                raise VaultError(f"{audience!r} is not a contact", path=path, line=number)
        for ref in (fact.superseded_by, fact.derived_from):
            if ref is not None and ref not in by_fact:
                raise VaultError(f"unknown fact {ref!r}", path=path, line=number)

    def cardinality(predicate: str) -> str:
        return predicates.get(predicate) or DEFAULT_CARDINALITY.get(predicate, ONE)

    timelines: dict[tuple[str, str, str], list[tuple[Fact, str, int]]] = defaultdict(list)
    for fact, path, number in facts:
        if fact.believed and cardinality(fact.predicate) == ONE:
            timelines[fact.key].append((fact, path, number))
    for entries in timelines.values():
        entries.sort(key=lambda e: (e[0].valid_from, e[0].id))
        for (earlier, _, _), (later, path, number) in zip(entries, entries[1:], strict=False):
            if earlier.valid_to is None or earlier.valid_to > later.valid_from:
                raise VaultError(f"{earlier.id} and {later.id} overlap in valid time for "
                                 f"single-valued {later.predicate!r}", path=path, line=number)

    seen: set[str] = set()
    for disclosure, page, label, path, number in told:
        if disclosure.id in seen:
            raise VaultError(f"duplicate disclosure id {disclosure.id!r}", path=path,
                             line=number)
        seen.add(disclosure.id)
        if by_node[disclosure.contact].role != "contact":
            raise VaultError("only contacts have a Told section", path=path, line=number)
        fact = by_fact.get(disclosure.fact_id)
        if fact is None:
            raise VaultError(f"unknown fact {disclosure.fact_id!r}", path=path, line=number)
        if (page, label) != (fact.subject, fact.predicate):
            raise VaultError(f"link should read [[{fact.subject}#^{fact.id}|{fact.predicate}]]",
                             path=path, line=number)
        if not Viewer.contact(disclosure.contact).can_see(fact.scope):
            raise VaultError(f"{fact.id} is not visible to {disclosure.contact!r}", path=path,
                             line=number)


# -- rebuild -------------------------------------------------------------------------------

def rebuild_from_vault(vault_dir: str | Path, db_path: str | Path, *, force: bool = False,
                       fts: bool | None = None) -> RebuildReport:
    """Replace the database at ``db_path`` with one built from the vault.

    The vault is parsed and validated before anything is touched. The new database is
    built next to the old one and swapped in with an atomic rename, together with its
    WAL sidecar files (a stale ``-wal`` file next to a new database would be replayed
    into it). Cached embeddings are carried over when the old file is readable, so a
    rebuild does not pay to embed unchanged text again. If the old database holds facts
    the vault does not, the rebuild refuses unless ``force=True``.
    """
    data = read_vault(vault_dir)
    target = Path(db_path)
    existing = _read_old(target, "SELECT id FROM edges")
    if existing is not None and not force:
        missing = {row[0] for row in existing} - {f.id for f in data.facts}
        if missing:
            raise VaultError(f"{len(missing)} fact(s) in the database are not in the vault "
                             f"(for example {sorted(missing)[0]}); export first, or rebuild "
                             "with force=True to discard them")
    staging = target.with_name(target.name + ".rebuild")
    _remove_database(staging)
    store = SQLiteGraphStore(staging, fts=fts)
    try:
        nodes = {n.id: n for n in data.nodes}
        with store.transaction():
            store.defer_foreign_keys()
            for name, cardinality in sorted(data.predicates.items()):
                store.put_predicate(name, cardinality)
            for node in sorted(data.nodes, key=lambda n: n.id):
                store.put_node(node)
            for fact in sorted(data.facts, key=lambda f: (f.recorded_at, f.id)):
                text = render_fact_text(nodes[fact.subject], fact.predicate, fact.value,
                                        nodes[fact.object] if fact.object else None)
                store.insert_fact(replace(fact, text=text))
            for disclosure in sorted(data.disclosures, key=lambda d: (d.told_at, d.id)):
                store.insert_disclosure(disclosure)
            stamps = [n.created_at for n in data.nodes] + [f.recorded_at for f in data.facts]
            stamps += [f.ended_at for f in data.facts if f.ended_at is not None]
            stamps += [d.told_at for d in data.disclosures]
            if stamps:
                store.restore_tx_time(max(stamps))
        kept = _carry_embeddings(target, store)
    finally:
        store.close()
    _remove_database(target)
    os.replace(staging, target)
    return RebuildReport(len(data.nodes), len(data.facts), len(data.disclosures), kept)


def _read_old(path: Path, sql: str) -> list[tuple] | None:
    """Rows from the previous database, or None if it is missing or unreadable."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(path)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return None


def _carry_embeddings(old: Path, store: SQLiteGraphStore) -> int:
    rows = _read_old(old, "SELECT model, text_hash, dim, vector FROM embeddings") or []
    by_model: dict[str, list[tuple[str, bytes, int]]] = defaultdict(list)
    for model, digest, dim, vector in rows:
        if store.hash_in_use(digest):
            by_model[model].append((digest, vector, dim))
    for model, items in sorted(by_model.items()):
        store.put_embeddings(model, items)
    return sum(len(items) for items in by_model.values())


def _remove_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
