"""``tgmem``: a command line for inspecting and maintaining a memory database.

Global options go before the command: ``tgmem --json search "opening hours" --as priya``.
Configuration comes from ``TGMEM_*`` environment variables, optionally loaded from a file
with ``--env-file`` (variables already set in the environment win).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .config import Settings, load_env_file
from .embeddings import EmbeddingError
from .handles import canonical_handle
from .memory import Memory, NotFound, ScopeError
from .model import Fact, Viewer
from .timeutil import Clock, parse_duration, parse_time, to_text
from .vault import VaultError, read_vault, rebuild_from_vault


class _Context:
    def __init__(self, args: argparse.Namespace, settings: Settings, clock: Clock | None,
                 out: TextIO) -> None:
        self.args, self.settings, self.clock, self.out = args, settings, clock, out
        self._memory: Memory | None = None

    @property
    def memory(self) -> Memory:
        if self._memory is None:
            self._memory = self.settings.open_memory(clock=self.clock)
        return self._memory

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()

    def time(self, text: str | None) -> datetime | None:
        if text is None:
            return None
        return parse_time(text, tz=self.settings.tz(), now=self.memory.now())

    def viewer(self, who: str | None) -> Viewer:
        """``--as`` takes a sender handle (phone/email) or a contact id."""
        if who is None:
            return Viewer.owner()
        if canonical_handle(who, default_country_code=self.settings.default_country_code):
            return self.memory.viewer_for(who)
        node = self.memory.store.get_node(who)
        if node is not None and node.role == "owner":
            return Viewer.owner(node.id)
        if node is not None and node.role == "contact":
            return Viewer.contact(node.id)
        return Viewer.anonymous()

    def emit(self, data: Any, text: str | Callable[[], str]) -> None:
        if self.args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str), file=self.out)
        else:
            print(text() if callable(text) else text, file=self.out)


# -- formatting ----------------------------------------------------------------------------

def _fact_json(fact: Fact) -> dict[str, Any]:
    def stamp(value: datetime | None) -> str | None:
        return to_text(value) if value is not None else None

    return {"id": fact.id, "subject": fact.subject, "predicate": fact.predicate,
            "value": fact.value, "object": fact.object, "scope": fact.scope,
            "valid_from": stamp(fact.valid_from), "valid_to": stamp(fact.valid_to),
            "recorded_at": stamp(fact.recorded_at), "status": fact.status,
            "ended_at": stamp(fact.ended_at), "superseded_by": fact.superseded_by,
            "derived_from": fact.derived_from, "asserted_by": fact.asserted_by,
            "source": fact.source}


def _fact_line(fact: Fact) -> str:
    value = f"[[{fact.object}]]" if fact.object is not None else (fact.value or "").replace(
        "\n", "\\n")
    until = to_text(fact.valid_to) if fact.valid_to else "open"
    return (f"{fact.id}  {fact.subject}.{fact.predicate} = {value}  [{fact.scope}]  "
            f"{to_text(fact.valid_from)} -> {until}  {fact.status}")


def _lines(facts: Sequence[Fact]) -> str:
    return "\n".join(_fact_line(f) for f in facts) if facts else "(no facts)"


# -- commands ------------------------------------------------------------------------------

def cmd_init(ctx: _Context) -> int:
    stats = ctx.memory.stats()  # opening the database creates the schema
    ctx.settings.vault_dir.mkdir(parents=True, exist_ok=True)
    print(f"initialised {ctx.settings.db_path} ({stats['facts']} facts, keyword index: "
          f"{stats['keyword_index']}, embedder: {stats['embedder'] or 'none'})", file=ctx.out)
    return 0


def cmd_add_entity(ctx: _Context) -> int:
    a = ctx.args
    if a.id:
        node = ctx.memory.upsert_entity(a.id, a.name, kind=a.kind, aliases=a.alias)
    else:
        node = ctx.memory.add_entity(a.name, kind=a.kind, aliases=a.alias)
    print(node.id, file=ctx.out)
    return 0


def cmd_add_contact(ctx: _Context) -> int:
    a = ctx.args
    node = ctx.memory.add_contact(a.name, a.handles, role="owner" if a.owner else "contact",
                                  id=a.id, aliases=a.alias)
    print(node.id, file=ctx.out)
    return 0


def cmd_define(ctx: _Context) -> int:
    ctx.memory.define_predicate(ctx.args.predicate, ctx.args.cardinality)
    print(f"predicate {ctx.args.predicate}: {ctx.args.cardinality}", file=ctx.out)
    return 0


def cmd_remember(ctx: _Context) -> int:
    a = ctx.args
    fact = ctx.memory.remember(
        a.subject, a.predicate, None if a.object else a.value,
        object=a.value if a.object else None, scope=a.scope,
        valid_from=ctx.time(a.valid_from), valid_to=ctx.time(a.valid_to),
        ttl=parse_duration(a.ttl) if a.ttl else None, asserted_by=a.by, source=a.source)
    ctx.emit(_fact_json(fact), _fact_line(fact))
    return 0


def cmd_correct(ctx: _Context) -> int:
    a = ctx.args
    fact = ctx.memory.correct(a.fact_id, None if a.object else a.value,
                              object=a.value if a.object else None,
                              valid_from=ctx.time(a.valid_from))
    ctx.emit(_fact_json(fact), _fact_line(fact))
    return 0


def cmd_retract(ctx: _Context) -> int:
    fact = ctx.memory.retract(ctx.args.fact_id)
    ctx.emit(_fact_json(fact), _fact_line(fact))
    return 0


def cmd_purge(ctx: _Context) -> int:
    if not ctx.args.yes:
        raise ValueError("purge permanently erases history; repeat with --yes to confirm")
    purged = ctx.memory.purge(ctx.args.fact_id)
    ctx.emit(purged, "purged " + ", ".join(purged))
    return 0


def cmd_facts(ctx: _Context) -> int:
    a = ctx.args
    facts = ctx.memory.facts(a.subject, a.predicate, viewer=ctx.viewer(a.sender),
                             as_of=ctx.time(a.as_of), known_at=ctx.time(a.known_at))
    ctx.emit([_fact_json(f) for f in facts], lambda: _lines(facts))
    return 0


def cmd_history(ctx: _Context) -> int:
    facts = ctx.memory.history(ctx.args.subject, ctx.args.predicate)
    ctx.emit([_fact_json(f) for f in facts], lambda: _lines(facts))
    return 0


def cmd_timeline(ctx: _Context) -> int:
    facts = ctx.memory.timeline(ctx.args.subject, ctx.args.predicate)
    ctx.emit([_fact_json(f) for f in facts], lambda: _lines(facts))
    return 0


def cmd_search(ctx: _Context) -> int:
    a = ctx.args
    result = ctx.memory.search(a.query, viewer=ctx.viewer(a.sender), as_of=ctx.time(a.as_of),
                               k=a.k)
    data = {"mode": result.mode, "degraded": result.degraded,
            "hits": [{**_fact_json(h.fact), "score": h.score, "signals": h.signals}
                     for h in result.hits]}

    def text() -> str:
        head = f"mode: {result.mode}" + (f" ({result.degraded})" if result.degraded else "")
        rows = [f"{h.score:.4f}  {_fact_line(h.fact)}  "
                f"via {', '.join(f'{k}#{v}' for k, v in h.signals.items())}"
                for h in result.hits]
        return "\n".join([head, *rows] if rows else [head, "(no matches)"])

    ctx.emit(data, text)
    return 0


def cmd_tell(ctx: _Context) -> int:
    told = ctx.memory.record_disclosure(ctx.args.contact, ctx.args.fact_ids,
                                        channel=ctx.args.channel)
    print(f"recorded {len(told)} disclosure(s) for {ctx.args.contact}", file=ctx.out)
    return 0


def cmd_told(ctx: _Context) -> int:
    told = ctx.memory.told(ctx.args.contact)
    data = [{"id": d.id, "fact_id": d.fact_id, "told_at": to_text(d.told_at),
             "channel": d.channel} for d in told]
    ctx.emit(data, lambda: "\n".join(f"{d.id}  {d.fact_id}  {to_text(d.told_at)}"
                                     f"{'  via ' + d.channel if d.channel else ''}"
                                     for d in told) or "(nothing told)")
    return 0


def cmd_corrections(ctx: _Context) -> int:
    owed = ctx.memory.corrections_owed(ctx.args.contact)
    data = [{"told": _fact_json(c.told), "told_at": to_text(c.told_at),
             "current": _fact_json(c.current) if c.current else None, "reason": c.reason}
            for c in owed]

    def text() -> str:
        if not owed:
            return "nothing owed"
        return "\n".join(f"{c.reason}: told {c.told.id} at {to_text(c.told_at)}; now "
                         f"{c.current.id if c.current else 'nothing'}" for c in owed)

    ctx.emit(data, text)
    return 0


def cmd_context(ctx: _Context) -> int:
    a = ctx.args
    scoped = ctx.memory.for_viewer(ctx.viewer(a.sender))
    pack = scoped.context(a.message, k=a.k,
                          repeat_after=parse_duration(a.repeat_after) if a.repeat_after else None)
    print(pack.text, file=ctx.out)
    return 0


def cmd_guard(ctx: _Context) -> int:
    leaked = ctx.memory.leaks(ctx.viewer(ctx.args.sender), ctx.args.reply)
    ctx.emit([_fact_json(f) for f in leaked],
             lambda: "\n".join(f"LEAK {_fact_line(f)}" for f in leaked) or "ok")
    return 1 if leaked else 0


def cmd_consolidate(ctx: _Context) -> int:
    report = ctx.memory.consolidate(dry_run=ctx.args.dry_run)
    if ctx.args.export and not ctx.args.dry_run:
        ctx.memory.export_vault(ctx.settings.vault_dir)
    data = {"ran_at": to_text(report.ran_at), "expired": report.expired,
            "duplicates": report.duplicates, "merged": report.merged,
            "embedded": report.embedded, "pending_embeddings": report.pending_embeddings,
            "dry_run": report.dry_run, "embed_error": report.embed_error}
    ctx.emit(data, report.summary())
    return 0


def cmd_reindex(ctx: _Context) -> int:
    report = ctx.memory.reindex()
    ctx.emit(report.__dict__, f"embedded {report.embedded}, pending {report.pending}"
             + (f" ({report.error})" if report.error else ""))
    return 0


def cmd_stats(ctx: _Context) -> int:
    stats = ctx.memory.stats()
    ctx.emit(stats, lambda: "\n".join(f"{k}: {v}" for k, v in stats.items()))
    return 0


def cmd_vault(ctx: _Context) -> int:
    vault_dir = ctx.settings.vault_dir
    if ctx.args.action == "export":
        report = ctx.memory.export_vault(vault_dir)
        print(f"wrote {len(report.written)} page(s), {report.unchanged} unchanged, "
              f"removed {len(report.removed)} -> {vault_dir}", file=ctx.out)
    elif ctx.args.action == "check":
        data = read_vault(vault_dir)
        print(f"ok: {len(data.nodes)} nodes, {len(data.facts)} facts, "
              f"{len(data.disclosures)} disclosures", file=ctx.out)
    else:
        report = rebuild_from_vault(vault_dir, ctx.settings.db_path, force=ctx.args.force)
        print(f"rebuilt {ctx.settings.db_path}: {report.nodes} nodes, {report.facts} facts, "
              f"{report.disclosures} disclosures, {report.embeddings_kept} cached "
              "embeddings kept", file=ctx.out)
    return 0


# -- parser --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgmem", description="Temporal graph memory for AI assistants.")
    parser.add_argument("--env-file", help="load TGMEM_* settings from this file")
    parser.add_argument("--db", help="database path (default: $TGMEM_DB or memory.db)")
    parser.add_argument("--vault", help="vault directory (default: $TGMEM_VAULT or vault)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, handler: Callable[[_Context], int], help_text: str):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(handler=handler)
        return p

    command("init", cmd_init, "create the database and vault directory")
    p = command("add-entity", cmd_add_entity, "add a node (a business, document, topic...)")
    p.add_argument("name")
    p.add_argument("--id", help="explicit id; updates the node if it exists")
    p.add_argument("--kind", default="thing")
    p.add_argument("--alias", action="append", default=[])
    p = command("add-contact", cmd_add_contact, "add someone the assistant talks to")
    p.add_argument("name")
    p.add_argument("handles", nargs="+", help="phone numbers (E.164) or email addresses")
    p.add_argument("--id")
    p.add_argument("--owner", action="store_true", help="this person owns the assistant")
    p.add_argument("--alias", action="append", default=[])
    p = command("define", cmd_define, "set whether a predicate holds one value or many")
    p.add_argument("predicate")
    p.add_argument("cardinality", choices=["one", "many"])
    p = command("remember", cmd_remember, "record a fact (supersedes overlapping values)")
    p.add_argument("subject")
    p.add_argument("predicate")
    p.add_argument("value")
    p.add_argument("--object", action="store_true", help="VALUE is a node id, not text")
    p.add_argument("--scope", help="public, internal or private:<contact-id>")
    p.add_argument("--from", dest="valid_from", help="valid from (date, time, or +3d)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--to", dest="valid_to", help="valid until (exclusive)")
    group.add_argument("--ttl", help="valid for a duration, e.g. 7d")
    p.add_argument("--by", default="owner", help="who asserted it (default: owner)")
    p.add_argument("--source", help="short provenance tag, e.g. sms or email")
    p = command("correct", cmd_correct, "replace a fact's value (from its start, or --from)")
    p.add_argument("fact_id")
    p.add_argument("value")
    p.add_argument("--object", action="store_true")
    p.add_argument("--from", dest="valid_from")
    p = command("retract", cmd_retract, "withdraw a fact (history keeps it)")
    p.add_argument("fact_id")
    p = command("purge", cmd_purge, "erase a fact and its derived rows permanently")
    p.add_argument("fact_id")
    p.add_argument("--yes", action="store_true")
    p = command("facts", cmd_facts, "facts valid at a point in time")
    p.add_argument("subject", nargs="?")
    p.add_argument("--predicate")
    p.add_argument("--as-of")
    p.add_argument("--known-at", help="as the memory believed at this time")
    p.add_argument("--as", dest="sender", help="view as this sender (handle or contact id)")
    for name, handler, text in (("history", cmd_history, "every row ever recorded"),
                                ("timeline", cmd_timeline, "currently believed values over time")):
        p = command(name, handler, text)
        p.add_argument("subject")
        p.add_argument("predicate", nargs="?")
    p = command("search", cmd_search, "hybrid search")
    p.add_argument("query")
    p.add_argument("--as", dest="sender")
    p.add_argument("--as-of")
    p.add_argument("-k", type=int, default=8)
    p = command("tell", cmd_tell, "record that a contact was told some facts")
    p.add_argument("contact")
    p.add_argument("fact_ids", nargs="+")
    p.add_argument("--channel")
    p = command("told", cmd_told, "what a contact has been told")
    p.add_argument("contact")
    p = command("corrections", cmd_corrections, "outdated things a contact was told")
    p.add_argument("contact")
    p = command("context", cmd_context, "print the prompt block for a message from a sender")
    p.add_argument("sender")
    p.add_argument("message")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--repeat-after", help="facts told longer ago than this may be repeated")
    p = command("guard", cmd_guard, "check a reply for other people's data (exit 1 on leak)")
    p.add_argument("sender")
    p.add_argument("reply")
    p = command("consolidate", cmd_consolidate, "nightly job: expire, dedupe, merge, embed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export", action="store_true", help="export the vault afterwards")
    command("reindex", cmd_reindex, "embed facts that have no vector yet")
    command("stats", cmd_stats, "counts and configuration")
    p = command("vault", cmd_vault, "export, check or rebuild from the Markdown vault")
    p.add_argument("action", choices=["export", "check", "rebuild"])
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the database has facts the vault lacks")
    return parser


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         clock: Clock | None = None, out: TextIO | None = None,
         err: TextIO | None = None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    environment = dict(env if env is not None else os.environ)
    try:
        if args.env_file:
            environment = {**load_env_file(args.env_file), **environment}
        settings = Settings.from_env(environment)
        if args.db:
            settings = replace(settings, db_path=Path(args.db))
        if args.vault:
            settings = replace(settings, vault_dir=Path(args.vault))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=err)
        return 1
    ctx = _Context(args, settings, clock, out)
    try:
        return args.handler(ctx)
    except (NotFound, ScopeError, VaultError, EmbeddingError, ValueError) as exc:
        print(f"error: {exc}", file=err)
    finally:
        ctx.close()
    return 1
