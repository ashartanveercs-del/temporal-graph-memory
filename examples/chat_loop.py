"""A small offline assistant wired to temporal-graph-memory.

Each input line is ``<sender>: <message>``, the way an SMS or chat webhook delivers a
message after verifying who sent it. Senders are phone numbers, or the demo names
``rosa`` (the owner), ``priya`` and ``mateo`` (customers) of a fictional print shop.

Replies are composed from the sender's scoped context only, so the loop shows every
moving part without needing an LLM: corrections owed come first, facts already told are
not repeated, the outbound guard checks every reply, and what the reply stated is
recorded in the audience ledger. Swap ``compose_reply`` for a model call to make it real
(see ``llm_assistant.py``).

Commands:  /wait 3d   advance the demo clock
           /consolidate   run the nightly job
           /vault     export the Markdown vault
           /quit

Teaching:  ``rosa: remember shop opening_hours = 10am-6pm Monday to Saturday``
           ``priya: remember deadline = invitations needed by June 1``

Run:  python examples/chat_loop.py
      python examples/chat_loop.py --script examples/demo_script.txt --show-context
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from tgmem import Memory, ScopedMemory, ScopeError
from tgmem.audience import ContextPack, value_text
from tgmem.ids import SequentialIds
from tgmem.timeutil import UTC, ManualClock, parse_duration

DEMO_START = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
SENDERS = {"rosa": "+15550000100", "priya": "+15550000101", "mateo": "+15550000102"}
SAFE_FALLBACK = "Let me check on that and get back to you."


def build_demo(db_path: Path, clock: ManualClock) -> Memory:
    memory = Memory.open(db_path, clock=clock, ids=SequentialIds())
    memory.add_contact("Rosa Delgado", [SENDERS["rosa"]], role="owner", id="rosa")
    memory.add_contact("Priya Shah", [SENDERS["priya"]], id="priya")
    memory.add_contact("Mateo Ruiz", [SENDERS["mateo"]], id="mateo")
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    memory.remember("shop", "opening_hours", "9am-5pm Monday to Saturday")
    memory.remember("shop", "turnaround", "business cards ship in 3 days")
    memory.remember("shop", "invoice_terms", "net 30 by bank transfer")
    memory.remember("shop", "supplier_margin", "40 percent on paper stock", scope="internal")
    memory.remember("priya", "budget", "$480 for wedding invitations")
    memory.remember("mateo", "order", "500 flyers due Friday")
    return memory


def compose_reply(scoped: ScopedMemory, pack: ContextPack) -> str:
    """Deterministic stand-in for a model: say what is owed, then what is new."""
    memory = scoped.memory
    parts = []
    for correction in pack.corrections:
        was = value_text(memory, correction.told)
        if correction.current is None:
            parts.append(f'Quick correction: earlier I said "{was}", but that no longer '
                         "applies.")
        else:
            parts.append(f'Quick correction: earlier I said "{was}"; it is now '
                         f'"{value_text(memory, correction.current)}".')
    fresh = [cf.fact for cf in pack.facts if cf.told_at is None][:2]
    repeated = [cf.fact for cf in pack.facts if cf.told_at is not None][:1]
    if fresh:
        parts.append(" ".join(f"{fact.text}." for fact in fresh))
    elif repeated and not parts:
        parts.append(f"As I mentioned earlier: {repeated[0].text}.")
    return " ".join(parts) or "I don't have that on record; I'll check with the team."


class Assistant:
    def __init__(self, memory: Memory, out: TextIO, *, show_context: bool = False) -> None:
        self.memory, self.out, self.show_context = memory, out, show_context

    def handle(self, sender: str, message: str) -> str:
        scoped = self.memory.for_sender(sender)
        if message.lower().startswith("remember "):
            return self._teach(scoped, message[len("remember "):])
        pack = scoped.context(message)
        if self.show_context:
            for line in pack.text.splitlines():
                print(f"    | {line}", file=self.out)
        reply = compose_reply(scoped, pack)
        if scoped.guard(reply):  # defence in depth: never send someone else's data
            reply = SAFE_FALLBACK
        scoped.record_reply(reply, pack, channel="sms")
        return reply

    def _teach(self, scoped: ScopedMemory, text: str) -> str:
        head, sep, value = text.partition("=")
        words = head.split()
        if not sep or not words:
            return 'Try "remember <predicate> = <value>".'
        try:
            if scoped.viewer.kind == "owner" and len(words) == 2:
                fact = scoped.remember(words[1], value.strip(), subject=words[0])
            elif scoped.viewer.kind == "owner":
                return 'Owners say "remember <subject> <predicate> = <value>".'
            else:
                fact = scoped.remember("_".join(words), value.strip())
        except ScopeError:
            return "I can only save details for people I know."
        return f"Saved: {fact.text} ({fact.scope})."


def run(lines: Iterable[str], memory: Memory, clock: ManualClock, vault_dir: Path,
        out: TextIO, *, show_context: bool = False) -> None:
    assistant = Assistant(memory, out, show_context=show_context)
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "/quit":
            break
        if line.startswith("/wait "):
            clock.advance(parse_duration(line.split(maxsplit=1)[1]))
            print(f"-- clock is now {clock.now():%Y-%m-%d %H:%M} UTC", file=out)
        elif line == "/consolidate":
            print(f"-- consolidate: {memory.consolidate().summary()}", file=out)
        elif line == "/vault":
            report = memory.export_vault(vault_dir)
            print(f"-- vault: wrote {len(report.written)} page(s) to {vault_dir}", file=out)
        else:
            sender, sep, message = line.partition(":")
            if not sep:
                print('-- lines look like "priya: hello"', file=out)
                continue
            name = sender.strip()
            print(f"{name} > {message.strip()}", file=out)
            reply = assistant.handle(SENDERS.get(name.lower(), name), message.strip())
            print(f"assistant > {reply}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", help="read lines from this file instead of stdin")
    parser.add_argument("--db", help="database path (default: a temporary directory)")
    parser.add_argument("--vault", help="vault directory (default: next to the database)")
    parser.add_argument("--show-context", action="store_true",
                        help="print the prompt block each reply was built from")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(args.db) if args.db else Path(tmp) / "demo.db"
        vault = Path(args.vault) if args.vault else db.parent / "vault"
        clock = ManualClock(DEMO_START)
        memory = build_demo(db, clock)
        try:
            if args.script:
                lines = Path(args.script).read_text(encoding="utf-8").splitlines()
            else:
                print('Type "priya: what are your hours?" (or /wait 2d, /vault, /quit).')
                lines = sys.stdin
            run(lines, memory, clock, vault, sys.stdout, show_context=args.show_context)
        finally:
            memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
