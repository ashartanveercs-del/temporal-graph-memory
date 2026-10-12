"""The Markdown vault: deterministic export, strict parsing, and rebuilding the graph."""

import re
from datetime import timedelta

import pytest
from conftest import seed
from fakes import PRIYA, T0, SwitchableEmbedder

from tgmem.embeddings import HashingEmbedder
from tgmem.ids import SequentialIds
from tgmem.memory import Memory
from tgmem.timeutil import ManualClock
from tgmem.vault import VaultError, read_vault, rebuild_from_vault

SHOP_PAGE = """\
---
id: "shop"
name: "Lakeside Print Shop"
kind: "org"
role: null
handles: []
aliases: []
created: "2026-03-02T09:00:00Z"
---

# Lakeside Print Shop

## Facts

- opening_hours: 9am-6pm  {public; valid 2026-03-03T09:02:00Z -> open; recorded 2026-03-03T09:02:00Z; by owner}  ^f-000002

## History

- opening_hours: 9am-5pm  {public; valid 2026-03-02T09:02:00Z -> open; recorded 2026-03-02T09:02:00Z; by owner; superseded 2026-03-03T09:02:00Z by f-000002}  ^f-000001
- opening_hours: 9am-5pm  {public; valid 2026-03-02T09:02:00Z -> 2026-03-03T09:02:00Z; recorded 2026-03-03T09:02:00Z; by owner; expired 2026-03-03T09:02:00Z; from f-000001}  ^f-000003
"""  # noqa: E501

PRIYA_PAGE = """\
---
id: "priya"
name: "Priya Shah"
kind: "person"
role: "contact"
handles: ["+15550000101"]
aliases: []
created: "2026-03-02T09:01:00Z"
---

# Priya Shah

## Told

- [[shop#^f-000002|opening_hours]]  {at 2026-03-03T09:03:00Z; via sms}  ^d-000001
"""


def test_pages_have_a_stable_human_readable_format(tmp_path):
    clock = ManualClock(T0)
    memory = Memory.open(tmp_path / "m.db", clock=clock, ids=SequentialIds())
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    clock.advance(minutes=1)
    memory.add_contact("Priya Shah", ["+1 555 000 0101"], id="priya")
    clock.advance(minutes=1)
    memory.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(days=1)
    memory.remember("shop", "opening_hours", "9am-6pm")
    clock.advance(minutes=1)
    memory.record_disclosure("priya", ["f-000002"], channel="sms")
    memory.export_vault(tmp_path / "vault")
    entities = tmp_path / "vault" / "entities"
    assert (entities / "shop.md").read_text(encoding="utf-8") == SHOP_PAGE
    assert (entities / "priya.md").read_text(encoding="utf-8") == PRIYA_PAGE
    index = (tmp_path / "vault" / "index.md").read_text(encoding="utf-8")
    assert index.startswith('---\ntgmem-vault: 1\npredicates: {}\n---\n')
    assert "- [[priya]] Priya Shah (person)" in index
    memory.close()


@pytest.fixture
def busy(mem, clock):
    """A memory exercising every feature the vault has to carry."""
    mem.define_predicate("likes", "many")
    mem.upsert_entity("acme", "Acme Logistics", kind="org", aliases=("Acme",))
    clock.advance(hours=1)
    hours = mem.remember("shop", "opening_hours", "9am-5pm")
    mem.remember("shop", "opening_hours", "closed for the holidays",
                 valid_from=clock.now() + timedelta(days=20),
                 valid_to=clock.now() + timedelta(days=23))
    mem.remember("priya", "works_at", object="acme")
    mem.remember("priya", "likes", "linen paper", source="sms")
    mem.remember("priya", "likes", "letterpress")
    mem.remember("mateo", "order", "500 flyers\ndue Friday {rush}")
    mem.for_sender(PRIYA).remember("deadline", "June 1")
    promo = mem.remember("shop", "promo", "[[not a link]] 20% off \\ today  {x}  ^f-1")
    clock.advance(hours=2)
    mem.retract(promo.id)
    turnaround = mem.remember("shop", "turnaround", "3 days")
    mem.correct(turnaround.id, "5 days")
    mem.remember("shop", "supplier_margin", "40% · paper; card", scope="internal")
    mem.record_disclosure("priya", [hours.id], channel="sms")
    mem.record_disclosure("mateo", [turnaround.id])
    return mem


def test_export_is_idempotent_and_uses_lf_only(busy, tmp_path):
    vault = tmp_path / "vault"
    first = busy.export_vault(vault)
    assert first.unchanged == 0 and len(first.written) == len(busy.nodes()) + 1
    second = busy.export_vault(vault)
    assert second.written == [] and second.unchanged == len(first.written)
    for page in vault.rglob("*.md"):
        assert b"\r\n" not in page.read_bytes()


def test_round_trip_reproduces_the_vault_byte_for_byte(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    before = {p.relative_to(vault): p.read_bytes() for p in vault.rglob("*.md")}
    rebuild_from_vault(vault, tmp_path / "rebuilt.db")
    rebuilt = Memory.open(tmp_path / "rebuilt.db")
    rebuilt.export_vault(tmp_path / "vault2")
    after = {p.relative_to(tmp_path / "vault2"): p.read_bytes()
             for p in (tmp_path / "vault2").rglob("*.md")}
    assert after == before
    rebuilt.close()


def test_the_rebuilt_graph_answers_queries_identically(busy, tmp_path):
    busy.export_vault(tmp_path / "vault")
    rebuild_from_vault(tmp_path / "vault", tmp_path / "rebuilt.db")
    rebuilt = Memory.open(tmp_path / "rebuilt.db", clock=busy.clock)
    probes = [busy.now() + timedelta(days=d) for d in (-1, 0, 21, 30)]
    for when in probes:
        assert rebuilt.facts(as_of=when) == busy.facts(as_of=when)
    for subject in ("shop", "priya", "mateo"):
        assert rebuilt.history(subject) == busy.history(subject)
    assert rebuilt.told("priya") == busy.told("priya")
    assert rebuilt.corrections_owed("mateo") == busy.corrections_owed("mateo")
    assert [h.fact.id for h in rebuilt.search("opening hours").hits] == [
        h.fact.id for h in busy.search("opening hours").hits]
    assert rebuilt.cardinality("likes") == "many"
    rebuilt.close()


def test_rebuild_restores_the_transaction_clock(busy, tmp_path):
    busy.export_vault(tmp_path / "vault")
    rebuild_from_vault(tmp_path / "vault", tmp_path / "rebuilt.db")
    rebuilt = Memory.open(tmp_path / "rebuilt.db", clock=ManualClock(T0))  # clock far behind
    fact = rebuilt.remember("shop", "fax", "none")
    newest = max(f.recorded_at for s in ("shop", "priya", "mateo") for f in busy.history(s))
    assert fact.recorded_at > newest
    rebuilt.close()


def test_disaster_recovery_from_a_corrupted_database(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    when = busy.now()
    expected = busy.facts(as_of=when)
    db = tmp_path / "memory.db"
    busy.close()
    for sidecar in (db.with_name("memory.db-wal"), db.with_name("memory.db-shm")):
        sidecar.unlink(missing_ok=True)
    db.write_bytes(b"this is not a database" * 100)
    report = rebuild_from_vault(vault, db)
    assert report.facts == sum(1 for p in vault.rglob("*.md")
                               for line in p.read_text(encoding="utf-8").splitlines()
                               if re.search(r"  \^f-[0-9a-z]+$", line))
    restored = Memory.open(db)
    assert restored.facts(as_of=when) == expected
    restored.close()


def test_rebuild_refuses_to_drop_facts_missing_from_a_stale_vault(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    busy.remember("shop", "fax", "none")  # written after the export
    path = busy.store.path
    busy.close()
    with pytest.raises(VaultError, match="not in the vault"):
        rebuild_from_vault(vault, path)
    rebuild_from_vault(vault, path, force=True)
    forced = Memory.open(path)
    assert forced.facts("shop", "fax") == []
    forced.close()


def test_rebuild_keeps_cached_embeddings(tmp_path):
    counting = SwitchableEmbedder(HashingEmbedder(64))
    memory = Memory.open(tmp_path / "m.db", clock=ManualClock(T0), ids=SequentialIds(),
                         embedder=counting)
    seed(memory)
    memory.remember("shop", "opening_hours", "9am-5pm")
    memory.remember("shop", "turnaround", "3 days")
    memory.export_vault(tmp_path / "vault")
    memory.close()
    report = rebuild_from_vault(tmp_path / "vault", tmp_path / "m.db", force=True)
    assert report.embeddings_kept == 2
    calls = counting.calls
    reopened = Memory.open(tmp_path / "m.db", embedder=counting)
    assert reopened.reindex().embedded == 0
    assert counting.calls == calls  # nothing had to be embedded again
    reopened.close()


def test_hand_edits_are_picked_up_by_a_rebuild(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    page = vault / "entities" / "mateo.md"
    page.write_text(page.read_text(encoding="utf-8").replace("500 flyers", "600 flyers"),
                    encoding="utf-8", newline="\n")
    rebuild_from_vault(vault, tmp_path / "edited.db")
    edited = Memory.open(tmp_path / "edited.db")
    [order] = edited.facts("mateo", "order")
    assert order.value.startswith("600 flyers")
    assert edited.search("600 flyers").hits[0].fact.id == order.id
    edited.close()


def test_export_removes_stale_managed_pages_but_keeps_other_files(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    stale = vault / "entities" / "ghost.md"
    stale.write_text((vault / "entities" / "acme.md").read_text(encoding="utf-8")
                     .replace('id: "acme"', 'id: "ghost"'), encoding="utf-8")
    notes = vault / "entities" / "my-notes.md"
    notes.write_text("# scratch notes\n", encoding="utf-8")
    report = busy.export_vault(vault)
    assert report.removed == ["entities/ghost.md"]
    assert not stale.exists()
    assert notes.exists()


# -- strict validation --------------------------------------------------------------------

def corrupt(vault, page, old, new):
    path = vault / "entities" / page
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    ("page", "old", "new", "message"),
    [
        ("shop.md", "  ^f-", "  ^bad id ", "shop.md:1[0-9]: "),
        ("shop.md", "valid 2026", "valid yesterday", "timestamp"),
        ("shop.md", "superseded", "vanished", "vanished"),
        ("priya.md", "[[acme]]", "[[nowhere]]", "unknown node 'nowhere'"),
        ("priya.md", "private:priya", "private:shop", "not a contact"),
        ("shop.md", "by f-0", "by f-9", "unknown fact"),
        ("mateo.md", "{at ", "{when ", "told line"),
    ],
    ids=["bad-id", "bad-time", "bad-status", "dangling-object", "bad-audience",
         "dangling-successor", "bad-told-line"],
)
def test_malformed_vaults_fail_with_file_and_line(busy, tmp_path, page, old, new, message):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    corrupt(vault, page, old, new)
    with pytest.raises(VaultError, match=message):
        read_vault(vault)


def test_overlapping_single_valued_facts_are_rejected(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    page = vault / "entities" / "mateo.md"
    lines = page.read_text(encoding="utf-8").splitlines()
    [line] = [x for x in lines if x.startswith("- order:")]
    lines.insert(lines.index(line) + 1, line.replace("^f-", "^f-9").replace(
        "- order: 500", "- order: 900"))
    page.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(VaultError, match="overlap"):
        read_vault(vault)


def test_disclosures_must_be_visible_to_their_contact(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    budget = busy.remember("priya", "budget", "$480")
    busy.export_vault(vault)
    page = vault / "entities" / "mateo.md"
    page.write_text(page.read_text(encoding="utf-8") + (
        f"- [[priya#^{budget.id}|budget]]  {{at 2026-03-02T09:00:00Z}}  ^d-999999\n"),
        encoding="utf-8", newline="\n")
    with pytest.raises(VaultError, match="not visible"):
        read_vault(vault)


def test_unknown_vault_versions_are_rejected(busy, tmp_path):
    vault = tmp_path / "vault"
    busy.export_vault(vault)
    index = vault / "index.md"
    index.write_text(index.read_text(encoding="utf-8").replace("tgmem-vault: 1",
                                                                "tgmem-vault: 99"),
                     encoding="utf-8", newline="\n")
    with pytest.raises(VaultError, match="version"):
        read_vault(vault)
