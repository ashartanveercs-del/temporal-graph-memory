"""Stress and regression suite. Everything is seeded, so every run is identical; the fake
embedder is deterministic feature hashing, so vector rankings are reproducible too."""

import random
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from fakes import T0, SwitchableEmbedder

from tgmem.embeddings import CircuitBreaker, HashingEmbedder
from tgmem.ids import SequentialIds
from tgmem.memory import Memory
from tgmem.model import Viewer
from tgmem.timeutil import ManualClock
from tgmem.vault import rebuild_from_vault

HOUR, DAY = timedelta(hours=1), timedelta(days=1)


class Timeline:
    """An independent model of single-valued valid-time semantics: a list of
    non-overlapping (start, end, value) segments. Written separately from the library so
    the two can be compared."""

    def __init__(self) -> None:
        self.segments: list[tuple[datetime, datetime | None, str]] = []

    def set(self, start: datetime, end: datetime | None, value: str) -> None:
        for s, e, v in self.segments:
            if v == value and s <= start and (e is None or (end is not None and end <= e)):
                return  # already true over that whole interval
        kept = []
        for s, e, v in self.segments:
            overlap = (end is None or s < end) and (e is None or start < e)
            if not overlap:
                kept.append((s, e, v))
                continue
            if s < start:
                kept.append((s, start, v))
            if end is not None and (e is None or end < e):
                kept.append((end, e, v))
        self.segments = sorted([*kept, (start, end, value)], key=lambda seg: seg[0])

    def remove(self, start: datetime, end: datetime | None, value: str) -> None:
        self.segments.remove((start, end, value))

    def at(self, when: datetime) -> list[str]:
        return [v for s, e, v in self.segments if s <= when and (e is None or when < e)]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_histories_match_an_independent_model(tmp_path, seed):
    rng = random.Random(seed)
    clock = ManualClock(T0)
    memory = Memory.open(tmp_path / "m.db", clock=clock, ids=SequentialIds())
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    model = Timeline()
    snapshots: list[tuple[datetime, list]] = []
    probes = [T0 + timedelta(hours=h) for h in range(-30 * 24, 60 * 24, 17)]

    for _ in range(200):
        roll = rng.random()
        believed = memory.timeline("shop", "status")
        if roll < 0.5:
            start = clock.now().replace(minute=0, second=0, microsecond=0) + rng.randint(
                -240, 240) * HOUR
            end = start + rng.randint(1, 120) * HOUR if rng.random() < 0.4 else None
            value = rng.choice("abcd")
            memory.remember("shop", "status", value, valid_from=start, valid_to=end)
            model.set(start, end, value)
        elif roll < 0.65 and believed:
            target = rng.choice(believed)
            memory.retract(target.id)
            model.remove(target.valid_from, target.valid_to, target.value)
        elif roll < 0.8 and believed:
            target = rng.choice(believed)
            value = rng.choice("abcde")
            memory.correct(target.id, value)
            model.set(target.valid_from, target.valid_to, value)
        else:
            clock.advance(rng.randint(1, 48) * HOUR)
        # Key snapshots by the last transaction time, not the wall clock: the next write
        # is stamped strictly after it, but may be stamped exactly at the wall clock.
        snapshots.append((memory.store.last_tx_time(), list(model.segments)))

        timeline = memory.timeline("shop", "status")
        for earlier, later in zip(timeline, timeline[1:], strict=False):
            assert earlier.valid_to is not None and earlier.valid_to <= later.valid_from
        for when in rng.sample(probes, 8):
            assert sorted(f.value for f in memory.facts("shop", "status", as_of=when)) == (
                sorted(model.at(when)))

    # Bitemporal check: what the memory believed after step i is exactly what the
    # model held after step i, for every probed point in valid time.
    for known_at, segments in rng.sample(snapshots, 25):
        past = Timeline()
        past.segments = segments
        for when in rng.sample(probes, 6):
            believed_then = memory.facts("shop", "status", as_of=when, known_at=known_at)
            assert sorted(f.value for f in believed_then) == sorted(past.at(when))
    memory.close()


def build_customers(memory: Memory, rng: random.Random, count: int) -> dict[str, list]:
    private: dict[str, list] = {}
    with memory.bulk():
        memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
        for i in range(count):
            node = f"c{i:03d}"
            memory.add_contact(f"Customer {i:03d}", [f"+1555000{i:04d}"], id=node)
            private[node] = [
                memory.remember(node, "budget", f"${rng.randint(100, 999)} for project {i}"),
                memory.remember(node, "order", f"{rng.randint(10, 99)} posters, order {i}"),
                memory.remember(node, "note", f"prefers {rng.choice(['matte', 'gloss'])} "
                                              f"paper for project {i}"),
            ]
        for j in range(100):
            memory.remember("shop", f"faq_{j:03d}", f"answer {j} about posters and paper")
    return private


def test_isolation_holds_at_scale(tmp_path):
    rng = random.Random(7)
    memory = Memory.open(tmp_path / "m.db", clock=ManualClock(T0), ids=SequentialIds(),
                         embedder=HashingEmbedder(128))
    private = build_customers(memory, rng, 300)
    assert memory.stats()["pending_embeddings"] == 0  # the batch embedded on exit
    started = time.perf_counter()
    for _ in range(100):
        asker, target = rng.sample(sorted(private), 2)
        viewer = Viewer.contact(asker)
        secret = rng.choice(private[target])
        result = memory.search(secret.value, viewer=viewer, k=10)
        assert result.mode == "hybrid"
        assert all(viewer.can_see(hit.fact.scope) for hit in result.hits)
        assert not {f.id for f in private[target]} & {hit.fact.id for hit in result.hits}
        if secret.predicate == "budget":
            assert secret in memory.leaks(viewer, f"I heard it was {secret.value}")
    per_query = (time.perf_counter() - started) / 100
    assert per_query < 0.5, f"{per_query:.3f}s per search+check"
    memory.close()


def test_an_embedder_outage_never_breaks_reads_or_writes(tmp_path):
    clock = ManualClock(T0)
    switch = SwitchableEmbedder(HashingEmbedder(64))
    memory = Memory.open(tmp_path / "m.db", clock=clock, ids=SequentialIds(), embedder=switch,
                         breaker=CircuitBreaker(failure_threshold=2, clock=clock,
                                                reset_after=timedelta(minutes=1)))
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    for i in range(60):
        switch.down = 20 <= i < 40
        memory.remember("shop", f"faq_{i:02d}", f"posters in size {i} ship in {i % 5 + 1} days")
        if switch.down:
            result = memory.search("posters ship days")
            assert result.mode == "keyword" and result.hits
    assert memory.stats()["pending_embeddings"] > 0
    clock.advance(minutes=2)
    report = memory.reindex()
    assert (report.pending, report.error) == (0, None)
    assert memory.search("posters ship days").mode == "hybrid"
    memory.close()


def test_concurrent_writers_keep_one_linear_history(tmp_path):
    path = tmp_path / "m.db"
    setup = Memory.open(path)
    setup.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    setup.close()
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            memory = Memory.open(path)
            for i in range(25):
                memory.remember("shop", "status", f"writer {n} update {i}")
            memory.close()
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert errors == []
    memory = Memory.open(path)
    assert len(memory.facts("shop", "status")) == 1
    history = memory.history("shop", "status")
    originals = [f for f in history if f.derived_from is None]
    assert len(originals) == 100
    assert len({f.recorded_at for f in originals}) == 100  # one transaction time each
    timeline = memory.timeline("shop", "status")
    for earlier, later in zip(timeline, timeline[1:], strict=False):
        assert earlier.valid_to == later.valid_from  # contiguous, no gaps, no overlaps
    ids = {f.id for f in history}
    assert all(f.superseded_by in ids for f in history if f.superseded_by)
    memory.close()


AWKWARD = "abc XYZ 123 {}[]()\\/\"'`$%;:#^|~-_=+.,!?\n\t éü 東京 🙂 مرحبا"


def test_awkward_values_survive_the_vault_round_trip(tmp_path):
    rng = random.Random(11)
    memory = Memory.open(tmp_path / "m.db", clock=ManualClock(T0), ids=SequentialIds())
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    memory.add_contact("Priya Shah", ["+1 555 000 0101"], id="priya")
    values = ["[[shop]]", "[[not a link", "\\", "ends with a backslash\\", "x  {y}  ^f-1",
              "line one\nline two", "tab\there"]
    values += ["".join(rng.choice(AWKWARD) for _ in range(rng.randint(1, 60)))
               for _ in range(200)]
    written = 0
    for value in values:
        try:
            memory.remember(rng.choice(["shop", "priya"]), "note", value,
                            scope=rng.choice(["public", "internal", "private:priya"]))
            written += 1
        except ValueError:
            assert not value.strip()  # only blank values are refused
    memory.export_vault(tmp_path / "vault")
    before = {p.name: p.read_bytes() for p in (tmp_path / "vault").rglob("*.md")}
    rebuild_from_vault(tmp_path / "vault", tmp_path / "rebuilt.db")
    rebuilt = Memory.open(tmp_path / "rebuilt.db")
    for subject in ("shop", "priya"):
        assert rebuilt.history(subject) == memory.history(subject)
    rebuilt.export_vault(tmp_path / "vault2")
    after = {p.name: p.read_bytes() for p in (tmp_path / "vault2").rglob("*.md")}
    assert after == before
    assert written > 190
    rebuilt.close()
    memory.close()


def scenario(path) -> tuple[dict, list]:
    rng = random.Random(3)
    clock = ManualClock(T0)
    memory = Memory.open(path / "m.db", clock=clock, ids=SequentialIds(),
                         embedder=HashingEmbedder(64))
    build_customers(memory, rng, 40)
    for _ in range(30):
        clock.advance(rng.randint(1, 30) * HOUR)
        node = f"c{rng.randrange(40):03d}"
        memory.remember(node, "budget", f"${rng.randint(100, 999)} revised")
    memory.consolidate()
    memory.export_vault(path / "vault")
    ranking = [(h.fact.id, h.score) for h in memory.search("posters paper budget", k=15).hits]
    vault = {p.name: p.read_bytes() for p in (path / "vault").rglob("*.md")}
    memory.close()
    return vault, ranking


def test_identical_inputs_give_identical_vaults_and_rankings(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert scenario(tmp_path / "a") == scenario(tmp_path / "b")


def test_consolidation_is_idempotent_at_scale(tmp_path):
    rng = random.Random(5)
    memory = Memory.open(tmp_path / "m.db", clock=ManualClock(T0), ids=SequentialIds())
    memory.add_contact("Priya Shah", ["+1 555 000 0101"], id="priya")
    phrasings = ["prefers {c} card stock", "Prefers {c} card stock.", "prefers {c} card stock "
                 "for invitations", "likes {c} envelopes", "pays invoices within {n} days"]
    with memory.bulk():
        for _ in range(500):
            template = rng.choice(phrasings)
            memory.remember("priya", "note", template.format(
                c=rng.choice(["cream", "white", "ivory"]), n=rng.choice([7, 14, 30])))
    first = memory.consolidate()
    assert first.duplicates or first.merged
    second = memory.consolidate()
    assert second.changed is False
    survivors = [f.value for f in memory.facts("priya", "note")]
    normalized = {" ".join(v.casefold().rstrip(".").split()) for v in survivors}
    assert len(normalized) == len(survivors)  # no two survivors say the same thing
    memory.close()


def _forgetful_override(self, targets, new, now):
    """A plausible bug: temporary overrides never hand the old value back afterwards."""
    for old in targets:
        keep_left = old.valid_from < new.valid_from
        self.store.end_fact(old.id, status="superseded" if keep_left else "corrected",
                            ended_at=now, superseded_by=new.id)
        if keep_left:
            self.store.insert_fact(replace(
                old, id=self.ids.new("f"), valid_to=new.valid_from, recorded_at=now,
                status="expired" if new.valid_from <= now else "active",
                ended_at=now if new.valid_from <= now else None, superseded_by=None,
                derived_from=old.id))


def test_the_model_check_has_teeth(tmp_path, monkeypatch):
    monkeypatch.setattr(Memory, "_override", _forgetful_override)
    with pytest.raises(AssertionError):
        test_random_histories_match_an_independent_model(tmp_path, 1)
