"""Nightly consolidation: expire lapsed facts, fold duplicates, merge near-duplicates."""

from datetime import timedelta

from conftest import seed
from fakes import T0, SwitchableEmbedder

from tgmem.embeddings import CircuitBreaker, HashingEmbedder
from tgmem.ids import SequentialIds
from tgmem.memory import Memory
from tgmem.timeutil import ManualClock

DAY = timedelta(days=1)


def test_expire_marks_lapsed_facts_without_changing_what_was_true(mem, clock):
    promo = mem.remember("shop", "promo", "free delivery", ttl=7 * DAY)
    hours = mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(8 * DAY)
    report = mem.consolidate()
    assert report.expired == [promo.id]
    lapsed = mem.fact(promo.id)
    assert (lapsed.status, lapsed.ended_at) == ("expired", report.ran_at)
    assert mem.fact(hours.id).status == "active"
    assert mem.facts("shop", "promo", as_of=promo.valid_from) == [lapsed]
    assert mem.consolidate().changed is False


def test_exact_duplicates_are_folded_into_the_oldest(mem, clock):
    first = mem.remember("priya", "note", "Prefers matte finish.")
    clock.advance(DAY)
    second = mem.remember("priya", "note", "prefers  matte finish")
    assert second.id != first.id  # different text, so remember() kept both
    report = mem.consolidate()
    assert report.duplicates == [(second.id, first.id)]
    folded = mem.fact(second.id)
    assert (folded.status, folded.superseded_by) == ("duplicate", first.id)
    assert mem.facts("priya", "note") == [first]
    assert len(mem.history("priya", "note")) == 2


def test_near_duplicates_merge_into_the_most_informative(mem):
    short = mem.remember("priya", "note", "prefers matte finish")
    longer = mem.remember("priya", "note", "prefers matte finish on business cards")
    variant = mem.remember("priya", "note", "prefers matte finish for business cards")
    other = mem.remember("priya", "note", "pays invoices by bank transfer")
    report = mem.consolidate()
    assert sorted(report.merged) == sorted([(short.id, longer.id), (variant.id, longer.id)])
    assert sorted(f.id for f in mem.facts("priya", "note")) == sorted([longer.id, other.id])


def test_merge_threshold_is_configurable(mem):
    blue = mem.remember("priya", "note", "wants blue envelopes for the invitations")
    cream = mem.remember("priya", "note", "wants cream envelopes for the invitations")
    assert mem.consolidate().merged == []  # the default keeps conflicting notes apart
    assert mem.consolidate(merge_threshold=0.6).merged == [(cream.id, blue.id)]


def test_single_valued_facts_are_never_merged(mem, clock):
    mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(DAY)
    mem.remember("shop", "opening_hours", "9am-5pm Monday to Saturday")
    report = mem.consolidate()
    assert report.merged == [] and report.duplicates == []


def test_nothing_is_ever_merged_across_audiences(mem):
    public = mem.remember("shop", "note", "closed on public holidays")
    private = mem.remember("shop", "note", "closed on public holidays", scope="private:priya")
    internal = mem.remember("shop", "note", "closed on public holidays", scope="internal")
    report = mem.consolidate()
    assert report.duplicates == [] and report.merged == []
    assert {f.id for f in mem.facts("shop", "note")} == {public.id, private.id, internal.id}


def test_disclosures_follow_folded_facts(mem, clock):
    kept = mem.remember("priya", "note", "invoices go to accounts@example.com")
    clock.advance(DAY)
    dup = mem.remember("priya", "note", "Invoices go to accounts@example.com.")
    mem.record_disclosure("priya", [dup.id])
    mem.consolidate()
    assert mem.told_at("priya", kept) is not None
    assert mem.corrections_owed("priya") == []


def test_dry_run_reports_but_changes_nothing(mem, clock):
    promo = mem.remember("shop", "promo", "free delivery", ttl=DAY)
    mem.remember("priya", "note", "likes linen")
    mem.remember("priya", "note", "Likes linen.")
    clock.advance(2 * DAY)
    report = mem.consolidate(dry_run=True)
    assert report.dry_run and report.expired == [promo.id] and len(report.duplicates) == 1
    assert mem.fact(promo.id).status == "active"
    assert len(mem.facts("priya", "note")) == 2
    assert mem.consolidate().expired == [promo.id]  # the real run still has work to do


def test_consolidation_backfills_missing_embeddings(tmp_path):
    switch = SwitchableEmbedder(HashingEmbedder(64))
    switch.down = True
    memory = Memory.open(tmp_path / "m.db", clock=ManualClock(T0), ids=SequentialIds(),
                         embedder=switch, breaker=CircuitBreaker(failure_threshold=99))
    seed(memory)
    memory.remember("shop", "opening_hours", "9am-5pm")
    memory.remember("shop", "turnaround", "3 days")
    switch.down = False
    report = memory.consolidate()
    assert (report.embedded, report.pending_embeddings) == (2, 0)
    memory.close()


def test_folded_facts_leave_search_results(mem):
    mem.remember("priya", "note", "prefers matte finish")
    keep = mem.remember("priya", "note", "prefers matte finish on business cards")
    mem.consolidate()
    hits = mem.search("matte finish").hits
    assert [h.fact.id for h in hits] == [keep.id]


def test_the_vault_moves_expired_facts_to_history(mem, clock, tmp_path):
    mem.remember("shop", "promo", "free delivery", ttl=DAY)
    mem.export_vault(tmp_path / "vault")
    page = tmp_path / "vault" / "entities" / "shop.md"
    assert "## History" not in page.read_text(encoding="utf-8")
    clock.advance(2 * DAY)
    mem.consolidate()
    report = mem.export_vault(tmp_path / "vault")
    assert "entities/shop.md" in report.written
    history = page.read_text(encoding="utf-8").split("## History")[1]
    assert "free delivery" in history and "expired" in history


def test_summary_is_human_readable(mem, clock):
    mem.remember("shop", "promo", "free delivery", ttl=DAY)
    clock.advance(2 * DAY)
    text = mem.consolidate().summary()
    assert "expired 1" in text and "duplicates 0" in text and "merged 0" in text
