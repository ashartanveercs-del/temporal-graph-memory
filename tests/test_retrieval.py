"""Hybrid retrieval: keyword + vector + graph signals, scoping, and graceful fallback."""

from datetime import timedelta

import pytest
from conftest import seed
from fakes import MATEO, PRIYA, T0, FailingEmbedder, SwitchableEmbedder

from tgmem.embeddings import CircuitBreaker, HashingEmbedder
from tgmem.ids import SequentialIds
from tgmem.memory import Memory
from tgmem.model import Viewer
from tgmem.timeutil import ManualClock

SYNONYMS = {"bill": "invoice", "bills": "invoice", "billing": "invoice", "billed": "invoice"}


def open_memory(tmp_path, name="m.db", **kwargs):
    kwargs.setdefault("clock", ManualClock(T0))
    memory = Memory.open(tmp_path / name, ids=SequentialIds(), **kwargs)
    seed(memory)
    return memory


def add_business_facts(memory):
    memory.upsert_entity("acme", "Acme Logistics", kind="org")
    return {
        "terms": memory.remember("shop", "invoice_terms", "net 30, paid by bank transfer"),
        "hours": memory.remember("shop", "opening_hours", "9am-5pm Monday to Saturday"),
        "turnaround": memory.remember("shop", "turnaround", "business cards ship in 3 days"),
        "budget": memory.remember("priya", "budget", "$480 for wedding invitations"),
        "order": memory.remember("mateo", "order", "500 flyers due Friday"),
        "acme_terms": memory.remember("acme", "payment_terms", "net 45"),
    }


@pytest.fixture
def hybrid(tmp_path):
    memory = open_memory(tmp_path, embedder=HashingEmbedder(512, aliases=SYNONYMS))
    facts = add_business_facts(memory)
    yield memory, facts
    memory.close()


def ids(result):
    return [hit.fact.id for hit in result.hits]


def test_without_an_embedder_search_is_keyword_only(tmp_path):
    memory = open_memory(tmp_path)
    facts = add_business_facts(memory)
    result = memory.search("opening hours on Saturday")
    assert (result.mode, result.degraded) == ("keyword", None)
    assert ids(result)[0] == facts["hours"].id
    assert len(memory.search("net", k=1).hits) == 1
    memory.close()


def test_vector_search_finds_what_keywords_miss(hybrid, tmp_path):
    memory, facts = hybrid
    result = memory.search("billing?")
    assert result.mode == "hybrid"
    assert ids(result)[0] == facts["terms"].id
    assert result.hits[0].signals == {"vector": 1}
    keyword_only = open_memory(tmp_path, name="kw.db")
    add_business_facts(keyword_only)
    assert keyword_only.search("billing?").hits == ()
    keyword_only.close()


def test_hits_record_which_signals_found_them(hybrid):
    memory, facts = hybrid
    hit = memory.search("invoice terms").hits[0]
    assert hit.fact.id == facts["terms"].id
    assert set(hit.signals) == {"keyword", "vector"}
    assert hit.score > 0


def test_embedder_failure_falls_back_to_keywords_and_never_blocks_writes(tmp_path):
    failing = FailingEmbedder()
    memory = open_memory(tmp_path, embedder=failing)
    facts = add_business_facts(memory)  # every write tried to embed and failed quietly
    assert memory.stats()["pending_embeddings"] > 0
    result = memory.search("invoice terms")
    assert result.mode == "keyword"
    assert "vector search" in result.degraded
    assert ids(result)[0] == facts["terms"].id
    memory.close()


def test_the_circuit_breaker_stops_calling_a_dead_embedder(tmp_path):
    failing = FailingEmbedder()
    clock = ManualClock(T0)
    memory = open_memory(tmp_path, embedder=failing, clock=clock,
                         breaker=CircuitBreaker(failure_threshold=3, clock=clock))
    add_business_facts(memory)
    calls = failing.calls
    for _ in range(5):
        memory.search("invoice terms")
    assert failing.calls == calls  # the breaker opened during the writes
    assert "paused" in memory.search("invoice terms").degraded
    clock.advance(minutes=5)
    memory.search("invoice terms")
    assert failing.calls == calls + 1  # one probe after the cooldown
    memory.close()


def test_reindex_backfills_vectors_after_an_outage(tmp_path):
    switch = SwitchableEmbedder(HashingEmbedder(512, aliases=SYNONYMS))
    switch.down = True
    memory = open_memory(tmp_path, embedder=switch, breaker=CircuitBreaker(failure_threshold=99))
    facts = add_business_facts(memory)
    assert memory.search("billing?").hits == ()
    switch.down = False
    report = memory.reindex()
    assert report.embedded == memory.stats()["believed_texts"]
    assert (report.pending, report.error) == (0, None)
    assert ids(memory.search("billing?"))[0] == facts["terms"].id
    memory.close()


def test_reindex_reports_failures_and_leaves_texts_pending(tmp_path):
    memory = open_memory(tmp_path, embedder=FailingEmbedder())
    add_business_facts(memory)
    report = memory.reindex()
    assert report.embedded == 0
    assert report.pending > 0
    assert report.error
    memory.close()


def test_vectors_from_another_model_are_never_compared(tmp_path):
    first = open_memory(tmp_path, embedder=HashingEmbedder(64, aliases=SYNONYMS))
    facts = add_business_facts(first)
    first.close()
    second = Memory.open(tmp_path / "m.db", clock=ManualClock(T0),
                         embedder=HashingEmbedder(32, aliases=SYNONYMS))
    assert second.search("billing?").hits == ()  # cached 64-d vectors are ignored
    assert second.stats()["pending_embeddings"] > 0
    second.reindex()
    assert ids(second.search("billing?"))[0] == facts["terms"].id
    second.close()


def test_scoping_applies_to_every_signal(hybrid):
    memory, facts = hybrid
    query = "budget for wedding invitations"
    owner = memory.search(query)
    assert ids(owner)[0] == facts["budget"].id
    for viewer in (memory.viewer_for(MATEO), Viewer.anonymous()):
        result = memory.search(query, viewer=viewer)
        assert facts["budget"].id not in ids(result)
        assert all(viewer.can_see(hit.fact.scope) for hit in result.hits)
    assert ids(memory.search(query, viewer=memory.viewer_for(PRIYA)))[0] == facts["budget"].id


def test_search_supports_point_in_time(hybrid):
    memory, _ = hybrid
    before = memory.now()
    memory.clock.advance(timedelta(days=2))
    memory.remember("shop", "opening_hours", "10am-6pm Monday to Saturday")
    assert memory.search("opening hours").hits[0].fact.value == "10am-6pm Monday to Saturday"
    assert memory.search("opening hours", as_of=before).hits[0].fact.value == (
        "9am-5pm Monday to Saturday")


def test_graph_signal_follows_relationships_one_hop(hybrid):
    memory, facts = hybrid
    memory.remember("priya", "works_at", object="acme")  # private to Priya by default
    query = "When does Priya's employer settle up?"
    assert facts["acme_terms"].id in ids(memory.search(query))  # owner: via works_at
    priya = memory.search(query, viewer=memory.viewer_for(PRIYA))
    assert facts["acme_terms"].id in ids(priya)
    mateo = memory.search(query, viewer=memory.viewer_for(MATEO))
    # Acme's terms are public, but the only path to them is a relation private to Priya.
    assert facts["acme_terms"].id not in ids(mateo)


def test_first_person_queries_point_at_the_sender(hybrid):
    memory, facts = hybrid
    result = memory.search("what is my order?", viewer=memory.viewer_for(MATEO))
    assert ids(result)[0] == facts["order"].id
    assert "graph" in result.hits[0].signals


@pytest.mark.parametrize("fts", [True, False])
@pytest.mark.parametrize("query", ['invoice "due', "terms) OR (1=1", "NEAR(net 30)",
                                   "value: secret", "-net *", "'; DROP TABLE edges; --"])
def test_hostile_queries_never_raise(tmp_path, fts, query):
    memory = open_memory(tmp_path, fts=fts)
    add_business_facts(memory)
    memory.search(query)
    memory.search(query, viewer=Viewer.anonymous())
    memory.close()


def test_like_fallback_without_fts5(tmp_path):
    memory = open_memory(tmp_path, fts=False)
    facts = add_business_facts(memory)
    assert not memory.store.fts_enabled
    assert ids(memory.search("invoice terms"))[0] == facts["terms"].id
    memory.close()


def test_rankings_are_deterministic(tmp_path):
    def run(name):
        memory = open_memory(tmp_path, name=name, embedder=HashingEmbedder(64))
        add_business_facts(memory)
        for i in range(30):
            memory.remember("shop", "note", f"customer note {i % 7} about cards and invoices")
        out = [(h.fact.id, h.score) for h in memory.search("cards invoices note", k=20).hits]
        memory.close()
        return out

    assert run("a.db") == run("b.db")


def test_purge_removes_vectors_and_search_hits(hybrid):
    memory, facts = hybrid
    before = memory.stats()["embeddings"]
    memory.purge(facts["budget"].id)
    assert memory.stats()["embeddings"] == before - 1
    assert facts["budget"].id not in ids(memory.search("wedding invitations budget"))
