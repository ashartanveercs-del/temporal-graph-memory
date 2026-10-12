"""The GraphStore contract. Any backend must pass this suite; it runs against SQLite with
FTS5 and against SQLite's LIKE fallback (what you get on a build without FTS5)."""

from datetime import timedelta

import numpy as np
import pytest
from fakes import T0, make_fact, make_node

from tgmem.model import (
    ACTIVE,
    CORRECTED,
    EXPIRED,
    INTERNAL,
    PUBLIC,
    SUPERSEDED,
    Disclosure,
    private_scope,
)
from tgmem.store import HandleConflict, SQLiteGraphStore


@pytest.fixture(params=["fts5", "like"])
def store(request, tmp_path):
    s = SQLiteGraphStore(tmp_path / "memory.db", fts=request.param == "fts5")
    seed_nodes(s)
    yield s
    s.close()


def seed_nodes(s):
    s.put_node(make_node("shop", "Lakeside Print Shop", kind="org"))
    s.put_node(make_node("priya", "Priya Shah", kind="person", role="contact",
                         handles=("+15550000101",)))
    s.put_node(make_node("mateo", "Mateo Ruiz", kind="person", role="contact",
                         handles=("+15550000102", "mateo@example.com")))


def test_fts_mode_is_reported(store, request):
    assert store.fts_enabled == (request.node.callspec.params["store"] == "fts5")


def test_schema_creation_is_idempotent(tmp_path):
    path = tmp_path / "memory.db"
    SQLiteGraphStore(path).close()
    again = SQLiteGraphStore(path)
    assert again.counts()["facts"] == 0
    again.close()


def test_node_round_trip(store):
    node = store.get_node("mateo")
    assert node.name == "Mateo Ruiz"
    assert node.handles == ("+15550000102", "mateo@example.com")
    assert node.role == "contact"
    assert node.created_at == T0
    assert store.node_for_handle("mateo@example.com").id == "mateo"
    assert store.node_for_handle("+15550000199") is None
    assert [n.id for n in store.nodes()] == ["mateo", "priya", "shop"]


def test_handles_belong_to_exactly_one_node(store):
    with pytest.raises(HandleConflict):
        store.put_node(make_node("impostor", role="contact", handles=("+15550000101",)))
    assert store.get_node("impostor") is None  # the failed write left nothing behind
    assert store.node_for_handle("+15550000101").id == "priya"


def test_put_node_updates_in_place_and_replaces_handles(store):
    version = store.nodes_version()
    store.put_node(make_node("priya", "Priya S.", kind="person", role="contact",
                             handles=("+15550000111",), created_at=T0 + timedelta(days=9)))
    node = store.get_node("priya")
    assert (node.name, node.handles) == ("Priya S.", ("+15550000111",))
    assert node.created_at == T0  # creation time is never rewritten
    assert store.node_for_handle("+15550000101") is None
    assert store.nodes_version() > version


def test_fact_round_trip_preserves_every_field(store):
    original = make_fact(
        "f-7", subject="priya", predicate="works_at", value=None, object="shop",
        scope=private_scope("priya"), valid_from=T0.replace(microsecond=123456),
        valid_to=T0 + timedelta(days=30), recorded_at=T0 + timedelta(seconds=1),
        asserted_by="priya", source="sms", text="Priya Shah works at Lakeside Print Shop")
    store.insert_fact(original)
    assert store.get_fact("f-7") == original
    assert store.get_fact("missing") is None


def test_facts_must_reference_existing_nodes(store):
    with pytest.raises(ValueError):
        store.insert_fact(make_fact(subject="nobody"))
    with pytest.raises(ValueError):
        store.insert_fact(make_fact(value=None, object="nobody"))


def test_end_fact_records_status_time_and_successor(store):
    store.insert_fact(make_fact("f-1"))
    store.insert_fact(make_fact("f-2", value="9am-6pm"))
    store.end_fact("f-1", status=SUPERSEDED, ended_at=T0 + timedelta(hours=1),
                   superseded_by="f-2")
    ended = store.get_fact("f-1")
    assert (ended.status, ended.ended_at, ended.superseded_by) == (
        SUPERSEDED, T0 + timedelta(hours=1), "f-2")


def test_query_facts_filters_and_orders(store):
    day = timedelta(days=1)
    store.insert_fact(make_fact("f-a", valid_to=T0 + day))
    store.insert_fact(make_fact("f-b", value="9am-6pm", valid_from=T0 + day))
    store.insert_fact(make_fact("f-c", subject="priya", predicate="budget", value="$480",
                                scope=private_scope("priya")))
    store.insert_fact(make_fact("f-d", predicate="margin", value="40%", scope=INTERNAL,
                                status=CORRECTED, ended_at=T0))
    store.insert_fact(make_fact("f-e", subject="priya", predicate="works_at", value=None,
                                object="shop"))

    def ids(**kw):
        return [f.id for f in store.query_facts(**kw)]

    assert ids(subject="shop", predicate="opening_hours") == ["f-a", "f-b"]
    assert ids(subject="shop", valid_at=T0) == ["f-d", "f-a"]
    assert ids(subject="shop", valid_at=T0 + day) == ["f-d", "f-b"]  # half-open: f-a has ended
    assert ids(scopes=(PUBLIC, private_scope("priya")), subject="priya") == ["f-c", "f-e"]
    assert ids(scopes=(PUBLIC,), subject="priya") == ["f-e"]
    assert ids(statuses=(CORRECTED,)) == ["f-d"]
    assert ids(node="shop", statuses=(ACTIVE,)) == ["f-e", "f-a", "f-b"]
    assert ids(ids=["f-e", "f-a", "nope"]) == ["f-e", "f-a"]


def test_facts_known_at_reconstructs_past_belief(store):
    h = timedelta(hours=1)
    store.insert_fact(make_fact("f-old"))  # recorded T0
    store.insert_fact(make_fact("f-new", value="9am-6pm", recorded_at=T0 + 2 * h))
    store.end_fact("f-old", status=CORRECTED, ended_at=T0 + 2 * h, superseded_by="f-new")
    store.insert_fact(make_fact("f-lapsed", predicate="promo", value="10% off",
                                valid_to=T0 + h, status=EXPIRED, ended_at=T0 + h))

    def known(at):
        return [f.id for f in store.facts_known_at(at, valid_at=T0 + timedelta(minutes=30))]

    assert known(T0 + h) == ["f-old", "f-lapsed"]
    assert known(T0 + 3 * h) == ["f-new", "f-lapsed"]  # expired rows stay believed
    assert known(T0 - h) == []


def test_transactions_roll_back_completely(store):
    with pytest.raises(RuntimeError), store.transaction():
        store.insert_fact(make_fact("f-1"))
        store.put_node(make_node("ghost"))
        raise RuntimeError("boom")
    assert store.get_fact("f-1") is None
    assert store.get_node("ghost") is None


def test_nested_transactions_join_the_outer_one(store):
    with pytest.raises(RuntimeError), store.transaction():
        with store.transaction():
            store.insert_fact(make_fact("f-1"))
        raise RuntimeError("outer fails after inner finished")
    assert store.get_fact("f-1") is None


def test_transaction_time_is_strictly_monotonic(store):
    with store.transaction():
        first = store.next_tx_time(T0)
        same = store.next_tx_time(T0)
        backwards = store.next_tx_time(T0 - timedelta(hours=5))  # wall clock jumped back
        later = store.next_tx_time(T0 + timedelta(hours=1))
    assert first == T0
    assert first < same < backwards < later
    assert store.last_tx_time() == later


def test_transaction_time_requires_a_transaction(store):
    with pytest.raises(RuntimeError):
        store.next_tx_time(T0)


def test_transaction_time_survives_reopening(tmp_path):
    path = tmp_path / "memory.db"
    first = SQLiteGraphStore(path)
    with first.transaction():
        stamped = first.next_tx_time(T0 + timedelta(days=3))
    first.close()
    second = SQLiteGraphStore(path)
    with second.transaction():
        assert second.next_tx_time(T0) > stamped
    second.close()


def test_keyword_search_respects_scope_status_and_validity(store):
    store.insert_fact(make_fact("f-hours", text="Lakeside Print Shop opening hours 9am-5pm"))
    store.insert_fact(make_fact("f-budget", subject="priya", predicate="budget", value="$480",
                                scope=private_scope("priya"), text="Priya Shah budget 480"))
    store.insert_fact(make_fact("f-old", predicate="budget_note", value="old",
                                status=SUPERSEDED, ended_at=T0, text="shop budget old"))
    store.insert_fact(make_fact("f-later", predicate="budget_plan", value="later",
                                valid_from=T0 + timedelta(days=5), text="shop budget later"))

    def hits(terms, scopes):
        return [fid for fid, _ in store.keyword_search(terms, scopes=scopes, valid_at=T0,
                                                       limit=10)]

    assert hits(["budget"], None) == ["f-budget"]
    assert hits(["budget"], (PUBLIC,)) == []
    assert hits(["budget"], (PUBLIC, private_scope("mateo"))) == []
    assert hits(["opening", "hours"], (PUBLIC,)) == ["f-hours"]
    assert hits(["nothing"], None) == []


def test_keyword_search_ranks_better_matches_first(store):
    store.insert_fact(make_fact("f-1", text="invoice due date for order 1042"))
    store.insert_fact(make_fact("f-2", predicate="p2", text="invoice template"))
    store.insert_fact(make_fact("f-3", predicate="p3", text="business card paper stock"))
    ranked = store.keyword_search(["invoice", "due", "1042"], scopes=None, valid_at=T0, limit=5)
    assert [fid for fid, _ in ranked] == ["f-1", "f-2"]
    assert ranked[0][1] > ranked[1][1]


def test_set_fact_text_updates_the_keyword_index(store):
    store.insert_fact(make_fact("f-1", text="Shop opening hours 9am-5pm"))
    store.set_fact_text("f-1", "Lakeside Prints opening hours 9am-5pm")
    assert store.get_fact("f-1").text == "Lakeside Prints opening hours 9am-5pm"
    found = store.keyword_search(["lakeside"], scopes=None, valid_at=T0, limit=5)
    assert [fid for fid, _ in found] == ["f-1"]


def test_search_candidates_are_believed_visible_and_valid(store):
    store.insert_fact(make_fact("f-1", text="alpha"))
    store.insert_fact(make_fact("f-2", predicate="p2", text="beta", status=CORRECTED,
                                ended_at=T0))
    store.insert_fact(make_fact("f-3", predicate="p3", text="gamma", scope=INTERNAL))
    candidates = store.search_candidates(scopes=(PUBLIC,), valid_at=T0)
    assert [fid for fid, _ in candidates] == ["f-1"]
    assert len(candidates[0][1]) == 64  # sha256 of the text


def test_embeddings_are_cached_per_model_and_read_incrementally(store):
    vec = np.arange(4, dtype=np.float32).tobytes()
    store.put_embeddings("m1", [("h1", vec, 4), ("h2", vec, 4)])
    store.put_embeddings("m1", [("h1", vec, 4)])  # already cached: ignored
    store.put_embeddings("m2", [("h1", vec, 4)])
    rows = store.embeddings_after("m1", 0)
    assert [r[1] for r in rows] == ["h1", "h2"]
    assert store.embeddings_after("m1", rows[-1][0]) == []
    assert np.frombuffer(rows[0][3], dtype=np.float32).tolist() == [0, 1, 2, 3]
    store.delete_embeddings("h1")
    assert [r[1] for r in store.embeddings_after("m1", 0)] == ["h2"]
    assert store.embeddings_after("m2", 0) == []


def test_texts_missing_embedding_lists_only_believed_facts(store):
    store.insert_fact(make_fact("f-1", text="alpha"))
    store.insert_fact(make_fact("f-2", predicate="p2", text="alpha"))  # same text, same hash
    store.insert_fact(make_fact("f-3", predicate="p3", text="beta", status=SUPERSEDED,
                                ended_at=T0))
    missing = store.texts_missing_embedding("m1")
    assert [text for _, text in missing] == ["alpha"]
    store.put_embeddings("m1", [(missing[0][0], b"\x00" * 16, 4)])
    assert store.texts_missing_embedding("m1") == []
    assert len(store.texts_missing_embedding("m2")) == 1


def test_disclosures_round_trip_and_filter(store):
    store.insert_fact(make_fact("f-1"))
    store.insert_fact(make_fact("f-2", predicate="p2"))
    store.insert_disclosure(Disclosure("d-1", "priya", "f-1", T0, "sms"))
    store.insert_disclosure(Disclosure("d-2", "priya", "f-2", T0 + timedelta(hours=1)))
    store.insert_disclosure(Disclosure("d-3", "mateo", "f-1", T0 + timedelta(hours=2)))
    assert [d.id for d in store.disclosures(contact="priya")] == ["d-1", "d-2"]
    assert [d.id for d in store.disclosures(fact_ids=["f-1"])] == ["d-1", "d-3"]
    assert store.disclosures(contact="priya")[0] == Disclosure("d-1", "priya", "f-1", T0, "sms")


def test_delete_fact_removes_it_from_search_and_the_ledger(store):
    store.insert_fact(make_fact("f-1", text="secret budget 480"))
    store.insert_fact(make_fact("f-2", predicate="p2", value="x", superseded_by=None))
    store.end_fact("f-2", status=SUPERSEDED, ended_at=T0, superseded_by="f-1")
    store.insert_disclosure(Disclosure("d-1", "priya", "f-1", T0))
    store.delete_fact("f-1")
    assert store.get_fact("f-1") is None
    assert store.keyword_search(["secret"], scopes=None, valid_at=T0, limit=5) == []
    assert store.disclosures(contact="priya") == []
    assert store.get_fact("f-2").superseded_by is None


def test_predicates_registry(store):
    store.put_predicate("note", "many")
    store.put_predicate("likes", "many")
    store.put_predicate("likes", "one")
    assert store.predicates() == {"likes": "one", "note": "many"}
    with pytest.raises(ValueError):
        store.put_predicate("x", "several")


def test_awkward_values_round_trip_exactly(store):
    nasty = ("She said \"hi\" and used $vars, `ticks`; '); DROP TABLE edges; --\n"
             "tabs\there, emoji 🙂, RTL مرحبا, combining é, \\ backslash")
    store.insert_fact(make_fact("f-1", value=nasty, text=nasty))
    assert store.get_fact("f-1").value == nasty
    long_value = "x" * 10_000
    store.insert_fact(make_fact("f-2", predicate="p2", value=long_value, text=long_value))
    assert store.get_fact("f-2").value == long_value


def test_counts(store):
    store.insert_fact(make_fact("f-1"))
    store.insert_fact(make_fact("f-2", predicate="p2", status=CORRECTED, ended_at=T0))
    store.insert_disclosure(Disclosure("d-1", "priya", "f-1", T0))
    assert store.counts() == {"nodes": 3, "facts": 2, "active_facts": 1, "disclosures": 1,
                              "embeddings": 0, "believed_texts": 1}


def test_facts_version_changes_whenever_fact_content_changes(store):
    start = store.facts_version()
    store.insert_fact(make_fact("f-1"))
    after_insert = store.facts_version()
    store.set_fact_text("f-1", "new text")
    after_text = store.facts_version()
    store.delete_fact("f-1")
    assert start < after_insert < after_text < store.facts_version()


def test_fact_values_is_a_narrow_scan_of_every_fact(store):
    store.insert_fact(make_fact("f-1", value="9am-5pm"))
    store.insert_fact(make_fact("f-2", predicate="p2", value=None, object="priya",
                                scope=private_scope("mateo"), status=CORRECTED, ended_at=T0))
    assert sorted(store.fact_values()) == [("f-1", PUBLIC, "9am-5pm", None),
                                           ("f-2", "private:mateo", None, "priya")]


def test_texts_with_phrase_respects_scope(store):
    store.insert_fact(make_fact("f-1", text="minimum order is 500 flyers"))
    store.insert_fact(make_fact("f-2", predicate="p2", text="secret 500 flyers due",
                                scope=private_scope("priya")))
    assert store.texts_with_phrase(["500", "flyers"], scopes=(PUBLIC,)) == [
        "minimum order is 500 flyers"]
    assert sorted(store.texts_with_phrase(["500", "flyers"], scopes=None)) == [
        "minimum order is 500 flyers", "secret 500 flyers due"]
    assert store.texts_with_phrase(["flyers", "500"], scopes=(PUBLIC,)) in ([], [
        "minimum order is 500 flyers"])  # word order is checked by the caller
