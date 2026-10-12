import os
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pytest
from fakes import T0, FakeEmbeddingServer

from tgmem.embeddings import (
    CircuitBreaker,
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    VectorIndex,
)
from tgmem.store import SQLiteGraphStore
from tgmem.timeutil import ManualClock

API_KEY = "test-key-not-real-123456"


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# -- hashing embedder ---------------------------------------------------------------------

def test_hashing_embedder_is_deterministic_and_unit_length():
    emb = HashingEmbedder(64)
    first, second = emb.embed(["Invoice 1042 is due Friday"] * 2)
    assert first == second
    assert len(first) == 64
    assert abs(np.linalg.norm(first) - 1.0) < 1e-9
    assert emb.model_id == "hashing-v1-64"


def test_hashing_embedder_is_identical_across_processes():
    # Python's built-in hash() is salted per process; a fake embedder built on it would
    # produce different vectors on every run and quietly break cached embeddings.
    script = ("import json; from tgmem.embeddings import HashingEmbedder; "
              "print(json.dumps(HashingEmbedder(16).embed(['matte business cards'])[0]))")
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                         text=True, check=True).stdout
    import json
    assert json.loads(out) == HashingEmbedder(16).embed(["matte business cards"])[0]


def test_hashing_embedder_reflects_word_overlap():
    emb = HashingEmbedder(256)
    query, close, far = emb.embed(["when is the invoice due",
                                   "the invoice for order 1042 is due on Friday",
                                   "we print on linen card stock"])
    assert cosine(query, close) > cosine(query, far)


def test_hashing_embedder_aliases_simulate_synonyms():
    plain = HashingEmbedder(64)
    synonyms = HashingEmbedder(64, aliases={"bill": "invoice", "bills": "invoice"})
    assert cosine(*plain.embed(["bill", "invoice"])) < 0.5
    assert cosine(*synonyms.embed(["bill", "invoice"])) == pytest.approx(1.0)
    assert synonyms.model_id != plain.model_id  # different vectors, different cache key


def test_hashing_embedder_handles_text_without_words():
    vector = HashingEmbedder(8).embed(["?!"])[0]
    assert vector == [0.0] * 8


# -- circuit breaker ----------------------------------------------------------------------

def test_circuit_breaker_opens_after_repeated_failures_and_probes_after_cooldown():
    clock = ManualClock(T0)
    breaker = CircuitBreaker(failure_threshold=3, reset_after=timedelta(seconds=30),
                             clock=clock)
    for _ in range(2):
        assert breaker.allow()
        breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()
    clock.advance(seconds=31)
    assert breaker.allow()  # one trial call
    assert breaker.state == "half-open"
    assert not breaker.allow()  # only one trial at a time
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(seconds=31)
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow()


def test_success_resets_the_failure_count():
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"


# -- OpenAI-compatible embedder -------------------------------------------------------------

def test_openai_compatible_request_and_response():
    with FakeEmbeddingServer(dim=8) as server:
        emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url=server.url,
                                       model="text-embedding-3-small", dimensions=8)
        vectors = emb.embed(["alpha", "beta"])
    [request] = server.requests
    assert request["path"] == "/v1/embeddings"
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["json"] == {"model": "text-embedding-3-small", "input": ["alpha", "beta"],
                               "dimensions": 8}
    # the server answered in reverse order; results follow the input order
    assert vectors == HashingEmbedder(8).embed(["alpha", "beta"])


def test_openai_compatible_batches_requests():
    with FakeEmbeddingServer() as server:
        emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url=server.url, batch_size=2)
        assert len(emb.embed([f"text {i}" for i in range(5)])) == 5
    assert [len(r["json"]["input"]) for r in server.requests] == [2, 2, 1]
    assert "dimensions" not in server.requests[0]["json"]


def test_model_id_names_host_model_and_dimensions():
    emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url="https://api.example.com/v1",
                                   model="embed-small", dimensions=256)
    assert emb.model_id == "api.example.com/embed-small@256"


@pytest.mark.parametrize(
    "server_kwargs",
    [
        {"status": 500},
        {"status": 401},
        {"body": b"not json"},
        {"body": b'{"data": []}'},
        {"body": b'{"data": [{"index": 0, "embedding": [1, 2]}, '
                 b'{"index": 1, "embedding": [1, 2, 3]}]}'},
        {"body": b'{"data": [{"index": 0, "embedding": [1, "x"]}, '
                 b'{"index": 1, "embedding": [1, 2]}]}'},
    ],
    ids=["http-500", "http-401", "not-json", "missing-items", "ragged", "non-numeric"],
)
def test_bad_responses_raise_embedding_error_without_leaking_the_key(server_kwargs):
    with FakeEmbeddingServer(**server_kwargs) as server:
        emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url=server.url)
        with pytest.raises(EmbeddingError) as err:
            emb.embed(["alpha", "beta"])
    assert API_KEY not in str(err.value)


def test_timeouts_become_embedding_errors():
    with FakeEmbeddingServer(delay=0.5) as server:
        emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url=server.url, timeout=0.1)
        with pytest.raises(EmbeddingError, match="timed out|timeout"):
            emb.embed(["alpha"])


def test_unreachable_endpoints_become_embedding_errors():
    with FakeEmbeddingServer() as server:
        url = server.url
    emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url=url, timeout=0.3)
    with pytest.raises(EmbeddingError):
        emb.embed(["alpha"])


# -- vector index ---------------------------------------------------------------------------

def test_vector_index_ranks_only_the_given_candidates(tmp_path):
    store = SQLiteGraphStore(tmp_path / "m.db")
    emb = HashingEmbedder(256)
    index = VectorIndex(store, emb.model_id)
    texts = {"h-invoice": "invoice due friday", "h-cards": "matte business cards",
             "h-secret": "invoice due friday"}  # same text as h-invoice, but not a candidate
    index.add_texts({h: v for h, v in zip(texts, emb.embed(list(texts.values())), strict=True)})
    query = emb.embed(["when is the invoice due"])[0]
    ranked = index.top(query, [("f-1", "h-cards"), ("f-2", "h-invoice"), ("f-3", "h-missing")],
                       limit=5)
    assert ranked[0][0] == "f-2"
    assert {fid for fid, _ in ranked} <= {"f-1", "f-2"}  # only candidates with vectors
    assert [score for _, score in ranked] == sorted((s for _, s in ranked), reverse=True)
    assert all(score > 0 for _, score in ranked)
    store.close()


def test_vector_index_breaks_ties_by_fact_id(tmp_path):
    store = SQLiteGraphStore(tmp_path / "m.db")
    emb = HashingEmbedder(32)
    index = VectorIndex(store, emb.model_id)
    index.add_texts({"h": emb.embed(["same words"])[0]})
    ranked = index.top(emb.embed(["same words"])[0], [("f-9", "h"), ("f-2", "h"), ("f-5", "h")],
                       limit=3)
    assert [fid for fid, _ in ranked] == ["f-2", "f-5", "f-9"]
    store.close()


def test_vector_index_loads_new_rows_written_by_other_processes(tmp_path):
    path = tmp_path / "m.db"
    reader, writer = SQLiteGraphStore(path), SQLiteGraphStore(path)
    emb = HashingEmbedder(16)
    index = VectorIndex(reader, emb.model_id)
    query = emb.embed(["alpha beta"])[0]
    assert index.top(query, [("f-1", "h1")], limit=1) == []
    vector = np.asarray(emb.embed(["alpha beta"])[0], dtype=np.float32)
    writer.put_embeddings(emb.model_id, [("h1", vector.tobytes(), 16)])
    writer.put_embeddings("some-other-model", [("h1", vector.tobytes()[:32], 8)])
    assert [fid for fid, _ in index.top(query, [("f-1", "h1")], limit=1)] == ["f-1"]
    reader.close()
    writer.close()


def test_vector_index_ignores_vectors_of_a_different_dimension(tmp_path):
    store = SQLiteGraphStore(tmp_path / "m.db")
    index = VectorIndex(store, "model")
    index.add_texts({"h1": [1.0, 0.0, 0.0], "h2": [1.0, 0.0]})
    ranked = index.top([1.0, 0.0, 0.0], [("f-1", "h1"), ("f-2", "h2")], limit=5)
    assert [fid for fid, _ in ranked] == ["f-1"]
    store.close()


def test_ties_at_the_cutoff_are_still_broken_by_fact_id(tmp_path):
    store = SQLiteGraphStore(tmp_path / "m.db")
    emb = HashingEmbedder(32)
    index = VectorIndex(store, emb.model_id)
    index.add_texts({"same": emb.embed(["identical words"])[0],
                     "other": emb.embed(["completely unrelated"])[0]})
    candidates = [(f"f-{i:03d}", "same") for i in range(40, 0, -1)] + [("f-500", "other")]
    ranked = index.top(emb.embed(["identical words"])[0], candidates, limit=5)
    assert [fid for fid, _ in ranked] == ["f-001", "f-002", "f-003", "f-004", "f-005"]
    store.close()


def test_forgotten_vectors_are_not_returned(tmp_path):
    store = SQLiteGraphStore(tmp_path / "m.db")
    emb = HashingEmbedder(32)
    index = VectorIndex(store, emb.model_id)
    index.add_texts({"h1": emb.embed(["alpha"])[0]})
    index.forget({"h1"})
    assert index.top(emb.embed(["alpha"])[0], [("f-1", "h1")], limit=3) == []
    store.close()


def test_keys_never_appear_in_reprs_or_degraded_search_messages(tmp_path):
    from tgmem.config import Settings
    from tgmem.memory import Memory

    emb = OpenAICompatibleEmbedder(api_key=API_KEY, base_url="http://127.0.0.1:9/v1")
    assert API_KEY not in repr(emb)
    assert API_KEY not in repr(Settings.from_env({"OPENAI_API_KEY": API_KEY}))
    with FakeEmbeddingServer(status=401) as server:
        memory = Memory.open(tmp_path / "m.db", embedder=OpenAICompatibleEmbedder(
            api_key=API_KEY, base_url=server.url))
        result = memory.search("anything")
        memory.close()
    assert result.degraded and API_KEY not in result.degraded
