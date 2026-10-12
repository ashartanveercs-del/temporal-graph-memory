"""Embedders, a circuit breaker for them, and the in-memory vector index.

Embeddings are an optimisation, never a dependency. A provider outage, a missing key or a
malformed response raises ``EmbeddingError``; the memory layer catches it, keeps the fact
keyword-searchable, and backfills the vector later. Vectors are cached by
``(model_id, sha256(text))``, so a rebuild or a rename never pays to embed the same text
twice, and vectors from different models are never compared with each other.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from .textindex import STOPWORDS, tokenize
from .timeutil import Clock, SystemClock

log = logging.getLogger("tgmem")

Transport = Callable[[str, dict, dict, float], Any]


class EmbeddingError(RuntimeError):
    """The embedder could not produce vectors. Messages never contain credentials."""


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder: no network, no model, no randomness.

    Each content word (lowercased, stopwords dropped, a trailing plural "s" removed) is
    hashed with BLAKE2b to a signed bucket. Python's ``hash()`` would be simpler, but it is
    salted per process, so vectors would change between runs. The result captures word
    overlap, not meaning; ``aliases`` maps words onto each other to simulate synonyms in
    tests ("bill" -> "invoice").
    """

    def __init__(self, dim: int = 256, *, aliases: Mapping[str, str] | None = None) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.aliases = {k.casefold(): v.casefold() for k, v in (aliases or {}).items()}
        suffix = ""
        if self.aliases:
            digest = json.dumps(sorted(self.aliases.items())).encode()
            suffix = "-" + hashlib.blake2b(digest, digest_size=4).hexdigest()
        self.model_id = f"hashing-v1-{dim}{suffix}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float64)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dim] += 1.0 if (value >> 32) & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm else vector.tolist()

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        content = [t for t in tokens if t not in STOPWORDS] or tokens
        return [self._canonical(t) for t in content]

    def _canonical(self, token: str) -> str:
        if token in self.aliases:
            return self.aliases[token]
        stem = token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith(
            "ss") else token
        return self.aliases.get(stem, stem)


class OpenAICompatibleEmbedder:
    """``POST {base_url}/embeddings`` in the OpenAI format, which many providers and local
    servers also speak. Uses only the standard library; ``transport`` is injectable."""

    def __init__(self, *, api_key: str, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1", dimensions: int | None = None,
                 timeout: float = 10.0, batch_size: int = 64,
                 transport: Transport | None = None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.dimensions = dimensions
        self.timeout = timeout
        self.batch_size = batch_size
        self._transport = transport or post_json
        host = urlparse(base_url).netloc or base_url
        self.model_id = f"{host}/{model}" + (f"@{dimensions}" if dimensions else "")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            payload: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                payload["dimensions"] = self.dimensions
            body = self._transport(self.url, payload, headers, self.timeout)
            vectors.extend(_parse_embeddings(body, len(batch)))
        if len({len(v) for v in vectors}) > 1:
            raise EmbeddingError("embedding response mixes vector dimensions")
        return vectors

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleEmbedder(model_id={self.model_id!r})"


def post_json(url: str, payload: dict, headers: dict, timeout: float) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise EmbeddingError(f"embedding request failed with HTTP {exc.code}") from None
    except TimeoutError:
        raise EmbeddingError(f"embedding request timed out after {timeout}s") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise EmbeddingError(f"embedding request timed out after {timeout}s") from None
        raise EmbeddingError(f"embedding endpoint unreachable: {exc.reason}") from None
    except OSError as exc:
        raise EmbeddingError(f"embedding endpoint unreachable: {exc.strerror}") from None
    try:
        return json.loads(raw)
    except ValueError:
        raise EmbeddingError("embedding response is not valid JSON") from None


def _parse_embeddings(body: Any, expected: int) -> list[list[float]]:
    try:
        items = sorted(body["data"], key=lambda item: item["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in items]
    except (KeyError, TypeError, ValueError):
        raise EmbeddingError("malformed embedding response") from None
    if len(vectors) != expected:
        raise EmbeddingError(f"expected {expected} embeddings, got {len(vectors)}")
    if len({len(v) for v in vectors}) != 1 or not vectors[0]:
        raise EmbeddingError("embedding response mixes vector dimensions")
    return vectors


class CircuitBreaker:
    """Stops calling a failing embedder for a while, so a dead provider costs one timeout
    per cooldown period instead of one per query.

    closed -> (``failure_threshold`` consecutive failures) -> open -> (``reset_after``)
    -> half-open: one trial call; success closes, failure re-opens.
    """

    def __init__(self, *, failure_threshold: int = 3,
                 reset_after: timedelta = timedelta(seconds=30),
                 clock: Clock | None = None) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self.clock = clock or SystemClock()
        self.state = "closed"
        self._failures = 0
        self._opened_at = None

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and self.clock.now() - self._opened_at >= self.reset_after:
            self.state = "half-open"
            return True
        return False

    def record_success(self) -> None:
        self.state = "closed"
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self.state == "half-open" or self._failures >= self.failure_threshold:
            self.state = "open"
            self._opened_at = self.clock.now()


class VectorIndex:
    """Normalized vectors for one model, keyed by text hash, loaded incrementally from
    the store's embedding cache (so rows written by other processes show up too).

    Vectors live in one float32 matrix per dimension; new rows are buffered and stacked
    lazily. Search is exact (brute force) over the candidate set the caller passes in.
    Candidates are already filtered by scope and validity, so the index never decides
    visibility.
    """

    def __init__(self, store: Any, model_id: str) -> None:
        self.store = store
        self.model_id = model_id
        self._rows: dict[str, tuple[int, int]] = {}  # text hash -> (dim, row)
        self._matrices: dict[int, np.ndarray] = {}
        self._pending: dict[int, list[np.ndarray]] = {}
        self._last_seq = 0

    def refresh(self) -> None:
        for seq, digest, dim, blob in self.store.embeddings_after(self.model_id,
                                                                  self._last_seq):
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape == (dim,):
                self._add(digest, vector)
            self._last_seq = max(self._last_seq, seq)

    def add_texts(self, vectors: Mapping[str, Sequence[float]]) -> None:
        for digest, values in vectors.items():
            normalized = _normalize(values)
            if normalized is not None:
                self._add(digest, normalized)

    def has(self, digest: str) -> bool:
        return digest in self._rows

    def forget(self, digests: set[str]) -> None:
        for digest in digests:
            self._rows.pop(digest, None)  # the matrix row becomes unreachable

    def _add(self, digest: str, vector: np.ndarray) -> None:
        dim = vector.shape[0]
        if self._rows.get(digest, (None,))[0] == dim:
            return  # same text, same model: the vector is already here
        pending = self._pending.setdefault(dim, [])
        stacked = self._matrices.get(dim)
        row = (0 if stacked is None else stacked.shape[0]) + len(pending)
        pending.append(vector.astype(np.float32))
        self._rows[digest] = (dim, row)

    def _matrix(self, dim: int) -> np.ndarray | None:
        pending = self._pending.get(dim)
        if pending:
            base = self._matrices.get(dim)
            self._matrices[dim] = np.vstack(([base] if base is not None else []) + pending)
            self._pending[dim] = []
        return self._matrices.get(dim)

    def top(self, query: Sequence[float], candidates: Sequence[tuple[str, str]],
            limit: int) -> list[tuple[str, float]]:
        """``(fact_id, cosine)`` for the best candidates, highest first, ties broken by
        fact id. Scores are computed once per distinct text (so facts with identical text
        tie exactly) and rounded to six decimals so ordering does not depend on the last
        bits of floating-point arithmetic."""
        self.refresh()
        q = _normalize(query)
        if q is None:
            return []
        matrix = self._matrix(q.shape[0])
        if matrix is None:
            return []
        fact_ids, rows = [], []
        for fact_id, digest in candidates:
            entry = self._rows.get(digest)
            if entry is not None and entry[0] == q.shape[0]:
                fact_ids.append(fact_id)
                rows.append(entry[1])
        if not rows:
            return []
        unique_rows, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
        scores = np.round((matrix[unique_rows] @ q).astype(np.float64), 6)[inverse]
        keep = np.flatnonzero(scores > 0)
        if keep.size > limit:
            # The limit-th best score; everything tied with it stays in, so the final
            # fact-id tie-break is applied to all of them, not to an arbitrary subset.
            cutoff = np.partition(scores[keep], keep.size - limit)[keep.size - limit]
            keep = keep[scores[keep] >= cutoff]
        ranked = sorted(((float(scores[i]), fact_ids[i]) for i in keep),
                        key=lambda pair: (-pair[0], pair[1]))
        return [(fact_id, score) for score, fact_id in ranked[:limit]]


def _normalize(values: Sequence[float]) -> np.ndarray | None:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


def to_blob(values: Sequence[float]) -> tuple[bytes, int]:
    """Normalized float32 bytes for the embedding cache, plus the dimension."""
    vector = _normalize(values)
    if vector is None:
        vector = np.zeros(len(values), dtype=np.float32)
    return vector.astype(np.float32).tobytes(), len(values)
