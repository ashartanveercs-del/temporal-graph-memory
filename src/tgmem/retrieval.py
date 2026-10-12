"""Result types, reciprocal-rank fusion and entity-mention matching for hybrid search.

Three signals rank candidate facts independently:

* keyword: FTS5 BM25 (or the LIKE fallback), via the store;
* vector: cosine similarity from the embedding index, when the embedder is healthy;
* graph: facts about entities named in the query, then facts one relationship away.

Reciprocal-rank fusion (``sum 1 / (60 + rank)``) merges them. It only looks at ranks, so
BM25 scores and cosines never have to be put on the same scale, and a signal that is
missing (embedder down) simply contributes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .model import Fact, Node
from .textindex import STOPWORDS, tokenize

RRF_K = 60
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})


@dataclass(frozen=True)
class Hit:
    fact: Fact
    score: float
    signals: dict[str, int] = field(default_factory=dict)  # signal -> 1-based rank


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Hit, ...]
    mode: str  # "hybrid" when vectors contributed, else "keyword"
    degraded: str | None = None  # why vector search was skipped, if it should have run

    @property
    def facts(self) -> list[Fact]:
        return [hit.fact for hit in self.hits]


def rrf(ranked: Mapping[str, Sequence[str]], k: int = RRF_K) -> list[tuple[str, float, dict]]:
    """Fuse ranked id lists. Ties are broken by id so results are fully deterministic."""
    scores: dict[str, float] = {}
    signals: dict[str, dict[str, int]] = {}
    for name, ids in ranked.items():
        for rank, item in enumerate(ids, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            signals.setdefault(item, {})[name] = rank
    fused = [(item, round(score, 9), signals[item]) for item, score in scores.items()]
    return sorted(fused, key=lambda row: (-row[1], row[0]))


class EntityMatcher:
    """Finds nodes named in a piece of text by whole-word matching of names and aliases.

    People are also matched by their first name ("Priya" -> "Priya Shah"). The index is
    rebuilt only when the store's node version changes.
    """

    def __init__(self) -> None:
        self._version = -1
        self._by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}

    def refresh(self, nodes_version: int, load_nodes: Callable[[], Sequence[Node]]) -> None:
        if nodes_version == self._version:
            return
        index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for node in load_nodes():
            names = [tuple(tokenize(name)) for name in (node.name, *node.aliases)]
            if node.kind == "person" and names and names[0]:
                first = names[0][:1]
                if first[0] not in STOPWORDS and len(first[0]) >= 3:
                    names.append(first)
            for tokens in set(names):
                if tokens:
                    index.setdefault(tokens[0], []).append((tokens, node.id))
        for entries in index.values():
            entries.sort(key=lambda e: (-len(e[0]), e[1]))
        self._by_first, self._version = index, nodes_version

    def mentions(self, text: str) -> list[str]:
        tokens = tokenize(text)
        found: list[str] = []
        for i, token in enumerate(tokens):
            for name, node_id in self._by_first.get(token, ()):
                if tuple(tokens[i:i + len(name)]) == name and node_id not in found:
                    found.append(node_id)
        return found


def mentions_self(text: str) -> bool:
    return any(token in FIRST_PERSON for token in tokenize(text))
