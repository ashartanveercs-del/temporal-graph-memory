"""temporal-graph-memory: long-term memory for AI assistants.

A temporal knowledge graph on SQLite: facts with validity intervals that supersede each
other without losing history, an audience ledger of what each contact was told, a
Markdown vault the graph rebuilds from, per-sender scoping, and hybrid retrieval that
degrades to keyword search when embeddings are unavailable.
"""

from .audience import ContextPack, Correction
from .config import Settings
from .consolidation import ConsolidationReport
from .embeddings import (
    CircuitBreaker,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
)
from .memory import Memory, NotFound, ReindexReport, ScopedMemory, ScopeError
from .model import INTERNAL, PUBLIC, Disclosure, Fact, Node, Viewer, private_scope
from .retrieval import Hit, SearchResult
from .store import GraphStore, HandleConflict, SQLiteGraphStore
from .tools import call_tool, tool_schemas
from .vault import VaultError, read_vault, rebuild_from_vault

__version__ = "0.1.0"

__all__ = [
    "INTERNAL",
    "PUBLIC",
    "CircuitBreaker",
    "ConsolidationReport",
    "ContextPack",
    "Correction",
    "Disclosure",
    "Embedder",
    "EmbeddingError",
    "Fact",
    "GraphStore",
    "HandleConflict",
    "HashingEmbedder",
    "Hit",
    "Memory",
    "Node",
    "NotFound",
    "OpenAICompatibleEmbedder",
    "ReindexReport",
    "SQLiteGraphStore",
    "ScopeError",
    "ScopedMemory",
    "SearchResult",
    "Settings",
    "VaultError",
    "Viewer",
    "__version__",
    "call_tool",
    "private_scope",
    "read_vault",
    "rebuild_from_vault",
    "tool_schemas",
]
