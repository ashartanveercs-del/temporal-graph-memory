"""Tokenization and safe keyword-query construction.

FTS5 has its own query language: quotes, parentheses, ``AND``/``OR``/``NOT``, ``NEAR()``,
``*`` prefixes, ``^`` anchors and ``column:`` filters. Passing user text straight to
``MATCH`` either raises a syntax error (one stray quote is enough) or silently changes
the meaning of the search. Queries are therefore reduced to plain word tokens, and each
token is passed as a double-quoted string, which FTS5 treats as text and never as syntax.
"""

from __future__ import annotations

import re
import unicodedata

# Mirrors FTS5's unicode61 tokenizer: runs of letters and digits; "_" separates words.
_WORD = re.compile(r"[^\W_]+")

STOPWORDS = frozenset({
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "please",
    "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
})

MAX_TERMS = 32


def tokenize(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def query_terms(text: str, *, max_terms: int = MAX_TERMS) -> list[str]:
    """Distinct search words in order of appearance, without stopwords when possible."""
    tokens = list(dict.fromkeys(tokenize(text)))
    meaningful = [t for t in tokens if t not in STOPWORDS]
    return (meaningful or tokens)[:max_terms]


def fts_expression(terms: list[str]) -> str | None:
    """OR of quoted terms: recall-friendly, and BM25 still ranks multi-term matches first."""
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def like_pattern(term: str) -> str:
    """``%term%`` with ``\\``, ``%`` and ``_`` escaped (use with ``ESCAPE '\\'``)."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
