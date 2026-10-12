"""Core records: nodes, facts (edges with validity intervals), disclosures and viewers.

A fact is an edge ``subject --predicate--> value`` (a literal) or ``--predicate--> object``
(another node). Two clocks describe it:

* valid time ``[valid_from, valid_to)``: when the fact holds in the world;
* transaction time ``[recorded_at, ended_at)``: when the memory believed this row.

Rows are append-only. A row's content and interval never change after it is written; the
only update a row ever receives is its *end* (a status plus ``ended_at``), so the state of
belief at any past moment can be reconstructed exactly.

A fact a contact asserted in their own private scope is that contact's *claim* (it is what
``ScopedMemory.remember`` writes for a contact). Claims get a timeline of their own within
the same subject, predicate and scope: they supersede, correct and merge only with that
contact's earlier claims, never with a fact anyone else recorded, and nobody else's writes
end them. Otherwise a customer could overwrite what the owner recorded about them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from .timeutil import ensure_aware

# Fact statuses. "Believed" rows make up the current picture of the world; the rest are
# kept only as history.
ACTIVE = "active"  # believed, and not known to have lapsed
EXPIRED = "expired"  # believed, but its validity interval has ended
SUPERSEDED = "superseded"  # replaced by a newer fact; still-true parts live on as derived rows
CORRECTED = "corrected"  # replaced entirely: it was never right as recorded
RETRACTED = "retracted"  # withdrawn with no replacement
DUPLICATE = "duplicate"  # same content as another active fact (consolidation)
MERGED = "merged"  # near-duplicate folded into a more complete fact (consolidation)

STATUSES = (ACTIVE, EXPIRED, SUPERSEDED, CORRECTED, RETRACTED, DUPLICATE, MERGED)
BELIEVED = (ACTIVE, EXPIRED)
ENDED = tuple(s for s in STATUSES if s != ACTIVE)

# Fact visibility.
PUBLIC = "public"  # anyone, including unknown senders
INTERNAL = "internal"  # the owner only
_PRIVATE_PREFIX = "private:"  # private:<contact-id>: that contact and the owner

ONE = "one"  # a new value for the same key replaces the old one
MANY = "many"  # values accumulate (notes, preferences, tags)
DEFAULT_CARDINALITY = {"note": MANY}  # every other predicate is ONE unless defined

NODE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PREDICATE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_VALUE_CHARS = 50_000


def private_scope(contact_id: str) -> str:
    return _PRIVATE_PREFIX + contact_id


def scope_audience(scope: str) -> str | None:
    return scope[len(_PRIVATE_PREFIX):] if scope.startswith(_PRIVATE_PREFIX) else None


def claimant_of(scope: str, asserted_by: str) -> str | None:
    """The contact whose own claim a fact is, or ``None`` when someone else recorded it."""
    audience = scope_audience(scope)
    return audience if audience is not None and asserted_by == audience else None


def validate_scope(scope: str) -> str:
    if scope in (PUBLIC, INTERNAL):
        return scope
    audience = scope_audience(scope)
    if audience is not None and NODE_ID.match(audience):
        return scope
    raise ValueError(f"invalid scope {scope!r}; use public, internal or private:<contact-id>")


def validate_node_id(node_id: str) -> str:
    if not NODE_ID.match(node_id):
        raise ValueError(f"invalid node id {node_id!r}; use lowercase letters, digits and dashes")
    return node_id


def normalize_predicate(raw: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if not _PREDICATE.match(name):
        raise ValueError(f"invalid predicate {raw!r}; it must start with a letter")
    return name


def normalize_value(raw: str) -> str:
    """NFC text with ``\\n`` line endings, no control characters except newline and tab."""
    text = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = text.strip()
    if not text:
        raise ValueError("a fact value cannot be empty")
    if len(text) > MAX_VALUE_CHARS:
        raise ValueError(f"a fact value is limited to {MAX_VALUE_CHARS} characters")
    return text


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: str
    created_at: datetime
    aliases: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    role: str | None = None  # "owner", "contact" or None for things

    @property
    def is_contact(self) -> bool:
        return self.role in ("owner", "contact")


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    predicate: str
    value: str | None
    object: str | None
    scope: str
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    asserted_by: str
    status: str = ACTIVE
    ended_at: datetime | None = None
    superseded_by: str | None = None
    derived_from: str | None = None
    source: str | None = None
    text: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) == (self.object is None):
            raise ValueError("a fact needs exactly one of value or object")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        ensure_aware(self.valid_from)
        ensure_aware(self.recorded_at)
        if self.valid_to is not None and ensure_aware(self.valid_to) <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from")

    @property
    def claimant(self) -> str | None:
        return claimant_of(self.scope, self.asserted_by)

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """The timeline a row belongs to. Supersession, corrections owed and consolidation
        never look beyond it, so a contact's claims and everyone else's facts about the
        same thing stay independent."""
        return (self.subject, self.predicate, self.scope, self.claimant)

    @property
    def believed(self) -> bool:
        return self.status in BELIEVED

    @property
    def audience(self) -> str | None:
        return scope_audience(self.scope)

    def valid_at(self, when: datetime) -> bool:
        return self.valid_from <= when and (self.valid_to is None or when < self.valid_to)


def node_label(node: Node) -> str:
    return f"{node.name} ({', '.join(node.aliases)})" if node.aliases else node.name


def render_fact_text(subject: Node, predicate: str, value: str | None,
                     obj: Node | None) -> str:
    """The sentence that is keyword-indexed and embedded for a fact. It is derived data:
    the vault does not store it, and renaming a node re-renders it."""
    target = value if value is not None else node_label(obj) if obj is not None else ""
    return f"{node_label(subject)} {predicate.replace('_', ' ')} {target}"


@dataclass(frozen=True)
class Disclosure:
    """One fact told to one contact at one moment: the audience ledger's unit."""

    id: str
    contact: str
    fact_id: str
    told_at: datetime
    channel: str | None = None


@dataclass(frozen=True)
class Viewer:
    """Who is asking. Built from the transport-verified sender, never from model output."""

    kind: str  # "owner", "contact" or "anonymous"
    node_id: str | None = None

    @classmethod
    def owner(cls, node_id: str | None = None) -> Viewer:
        return cls("owner", node_id)

    @classmethod
    def contact(cls, node_id: str) -> Viewer:
        if not node_id:
            raise ValueError("a contact viewer needs a node id")
        return cls("contact", node_id)

    @classmethod
    def anonymous(cls) -> Viewer:
        return cls("anonymous")

    def scopes(self) -> tuple[str, ...] | None:
        """Scopes this viewer may read; ``None`` means all of them."""
        if self.kind == "owner":
            return None
        if self.kind == "contact":
            return (PUBLIC, private_scope(self.node_id or ""))
        return (PUBLIC,)

    def can_see(self, scope: str) -> bool:
        allowed = self.scopes()
        return allowed is None or scope in allowed
