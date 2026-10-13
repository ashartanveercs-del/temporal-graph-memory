# temporal-graph-memory

**Bitemporal memory for assistants that need to remember both facts and conversations.**
Track what was true, what was believed at the time, and which version each person was told.
A SQLite graph combines scoped hybrid retrieval, attributed customer claims and a rebuildable
Markdown vault. Embedding outages leave keyword and graph retrieval available.

[![CI](https://github.com/ashartanveercs-del/temporal-graph-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/ashartanveercs-del/temporal-graph-memory/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Quick start](#quick-start) | [Architecture](#architecture) | [Design decisions](#the-parts-that-are-harder-than-they-look) | [Verification & proof map](docs/VERIFICATION.md) | [Security boundaries](#security-notes)

- **Two clocks, one history:** interval overrides preserve past values; `known_at` reconstructs past belief. An independent randomized model checks the semantics and detects a deliberately planted bug.
- **Scope before ranking:** keyword, vector and graph candidates are filtered for the verified sender. Customer claims have separate timelines from owner facts.
- **Recovery you can inspect:** deterministic Markdown exports round-trip byte for byte; rebuilds detect stale exports and retain purge tombstones.

## Try it offline

```bash
python -m pip install -e ".[dev]"
python examples/chat_loop.py --script examples/demo_script.txt
```

No API key or model is needed. The demo uses a temporary database and vault. In this fictional
print-shop conversation, an earlier answer becomes a correction owed to the same customer:

```text
priya > Hi! What are your opening hours?
assistant > Lakeside Print Shop opening hours 9am-5pm Monday to Saturday.
priya > Sorry, what were the opening hours again?
assistant > As I mentioned earlier: Lakeside Print Shop opening hours 9am-5pm Monday to Saturday.
mateo > What budget did Priya Shah give you?
assistant > I don't have that on record; I'll check with the team.
rosa > remember shop opening_hours = 10am-6pm Monday to Saturday
assistant > Saved: Lakeside Print Shop opening hours 10am-6pm Monday to Saturday (public).
-- clock is now 2026-03-03 09:00 UTC
priya > Thanks! Do you do letterpress?
assistant > Quick correction: earlier I said "9am-5pm Monday to Saturday"; it is now "10am-6pm Monday to Saturday".
```

## What's inside

| | |
|---|---|
| **Temporal facts** | Every fact is an edge with a validity interval. A new value supersedes what it overlaps without deleting it: a change keeps the old value for the past, a correction replaces it for all time, and a temporary override (holiday hours) hands the regular value back when it ends. |
| **Bitemporal reads** | Versioned history lets `facts(as_of=T, known_at=K)` answer "what did we believe at K about time T". |
| **Audience ledger** | Records which *fact version* each contact was told. The assistant avoids repeating itself, and knows which corrections it owes ("you told Priya 9-5; it's now 10-6"). |
| **Per-sender scoping** | Facts are `public`, `internal` (owner only) or `private:<contact>`. Scopes are enforced inside every retrieval signal, on writes (contacts can't plant shared facts), along graph traversals, and in error messages. What a contact says is kept as their claim and never replaces what the owner recorded, even about them. |
| **Hybrid retrieval** | FTS5/BM25 keywords + embeddings + one-hop graph expansion, fused with reciprocal-rank fusion. A failing embedder trips a circuit breaker and search degrades to keywords with failure metadata. |
| **Markdown vault** | One page per node, one line per fact, Obsidian-compatible. Export is byte-for-byte deterministic, and the database rebuilds from the vault (a hand edit included). A rebuild refuses to undo changes made after the export, and leaves out every fact the database or the vault records as purged. |
| **Nightly consolidation** | Expires lapsed facts, folds duplicates, merges near-duplicate notes (never across audiences or authors, never a note with a negation), and backfills embeddings that failed earlier. Idempotent. |
| **LLM tools** | OpenAI-style function tools bound to the verified sender: the model never names a sender, and a hidden fact errors exactly like a missing one. |
| **Outbound guard** | Checks each reply, before it is sent, for distinctive values, or fragments of them, from facts the recipient may not see. |
| **Tested offline** | 380 tests, no internet access and no keys. They include a randomized model check of the temporal logic and a test proving that check catches a planted bug. |

## Architecture

```mermaid
flowchart LR
    M[message + verified sender] --> S["for_sender(handle)<br/>ScopedMemory"]
    S --> C["context(message)"]
    C --> K["keyword<br/>FTS5 BM25 / LIKE"]
    C --> V["vector<br/>embeddings + breaker"]
    C --> G["graph<br/>named entities, 1 hop"]
    K & V & G --> F[reciprocal-rank fusion]
    F --> P["ContextPack<br/>facts + already-told + corrections owed"]
    P --> L[LLM or template]
    L --> R{"guard(reply)"}
    R -- clean --> O[send + record_reply]
    R -- leak --> X[safe fallback]
    O --> D[(audience ledger)]
```

```mermaid
flowchart LR
    W["remember / correct / retract"] --> T["interval override<br/>(one transaction)"]
    T --> DB[("SQLite<br/>nodes, edges, disclosures,<br/>FTS5, embedding cache")]
    DB -- export --> VA["Markdown vault<br/>entities/*.md"]
    VA -- "rebuild (validated, atomic swap)" --> DB
    N["consolidate (nightly)"] --> DB
    E[embedder] -.->|best effort, outside transactions| DB
```

| Module | Responsibility |
|---|---|
| [`memory.py`](src/tgmem/memory.py) | Facade: temporal rules, contacts, scoped views, hybrid search |
| [`store.py`](src/tgmem/store.py) | `GraphStore` protocol and its SQLite implementation |
| [`audience.py`](src/tgmem/audience.py) | Repetition, corrections owed, mentions, leak guard, context pack |
| [`vault.py`](src/tgmem/vault.py) | Deterministic export, strict parser, rebuild |
| [`embeddings.py`](src/tgmem/embeddings.py) | Hashing + OpenAI-compatible embedders, circuit breaker, vector index |
| [`consolidation.py`](src/tgmem/consolidation.py) | Expire, dedupe, merge, backfill |
| [`tools.py`](src/tgmem/tools.py) | Function-calling tools bound to one sender |
| [`cli.py`](src/tgmem/cli.py) | The `tgmem` command |

### The parts that are harder than they look

**1. A new value is interval arithmetic, not an UPDATE.** When fact *N* valid over `[s, e)`
arrives, every believed fact on the same timeline (same subject, predicate and scope; a
contact's own claims get a timeline of their own, see 2) that overlaps it is ended. Whatever
part of it *N* does not cover is re-recorded as a *remainder* row with the old value
([`memory.py`](src/tgmem/memory.py)). That one rule produces the three behaviours a
business needs:

```text
change      X=[a, ∞)  N=[s, ∞), a < s   ->  X ended, remainder [a, s)          "it was X, now N"
correction  X=[a, ∞)  N=[s, ∞), s <= a  ->  X ended, no remainder              "it was never X"
override    X=[a, ∞)  N=[s, e) inside X ->  remainders [a, s) and [e, ∞)       holiday hours end by themselves
```

A row's value and validity interval never change after it is written: rows are only ever
*ended* (with a status and a time). A row was believed from `recorded_at` until `ended_at`,
so `known_at` queries reconstruct past belief exactly. The test
suite checks all of this against an independent ~30-line model of the semantics through
1,000 random operations (5 seeds × 200). It also verifies that a plausibly broken override
fails that check. One subtlety the model check found: belief "as known at K" includes
transactions stamped at exactly K. Intervals are half-open, so snapshots must key on the
last transaction time, not the wall clock.

**2. Scoping has to hold in more places than the query.** Filtering after ranking leaks
through ranks and returns short result lists. So scope filters run inside the FTS query, define
the candidate set for vector search, and restrict which relationships the graph signal may
traverse. Acme's public payment terms are not found for Mateo through a relation that is
private to Priya. Writes are scoped too. A contact who says "the shop is closed all week"
creates a fact private to that contact, so one person cannot poison what everyone else is
told. Nor can they rewrite what the owner recorded about them. The owner's facts about Priya
are private to her as well, so they share her scope, but what she states is kept as her
claim, on a timeline of its own. "My credit limit is $50,000", even backdated to 2020, never
supersedes or corrects the owner's "$500 net 30", and consolidation never merges a claim
with the owner's notes. The assistant sees the claim marked "according to Priya Shah", and
the owner settles it by retracting it. The outbound guard follows the same rule: a phrase
counts as already known to the recipient only if it appears in a public fact or one the
owner recorded for them. Otherwise a customer could save every three-digit number in a note
and so whitelist all of them. Errors are scoped: asking for a hidden fact raises the same
`NotFound` with the same message as asking for one that doesn't exist. Senders map to
viewers by exact match on a canonical handle (E.164 or lowercase email). There is no "same
last ten digits" matching, because two people can share a ten-digit tail under different
country codes.

**3. "Already told" is about values, not rows; corrections are computed, not stored.** After a
change is scheduled, today's value lives in a new remainder row, and consolidation may fold
one note into another. So "was Priya told this?" matches the ledger by *key and value*, and
follows duplicate/merge links. Corrections owed are derived at read time from the latest
disclosure per key and what is believed now. Then a scheduled change is not owed before it
takes effect, a promotion that lapsed on schedule is not "corrected", holiday hours stop
being owed once the regular hours resume, and a corrected or retracted value is owed until the
contact is told the current one.

**4. A vault that is the source of truth has to be canonical.** Rendering is a pure function
of stored rows: fixed ordering and field order, canonical UTC timestamps, backslash escaping
so any value fits on one line (newlines, braces, a leading `[[`), and bytes written with `\n`.
Text mode on Windows would silently write `\r\n`. Export twice and nothing is rewritten.
Export → rebuild → export is byte-identical, and CI checks that on Linux and Windows. The
parser is strict and reports `file:line`. It checks dangling references, cross-person
disclosures and overlapping single-valued facts before touching the database. Rebuild builds
a fresh file next to the old one and swaps it in. It deletes the old `-wal`/`-shm` files,
since SQLite would replay a stale WAL into the new database. Export stamps `index.md` with a
SHA-256 of the canonical rendering of the database (`db-digest`). If the database still
matches it, the rebuild goes ahead and applies any hand edits. Otherwise it refuses, unless
forced, as soon as the database holds anything the vault lacks or shows differently: a
retraction, a new contact, a disclosure, a renamed node. Purged fact ids are kept as
tombstones in the database and in the vault, and every rebuild, forced or not, leaves them
out, so rebuilding from an older vault can't bring an erased value back into a database that
recorded the purge. Cached embeddings are carried over so unchanged text isn't paid for
twice.

**5. Degradation is designed in.** Embedding happens after the write commits, never while
holding the write lock, and a failure leaves the fact keyword-searchable and marked pending.
A circuit breaker turns a dead provider into one timeout per cooldown instead of one per
query. User text never reaches FTS5's query language (`"`, `NEAR()`, `col:` filters) except
as quoted tokens. Builds without FTS5 fall back to `LIKE`. Vectors are cached by
`(model, sha256(text))`, so vectors from different models are never compared, and
`reindex` backfills after an outage or a model change. The offline embedder hashes with
BLAKE2b, because Python's `hash()` is salted per process.

**6. Time has two clocks, and neither may run backwards.** SQLite has no datetime type, so
instants are fixed-width UTC strings that sort like the times they represent. Writes take
the lock with `BEGIN IMMEDIATE` *before* reading the current fact: two deferred transactions
that both read and then both write fail immediately, without waiting for the busy timeout.
Each transaction gets `max(wall clock, last + 1µs)` as its time, so history stays ordered
when NTP steps the clock back. Four threads writing the same key end up with exactly one
linear history (tested).

### Why SQLite rather than a graph database

Kuzu would have been the natural embedded graph database for this. Its upstream repository
was archived in October 2025 (last release 0.11.3), and a community fork, LadybugDB,
continues it. The graph here is two tables, `nodes` and `edges`, with validity intervals,
plus one-hop traversal in SQL. SQLite brings ACID transactions, FTS5 and single-file backups,
and it's available everywhere. The temporal rules sit above a small `GraphStore` protocol, so
another backend only implements the storage primitives and runs
[`tests/test_store_contract.py`](tests/test_store_contract.py). That suite already runs
twice, against the FTS5 and the `LIKE` search paths.

## Quick start

Requirements: Python 3.11+. No API keys are needed. Set `OPENAI_API_KEY` (or another
OpenAI-compatible endpoint) only if you want model embeddings.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

export TGMEM_DB=shop.db TGMEM_VAULT=shop-vault TGMEM_EMBEDDER=hash   # offline embedder
tgmem init
tgmem add-contact "Rosa Delgado" "+1 555 000 0100" --id rosa --owner
tgmem add-contact "Priya Shah" "+1 555 000 0101" --id priya
tgmem add-contact "Mateo Ruiz" "+1 555 000 0102" --id mateo
tgmem add-entity "Lakeside Print Shop" --id shop --kind org
tgmem remember shop opening_hours "9am-5pm Monday to Saturday"
tgmem remember priya budget '$480 for wedding invitations'   # about a contact: private by default
tgmem remember shop opening_hours "closed for stocktaking" --from 2026-12-24 --to 2026-12-27

tgmem search "opening hours" --as "+1 555 000 0102"    # Mateo's view: no budget in sight
tgmem facts shop --as-of 2026-12-25                    # the temporary override
tgmem facts shop --as-of 2026-12-28                    # regular hours are back on their own
tgmem guard "+1 555 000 0102" 'Her budget is $480.'    # prints the leak, exits with 1
tgmem vault export && cat shop-vault/entities/shop.md
tgmem consolidate                                      # the nightly job (cron it)
```

`tgmem --help` lists the rest, among them `history`, `timeline`, `correct`, `retract`,
`purge`, `tell`, `corrections`, `context`, `vault check|rebuild`, `reindex` and `stats`.
Add `--json` before the command for machine-readable output.

### From Python

```python
from tgmem import HashingEmbedder, Memory

memory = Memory.open("shop.db", embedder=HashingEmbedder())  # or OpenAICompatibleEmbedder(...)
memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
memory.add_contact("Priya Shah", ["+1 555 000 0101"], id="priya")
memory.add_contact("Mateo Ruiz", ["+1 555 000 0102"], id="mateo")

hours = memory.remember("shop", "opening_hours", "9am-5pm Monday to Saturday")
memory.remember("priya", "budget", "$480 for wedding invitations")  # private to Priya

# A change: the old value stays true for the past, and the new one applies from now on.
memory.remember("shop", "opening_hours", "10am-6pm Monday to Saturday")
past = memory.facts("shop", "opening_hours", as_of=hours.valid_from)
print(past[0].value)  # 9am-5pm Monday to Saturday

# Per-sender view, built from the sender your transport verified.
priya = memory.for_sender("+1 555 000 0101")
pack = priya.context("when are you open?")  # scoped facts + corrections owed
print(pack.text)
reply = "We're open 10am-6pm Monday to Saturday."
assert not priya.guard(reply)  # contains nothing from anyone else's private facts
priya.record_reply(reply, pack)  # recorded, so it won't be repeated

print(memory.export_vault("vault").written)
memory.close()
```

### With an LLM

[`examples/llm_assistant.py`](examples/llm_assistant.py) runs one turn with any
OpenAI-compatible chat model that supports tool calling (standard library only). It puts
`pack.text` in the system prompt, offers `tool_schemas(scoped)` and routes calls through
`call_tool(scoped, ...)`. It guards the reply, falls back to a safe message on a leak or a
model error, and records what was said. Its tests drive it against a scripted local server.
In those tests the model forges a `"sender"` argument (rejected), and in another it leaks
another customer's budget (never sent).

```bash
export OPENAI_API_KEY=... LLM_MODEL=<a chat model with tool calling>
python examples/llm_assistant.py --db shop.db --sender "+1 555 000 0101" "Are you open on Saturday?"
python examples/chat_loop.py --script examples/demo_script.txt --show-context   # offline
```

### The vault

```markdown
---
id: "shop"
name: "Lakeside Print Shop"
kind: "org"
...
---

# Lakeside Print Shop

## Facts

- opening_hours: 9am-6pm  {public; valid 2026-03-03T09:02:00Z -> open; recorded 2026-03-03T09:02:00Z; by owner}  ^f-000002

## History

- opening_hours: 9am-5pm  {public; valid 2026-03-02T09:02:00Z -> open; recorded 2026-03-02T09:02:00Z; by owner; superseded 2026-03-03T09:02:00Z by f-000002}  ^f-000001
- opening_hours: 9am-5pm  {public; valid 2026-03-02T09:02:00Z -> 2026-03-03T09:02:00Z; recorded 2026-03-03T09:02:00Z; by owner; expired 2026-03-03T09:02:00Z; from f-000001}  ^f-000003
```

The trailing `^f-...` is an Obsidian block id, so the `## Told` section on a contact's page
links to the exact fact it refers to: `[[shop#^f-000002|opening_hours]]`. Front matter is JSON
values, which is valid YAML, so Obsidian shows it as properties. The exact format is pinned
by a golden test in [`tests/test_vault.py`](tests/test_vault.py).

## Configuration

Environment variables (or `tgmem --env-file .env`; see [`.env.example`](.env.example)):

| Variable | Default | |
|---|---|---|
| `TGMEM_DB` | `memory.db` | SQLite database |
| `TGMEM_VAULT` | `vault` | Markdown vault directory |
| `TGMEM_EMBEDDER` | `auto` | `auto` (OpenAI-compatible if a key is set, else none), `none`, `hash`, `openai` |
| `TGMEM_EMBED_API_KEY` | `$OPENAI_API_KEY` | key for the embeddings endpoint |
| `TGMEM_EMBED_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible `/embeddings` server |
| `TGMEM_EMBED_MODEL` | `text-embedding-3-small` | |
| `TGMEM_EMBED_DIMENSIONS` | — | shorter vectors, if the model supports it |
| `TGMEM_EMBED_TIMEOUT` | `10` | seconds per request |
| `TGMEM_HASH_DIM` | `256` | dimension of the offline hashing embedder |
| `TGMEM_DEFAULT_COUNTRY_CODE` | — | e.g. `1` or `44`, for phone numbers written without one |
| `TGMEM_TIMEZONE` | `UTC` | zone for dates typed without an offset; storage is always UTC |

`examples/llm_assistant.py` also reads `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY`
(falling back to `OPENAI_API_KEY`).

## Testing

```bash
python -m pytest -q -ra            # 380 passed in the dated local verification
ruff check src tests examples
```

- **Fakes, not mocks.** [`HashingEmbedder`](src/tgmem/embeddings.py) is deterministic
  feature hashing. [`tests/fakes.py`](tests/fakes.py) serves OpenAI-compatible
  `/embeddings` and `/chat/completions` endpoints on `127.0.0.1` from a background thread,
  so the HTTP clients are exercised over a real socket, including timeouts, 5xx responses and
  malformed payloads. A `ManualClock` and sequential ids make histories and vaults
  reproducible.
- **Model check.** [`tests/test_stress.py`](tests/test_stress.py) replays seeded random
  histories against an independent model, checking point-in-time and `known_at` reads at
  every step. A companion test plants a realistic bug in the override logic and asserts that
  the check fails.
- **Stress and regression.** The stress suite covers isolation across 300 contacts and 100
  cross-person probes (search and guard), an embedder outage in the middle of a write stream,
  four concurrent writers, a vault round trip with 200 hostile values (quotes, braces,
  backslashes, newlines, emoji, right-to-left text), byte-identical output from identical
  inputs, and idempotent consolidation over 500 near-duplicate notes.
- **Examples are tested.** Both examples run in the suite, so they can't rot.

### Performance

Historical development measurements on a laptop with 10,000 facts across 1,000 contacts
and the 256-dimension hashing embedder (hardware was not recorded; these figures were not
rerun in the [2026-10-12 verification](docs/VERIFICATION.md) and are indicative, not a performance SLA):

| Operation | Time |
|---|---|
| Bulk insert + embed 10,000 facts | ~4 s |
| Search as a contact (hybrid) | p50 8 ms, p95 15 ms |
| Search as the owner (all 10,000 facts are candidates) | p50 ~44 ms |
| Leak check per reply | 0.1 ms; ~160 ms for the first check after a write (index rebuild) |
| Vault export / rebuild | ~0.9 s / ~5 s (cached embeddings carried over; the old database is checked first) |

Vector search is exact brute force over the candidates a viewer may see. That's comfortable
up to tens of thousands of facts per assistant; beyond that, see the roadmap.

## Security notes

- **The sender has to come from your transport** (a signed webhook, an authenticated session).
  `for_sender()` trusts the handle you give it. Hand the model a `ScopedMemory`, never the
  `Memory`.
- An unrecognized or malformed sender gets the anonymous view: public facts only, no writes.
- Contacts can only write facts private to themselves, and those are kept as their claims:
  they never supersede, correct or merge with what the owner recorded, including the owner's
  facts about that contact. Only the owner creates public or internal knowledge.
- The leak guard is a backstop, not the boundary. It matches deterministically: a hidden
  value's opening phrase, its numbers with three or more digits, any three-word run of it
  holding at least two words that are not stopwords and have three or more characters ("for
  wedding invitations"), and, for a fact about a person, two such words in a row when the
  reply also names that person ("Priya lives on Elm Street"). A one-word value such as
  "matte", a two-digit number, a two-word fragment without the person's name, or a paraphrase
  is not detected. A phrase counts as already known to the recipient only if it appears in a
  public fact or one the owner recorded for them, never in text the recipient wrote. The
  boundary is retrieval-time scoping, which decides what the model ever sees.
- The database and the vault contain everything, including private facts and phone numbers.
  Protect them like any database: file permissions, encryption at rest, backup policy.
- `retract()` keeps history. `purge()` erases a value from the database, the keyword index,
  the embedding cache and the audience ledger. It keeps the erased fact ids (never the
  values) as tombstones, so a rebuild from an older vault leaves them out. The value itself
  still sits in older vault exports and backups (for example a vault kept in git), which
  have to be handled separately.
- API keys never appear in exceptions, reprs or tool errors (tested), and unexpected internal
  errors reach the model only as `"internal error"`.

## Roadmap

- A LadybugDB adapter for the `GraphStore` protocol, validated by the contract suite
- An approximate-nearest-neighbour index (sqlite-vec or HNSW) for 100k+ facts
- Incremental updates to the leak guard's phrase index instead of a rebuild after writes
- LLM-assisted fact extraction from conversations, subject to the same write scoping
- Encryption at rest for the database and the vault

## License

MIT
