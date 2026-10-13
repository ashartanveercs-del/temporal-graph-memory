# Verification and engineering proof

Verified locally on **2026-10-12**, Windows 11, Python 3.12.10, pytest 9.1.1 and Ruff 0.17.0.
This is a dated local result. The README CI badge links to the separate GitHub workflow;
the Linux/Windows and Python 3.11/3.12/3.13 matrix is not claimed as a local run.

## Reproduce

From the repository root, after installing `.[dev]` into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m ruff check src tests examples
.\.venv\Scripts\python.exe examples\chat_loop.py --script examples\demo_script.txt
```

On Linux/macOS, use `.venv/bin/python` in place of the Windows executable.

| Check | Actual result |
|---|---|
| Complete pytest suite | **380 passed**, no skips, exit 0; 40.83 s on this run |
| Ruff | **All checks passed**, exit 0 |
| Offline assistant demo | Exit 0; repeat recognition, sender isolation, correction owed, attributed private write and vault export all exercised |

The demo uses a temporary database and vault and requires no key or external service.
HTTP-client tests use loopback OpenAI-compatible servers; they exercise wire parsing and
errors without contacting a hosted model. Elapsed time is descriptive, not a performance target.

## Proof map

| Design property | Implementation | Evidence to inspect |
|---|---|---|
| Valid-time overrides and past-belief queries preserve one coherent history | [`memory.py`](../src/tgmem/memory.py), [`store.py`](../src/tgmem/store.py) | [`test_temporal.py`](../tests/test_temporal.py) covers changes, corrections and temporary overrides; [`test_stress.py`](../tests/test_stress.py) compares 1,000 seeded operations with an independent model and verifies that a planted override defect fails the check |
| Scope applies before ranking and to writes, not just rendered answers | [`memory.py`](../src/tgmem/memory.py), [`retrieval.py`](../src/tgmem/retrieval.py), [`tools.py`](../src/tgmem/tools.py) | [`test_scoping.py`](../tests/test_scoping.py) covers claim poisoning, hidden/missing error parity and private writes; [`test_retrieval.py`](../tests/test_retrieval.py) probes every retrieval signal |
| Outdated disclosures are derived from the current value and the audience ledger | [`audience.py`](../src/tgmem/audience.py) | [`test_audience.py`](../tests/test_audience.py) covers scheduled changes, temporary overrides, repeat windows and malicious guard whitelisting |
| Vault export is canonical; rebuild rejects stale state and retains erasure tombstones | [`vault.py`](../src/tgmem/vault.py) | [`test_vault.py`](../tests/test_vault.py) checks LF-only output, byte-identical round trips, post-export mutations, stale WAL recovery and purged facts |
| Embedding failures preserve useful retrieval; another model's vectors are excluded | [`embeddings.py`](../src/tgmem/embeddings.py), [`memory.py`](../src/tgmem/memory.py) | [`test_retrieval.py`](../tests/test_retrieval.py), [`test_embeddings.py`](../tests/test_embeddings.py) cover outage, cooldown, backfill, model separation and malformed responses |
| Storage semantics survive both FTS5 and LIKE search paths | [`store.py`](../src/tgmem/store.py) | [`test_store_contract.py`](../tests/test_store_contract.py) runs the same contract against both paths; concurrent-writer histories are checked in [`test_stress.py`](../tests/test_stress.py) |

## Review changes

The first-screen README now links each major entry point and provides the real offline demo.
The CI workflow runs that demo explicitly on its OS/Python matrix and grants its token
read-only repository permissions. No runtime defect was found in this review; existing
regressions and their substantive documentation were retained.

## What this does not prove

- The caller must authenticate the sender. `for_sender()` trusts the supplied handle; the
  library does not verify webhooks or sessions itself.
- The outbound guard is a deterministic phrase matcher. It cannot detect every paraphrase;
  retrieval scoping is the primary data boundary. See [security notes](../README.md#security-notes).
- The SQLite database and vault are plaintext. Purging live state cannot erase older exports
  or external backups.
- Randomized model checks sample many histories; they are not a formal proof for all histories.
  A hashing embedder and scripted servers do not measure real-model answer quality.
- The README performance table is an older development measurement, not a benchmark rerun
  here. No throughput, latency or deployment-scale claim follows from this verification.

## Public-release audit

Reviewed the 47 tracked files and 62 unique historical text blobs reachable from the two
existing commits. Enumeration used `git ls-files`, `git rev-list --objects --all` and
`git cat-file --batch`; commit attribution was reviewed with
`git log --all --format="%h %an <%ae> | %s"`.

A pattern scan plus source/example review found no credential markers, private infrastructure
or workspace identifiers, or non-example email addresses in file contents. Contacts in the
fixtures and demo are fictional `555` numbers and `example.com` addresses. Maintainer name
and email are present in Git authorship metadata and will be public with the history.
No release blocker was identified and no history was rewritten. This is a source/history
review, not a guarantee that a heuristic scan detects every possible secret.
