# R4 — native memory, privacy-safe observation and portable interfaces

Hiyal remains a Rust-only library and CLI. No Python bindings, native FAISS,
Hugging Face, PyTorch, OpenTelemetry, or remote-memory service is required.

## Memory

`memory::MemoryBank` provides bounded agent-scoped structured JSON records.
Applications explicitly embed a bank in their `World` if memory should
participate in deterministic snapshots, forks and committed events. Hiyal does
**not** introduce global mutable memory or inject hidden observations. Memory
is not yet wired into the built-in counter example.

- `MemoryBank::new(capacity, max_payload_bytes)`: 1–10,000 records and
  1–65,536 serialized payload bytes per record.
- `put(agent, key, payload, logical_step, importance, ttl)`: validated upsert,
  logical-step TTL, importance 0–1,000 and deterministic victim selection.
- `get`, `recall_prefix`, `reinforce`, `prune`, `validate`: exact recall,
  deterministic ranking, adaptive usage and explicit validation after restore.
- Eviction is deterministic: lowest importance, oldest access step, then
  lexical record ID. Equal-priority recall uses access count, recency and key.
- No semantic embeddings/vector index: there is no demonstrated R4 workload
  justifying its cost. Users can supply explicit search policies later.
- Application data in memory is sensitive and **will be included** in world
  snapshots and authoritative event logs when the world serializes it. Encrypt
  or restrict access to the trial directory; do not export those files publicly.

## Observation

The authoritative `events.jsonl` and receipt ledger remain unchanged.
`observe::JsonlObserver::recover(dir)` first verifies the history then builds
a replaceable `observations.jsonl` projection. The built-in CLI now attaches
it on run and resume (including paired fork arms). The observer is isolated:
a projection-write failure is diagnostic, not a rollback of a committed world
event. Recovery is idempotent and does not require a history-sized vector.

Each redacted observation has `schema: 1`, trial ID, decimal-string
`sequence` and `logical_step`, decimal-string actor ID, kind
(`accepted`, `rejected` or `control`), event hash and optional state hash.
No world snapshots, actions, rejection reasons, model prompts, completions,
tool arguments, raw receipt data or user-defined memory payloads are copied
to the redacted projection. IDs and hashes can still correlate records and
are not a guarantee of anonymity.

```sh
cargo run --locked -p hiyal -- run examples/counter.toml /tmp/hiyal-r4
cargo run --locked -p hiyal -- export /tmp/hiyal-r4 /tmp/hiyal-summary.jsonl
```

The CLI export validates source state and effect receipts before creating a
new destination, then writes a stream of redacted summaries. It is intentionally
restricted to the built-in registered counter world. Separately compiled
custom-world binaries can use the public `EventSummary::from_entry` API.

## Contracts and release boundaries

- Schema `storage::SCHEMA=1` continues to govern the manifest, log and
  snapshot; the projection has its independent `OBSERVATION_SCHEMA=1`.
  The memory snapshot shape uses `MEMORY_SCHEMA=1`.
- The committed log is append-only, fsynced and hash-chained; snapshots
  identify the byte offset. Read-only validation replays the chain rather
  than trusting a snapshot. Event JSON key order is recursively sorted;
  it is *not* RFC 8785. u64 event identifiers are decimal strings.
- Storage is not encrypted at rest. Model/tool content in authoritative
  files must be protected by filesystem permissions, separate from redacted
  observability. The engine does not automatically scrub application payloads.
- Limits on memory records and exported payload apply at memory API boundary.
  Trial retention and disk quotas remain application/deployment controls; the
  engine does not impose a global log-byte cap or safely delete history.
- Native artifacts target Linux, macOS and Windows. The opt-in distribution
  workflow runs on version tags or manual dispatch; ordinary PRs use the
  existing budget-conscious Linux smoke workflow. Publication to a package
  registry is not yet enabled.

## Remaining verification

R4's native SDK and schema implementations need full GitHub CI and the
cross-platform release run before the milestone can be checked as complete.
Fuzz/property testing, adversarial malformed large files and reproducible
release checksum attestation belong to the R5 verification gate. The R0
historical Python baseline measurement remains deliberately unscheduled.
