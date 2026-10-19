# R1 — Native simulation engine

Hiyal now provides a Rust-only world/policy library, a built-in native CLI,
and a **separate Rust consumer crate** demonstrating user-defined research
worlds. No Python benchmarks are run by the R1 verification workflow.

## Quick start

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo test -p hiyal --features serde_json/preserve_order
cargo test -p hiyal --features live-http --no-run
cargo run -p hiyal -- run examples/counter.toml /tmp/hiyal-counter
cargo run -p hiyal -- inspect /tmp/hiyal-counter
cargo run -p hiyal -- resume /tmp/hiyal-counter 10
cargo run -p hiyal -- validate /tmp/hiyal-counter

# A separately compiled research consumer:
cargo run -p hiyal-custom-world-example -- run /tmp/hiyal-society 5
cargo run -p hiyal-custom-world-example -- resume /tmp/hiyal-society 5
cargo run -p hiyal-custom-world-example -- inspect /tmp/hiyal-society
```

The built-in CLI explicitly registers the counter world and policy. It cannot
load arbitrary Rust types from TOML; custom user-defined worlds compile their
own binaries against `hiyal::experiment` and `hiyal::storage`.

## Deterministic execution and storage

- Fixed stable `TrialId` from name/seed; stable `AgentId` decimal strings.
- Round-robin or trial-owned **ChaCha20 0.9** seeded random selection.
  Named streams have persisted exact word offsets and unbiased indices.
- One selected attempt increments the attempt sequence; only accepted actions
  increment the committed logical-step count. Rejections are logged.
- World reducers are cloned/staged, applied **before** the event is durably
  appended. Events contain the resulting state and are independently replayed
  by applying the recorded action again, comparing resulting state hashes.
- `manifest.json` contains the world genesis and RNG/schedule baseline.
  `events.jsonl` contains append-only transitions/rejections and a hash chain.
  `snapshot.json` contains versioned world/schedule/RNG state and the
  exact committed log offset. Resume verifies the complete log prefix and
  independently checks the snapshot before accepting state.
- The event append uses fsync before projecting committed state into memory.
  The snapshot writes via temp file + fsync + atomic rename (and directory
  sync on Unix). The store uses a **single-writer lock** and detects unfinished
  log tails. After a crash, a stale lock must only be cleared after verifying
  no writer is still active; never run two writers to one directory.
- Deterministic content hashing uses recursively sorted-key UTF-8 JSON in a
  pinned **R1 provisional** profile. This does **not** claim full RFC 8785
  compliance or cross-target floating-point equivalence. R1's test worlds
  use integer state and transitions.

## Models and effect replay

`EffectPolicy` creates typed model/tool requests and consumes results.
`EffectJournal` asynchronously resolves calls with bounded concurrency and
per-call deadlines. All completed receipts are fsynced to `effects.jsonl`
before the world action is accepted or rejected. Call identity binds the
trial, selected **attempt sequence**, actor, call index and request hash.
Repeat/replay looks up the whole key, validates exact content and reuses the
receipt without charging or calling a provider again. The policy receives
results in stable logical call order, not network completion order.

Backends:
- `ScriptedBackend`: deterministic offline model-shaped responses. Tools are
  explicitly unsupported without a registered implementation.
- `ReplayBackend`: **offline receipt lookup utility**, not an executable `Backend`; use `Trial::resume(..., replay_only = true)` for strict replay with the journal.
- `OpenAiCompatible` (`--features live-http`): opt-in minimal JSON chat
  request/response adapter. Supports ordered role messages, selected sampling
  settings and provider-reported model/usage. It refuses unimplemented
  streaming/tool-schema/output-schema capabilities. HTTPS is required except
  local loopback servers. Live provider generation is **D2**, not bitwise D1.

Live CLI example (set `model = true` and `model_id = "your-compatible-model"`
in the TOML; no model/API key is hard-coded):
```sh
OPENAI_COMPAT_URL=https://example.com/v1/chat/completions \
OPENAI_COMPAT_API_KEY=secret \
cargo run -p hiyal --features live-http -- run examples/counter.toml /tmp/live-trial --live
```
Do not commit credentials. Actual remote model IDs/capabilities depend on the provider; the built-in
demo defaults to an offline mock and requires explicit `model=true` plus
`--live` to contact a remote service.

## Guarantees, scope and known limits

Tests cover run/resume vs uninterrupted log hashes, random scheduling,
independent stream state, invalid snapshot rollback, timeout/rejection
behavior, observer failure isolation, keyed replay and custom-world smoke.
No Python package or interpreter is involved.

The single-process event store is not a distributed transaction engine:
external HTTP charges cannot be rolled back, and a crash after provider
completion but before its receipt fsync may cause one repeated external
request on restart. That provider idempotency gap is R3 work, not silently
considered solved. Automatic fork/sweep workflows, native CSV/Parquet,
advanced provider streaming, persistent observer projections and a language
neutral external-policy protocol belong to later slices. Do not treat a
clean CI smoke result as independent external security/provider validation.

## R1 review hardening

- Scheduler pause and resume are **explicit control entries** in the same
  hash-linked events log. They consume an attempt sequence but not a world
  step, and are replayed before actor selection. The consumer's Store exposes
  immutable world/progress views, not direct mutation.
- `effects.jsonl` receipts include content hashes over request identity,
  response, usage and effective model/provider. Corrupt or mismatched
  receipts fail strict resume/replay. The CLI `validate` command checks
  both the world log and effect ledger.
- The active receipt records the *backend actually used* (scripted vs live),
  while the effective model comes from the provider-reported response.
  Scripted responses are explicitly noncharged; live charges remain
  independently auditable even if the world rejects an action.
- Live URLs are parsed before use: HTTPS is allowed, cleartext HTTP only for
  a parsed loopback host, userinfo is forbidden, and redirects are disabled.
  The HTTP tests use a local in-process stub, not an external provider.
- Failed restores remove their writer locks. On open, abandoned
  `snapshot.tmp` files are discarded under the exclusive lock. Write,
  flush, and rename failures also clean up their temporary file.

## Review hardening: effect failures and inspection

- Live timeouts and transport/backend errors record `billing = unknown`, not `charged = false`; known scripted responses use `not_charged`, while successful live model results use `charged` as a disposition, not verified financial settlement. An effect's receipt binds to the committed trial ID. Until an external provider supplies idempotency or billing reconciliation, indeterminate charges remain an R3 limitation.
- Invalid/duplicated requests and missing replay receipts fail **before** any provider task is spawned; ledger write errors propagate without advancing the world or appending a rejected simulation step. Once calls have launched, the runner drains the entire result batch even when one journal append fails. A crashed network call before receipt durability may still require manual billing reconciliation.
- `inspect` and `validate` now verify both logs read-only, without locking, creating files or repairing torn tails; a concurrent incomplete write can produce a retryable inspection error. Mutating recovery remains exclusive to `resume`.
- Logged actions apply the reducer once in the active simulation; full reducer replay still verifies committed history on restore. Full-log replay remains deliberate until a fast snapshot-based verification profile can preserve audit guarantees.
