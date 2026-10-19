# R0 contracts: experimental API and storage boundaries

Design status: documented for issue #103. This document sketches the boundaries
to be validated in R1; it is **not** a claim that durable replay or network
execution already exists.

## Extensions and registration

An experiment author provides `World` and `Policy` implementations in a
separately compiled Rust application crate. `World::observe(actor)` is a pure
read-only projection, `Policy::propose(actor, observation)` is either a pure
decision or a request to resolve typed external effects, and
`World::apply(actor, action)` validates/stages the resulting transition.
The caller assigns stable actor IDs. Duplicate actor registration is an error.
A no-eligible selection returns `None` without busy-looping.

A policy registry maps stable strings to constructors and explicit config
schemas at **compile time**. The generic `hiyal` CLI loads only built-in
registrations. User binaries embed their own registry, use the same runner
library and may select custom types via configuration. TOML cannot inject new
physics or dynamically load arbitrary native Rust code.

## Proposed v1 TOML configuration (example, not yet implemented)

```toml
schema_version = 1
world = "builtin.scripted-counter"
policy = "builtin.increment"
seed = 42
trial = "reference-scripted-001"
max_committed_steps = 100
schedule = "round_robin"
[effects]
mode = "scripted" # scripted | replay | live
cache = false
max_in_flight = 8
[output]
directory = "out/reference-scripted-001"
format = "jsonl"
```

Reject unknown world/policy/scheduler identifiers and unknown schema versions
before running. A registered world validates its own parameters. Defaults,
including model sampling, are resolved and serialized in a manifest before
effects run. No arbitrary model API keys in TOML; use explicit environment or
secret integration for live providers.

## Transaction, effects and event semantics

1. Select eligible actors with trial-owned RNG; order selected actors by stable
   `AgentId` before applying completed effect results.
2. Produce immutable observations from the committed world at logical step `k`.
3. Emit requests with `(trial_id, logical_step, agent_id, call_index,
   canonical_request_hash)`. A failed/retried request reuses the *same* key
   unless the request itself changes; never append duplicated prompts to
   committed agent history.
4. Runner resolves requests asynchronously or via deterministic scripted
   results. Deadline, cancellation and unsupported-capability results are
   explicit. Accepted external charges are ledger entries independent of
   whether a later action is accepted.
5. Validate typed proposals, stage world/memory/schedule/queue/RNG changes, and
   check world invariants. Validation or commit-preparation failure leaves
   committed world/log offsets unchanged; the actor may receive a separately
   identified and logged rejection.
6. Commit a world transition with a single monotonically increasing event
   sequence. A world event contains a replayable delta or equivalent
   reconstruction operation; a proposed action alone may not suffice.
7. Invoke read-only observers **after** durable commit. Projection errors are
   diagnostics and never change the committed world. Events are streamed;
   retained metric history is explicitly bounded.

The existing `src/core.rs` provides only in-memory staged sequential commits.
Its `write_jsonl` is export of in-memory records, not crash-consistent
append-to-log. Prototype `CommittedEvent` contains an action but no world delta
and is **not yet sufficient** for generic event-sourced reconstruction.

## Durable log/checkpoint decisions for R1

- Envelope: `schema_version`, `trial_id`, `sequence`, `event_id`,
  `logical_step`, `world_type_version`, `transition`, `resulting_state_hash`.
- One single-writer committed log per trial; append a full length-delimited or
  newline-terminated validated record, flush and fsync by declared durability
  setting **before** publishing it as committed to readers.
- Handle torn final records by detecting/truncating only an uncommitted tail;
  a complete event must be idempotently replayed by `event_id`.
- Snapshot includes validated world/memory/agent history/scheduler/queue/named
  RNG states/progress, schema versions and the **next committed log offset**.
  Write temporary snapshot, fsync, atomic rename and fsync containing directory
  where supported. Reject malformed/incompatible restores without mutation.
- Resume reads a checkpoint, validates its referenced log prefix/hash and
  replays remaining committed records. Fork only from a committed boundary;
  preserve immutable ancestry and never share mutable descendants.

The final fsync policy and exact transition format must pass crash/restart
integration tests; network provider costs and tool side effects are recorded in
a separate effect ledger, not pretended transactional world state.

## Model and tool capabilities

The current `src/effects.rs` sketches role-ordered chat messages, model identity,
sampling, schemas, streaming flags, call identity and usage. The R1 runner must
supply provider effective/default parameters, explicit unsupported-capability
rejections, tracing IDs, deadlines, cost events and a separate embeddings
capability carrying model/dimension identity. Cache defaults **off** for
scientifically sampled runs; opt-in cache identity belongs in the manifest.

Recorded effects are looked up by complete logical call key (not a shared
completion-order cursor) and content hash. They are applied in deterministic
logical order even when workers finish out-of-order. D0 replay validates all
requests/content; live LLMs and nondeterministic numerical workloads are D2.
A model provider seed is a hint, never a D1 guarantee.

## External executable policy (later, not in R1)

Proposed protocol v1: newline or length-framed UTF-8 JSON, with
`protocol_version`, `trial_id`, `call_id`, message type and structured
`observation`, `requested_effects` or `proposal` fields. A policy executable
is a child process supervised by the Rust runner, with bounded stdin/stdout
frames, schema validation, per-request deadlines, cancellation and process
termination. stderr carries logs. Executables are untrusted; process
supervision alone is not a secure sandbox (OS sandbox is a separate deployment
choice). This is an interoperability boundary, **not** an embedded Python API,
MCP runtime guarantee, WASM sandbox, or permission to run arbitrary plugin
executables without user approval.

## Data and numerical contracts

See [R0 decisions](r0-decisions.md). Trial/world/agent IDs are logical and
independent of wall time. Stable event order is part of the schema. RFC 8785
canonical bytes use bounded numbers or decimal-string identifiers. For early
D1 claims, use unsigned/checked integer or explicit fixed-point model state;
floating-point reproducibility needs a named target/toolchain/numerical
profile. A numerical result containing NaN or infinity is invalid.
