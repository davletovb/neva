# Issue #103 — R0 decision record and R1 starter boundary

Status: **R0 in progress**. This is a new Rust engine informed by the archived
Python implementation, *not* a line-by-line translation. No Python bindings,
embedded Python runtime, or required Python analysis stack. The old package
lives under `hiyal-python/` for benchmarks and fixtures only.

## Naming and scope

- Project and repository: **Hiyal**. Arabic *ḥiyal*, used for ingenious
  devices/automata, also meaning tricks; pronounced approximately "hee-YAHL".
- Root Rust library package: `hiyal`; package publishing is deliberately
  disabled until name/ownership/legal checks and R0 acceptance. The old Python
  package having the same name is an archive, not a dependency.
- GitHub handle, domain and trademark checks, crates.io reservation and
  additional crate names: **unverified/open**; do not claim reserved names.
- One initial crate plus a separate application crate. Split into more crates
  only with a demonstrated API/compilation boundary. Native CLI is planned, not
  present in this first commit.

## Extension and execution model

An application implements public Rust `World` and `Policy` traits and
compiles a consumer binary; `examples/custom_world` demonstrates this without
touching library source. The eventual native generic CLI can select only
built-in or compiled-in registered worlds; TOML selects predefined models
and parameters, not arbitrary Rust code. External executable policies may
follow after the native vertical slice via an explicitly versioned, framed,
schema-validated JSON protocol with timeouts, cancellation and process
supervision. No implicit MCP sandbox or WASM guarantee.

Target decision sequence:

1. Scheduler selects an eligible actor; no actor ends a step without an event.
2. World produces an immutable observation.
3. Policy may request typed model/tool effects; runner performs asynchronous
   bounded I/O *outside* the deterministic world reducer.
4. Effect results are resolved into a proposed typed action and validated.
5. Stage the transition; commit all world changes plus one event in defined
   logical order, or commit **nothing** on rejection.
6. Observer consumes the committed event. Its failure cannot roll back or
   mutate the world; it may produce an independently recorded diagnostic.
7. Charged/irreversible effects have a separate ledger even when a proposal
   is rejected. Rejection feedback is explicit and logged, not silently
   appended to an agent conversation on retry.

**Implemented in this initial slice:** a pure synchronous scripted policy,
typed world, round-robin selection, deterministic trial/event IDs, staged
world/scheduler commits and in-memory committed records. JSONL export is
prototype-only. **Not yet implemented:** effect runner, durable event log,
restart, snapshots, replay, remote models, random scheduling, named RNG,
transactional file append, canonical JCS hashes, observer projections or CLI.

## Determinism and schema design

- IDs: `hiyal:id:v1` SHA-256 with each UTF-8 component **length-prefixed
  as u64 big-endian bytes**. Trial IDs derive from user-supplied trial
  label and seed; event IDs derive from trial, logical step and actor ID.
  Agent IDs are stable u64 values supplied by the application in this slice.
  No UUID4 or wall-clock value is inside deterministic identities.
- Clock: 0-based committed action index; rejected actions never advance it.
  Later effect calls use 0-based call indices per actor per logical step.
  Events commit sequentially; later concurrent effects are *applied* in stable
  logical actor/call order, never network-completion order.
- D0 means validated identity-matched effect replay; D1 means bitwise repeat
  for a pinned same-target toolchain/dependencies/RNG/numerical profile; D2
  means statistical invariants across targets/live providers. **This slice
  demonstrates only repeated scripted actor/event ordering, not full D1**.
- Named RNG streams planned: algorithm **ChaCha20**, pinned crate version and
  explicit byte-to-index sampling rule, separated by trial/stream name. Persist
  complete RNG state in checkpoints. No `StdRng` promise, no global RNG.
  Random scheduler is not implemented in this slice.
- Canonical events, manifests and request identity: RFC 8785 JCS + SHA-256
  candidate; verify chosen implementation with published test vectors and
  reject NaN/infinity plus unsupported JSON number shapes before R1 hashes.
  **serde_json output here is not claimed canonical.** Floats and parallel
  reductions remain outside D1 until a numerical profile is pinned.
- Source version, schema version, profile, seed/stream states, provider
  effective settings, cache setting, tools, costs, snapshot offset and
  environment/platform belong in manifest. The current prototype's
  `schema_version: 1` event is provisional, not a stable published contract.
- Source of truth in R1: crash-consistent append-only committed event JSONL,
  snapshot with validated offset, atomic restore and fork only at a committed
  boundary. fsync/write/rename/recovery policy must be specified and tested
  before claiming durable commits. Observer retention defaults bounded.

## Model boundary design (data shape only)

`EffectRequest` separates model from tool requests and includes stable
`EffectCallKey` (trial, step, actor, call index, JCS request hash). Model
requests include role messages, provider/model/effective version, sampling
options, tool schemas and output format. The runner must distinguish
*requested* vs *effective* parameters, verify model capabilities, track usage
and charges, and disable caching by default for sampled scientific runs.
Scripted, keyed recorded-replay and opt-in OpenAI-compatible HTTP are the
first actual backends; native provider dialects and embeddings follow only
as real workloads demand. A seed unsupported by a live provider is an
explicit unsupported capability, not a reproducibility promise.

## Regression/acceptance register

| Python issue from review | Required acceptance |
| --- | --- |
| All queued actors paused hangs | Bounded selection returns no eligible actor (**unit tested for round-robin**) |
| Duplicate actor registration biases selection | Duplicate is an error (**unit tested**) |
| Bad weights (negative, nonfinite, zero total) | Validate before weighted scheduler selection (R1) |
| Failed restore partially changes state | Validate whole snapshot and atomically replace (R1) |
| Unseeded/global RNG and different resumed run | Named trial streams; resume at k matches uninterrupted run (R1) |
| Agent input duplicated on retries | Retry call identity and conversation history are idempotent (R1) |
| Failed completion hook leaves partial state | Transition staged before commit; observers read events only (staging unit tested) |
| Mismatched vector dimensions silently truncate | Reject dimensions (if native vector search is later needed) |
| Random IDs/wall-clock in hashes | Stable logical identifiers (**unit tested**) |
| Omitted sampling/provider settings | Explicit requested/effective values and unsupported-capability errors (R1) |
| Replay shared completion cursor | Keyed replay by logical call ID, out-of-order completions stable (R1) |
| Unbounded observer full snapshots | Incremental metrics and bounded retention (R2) |

Minimal scripted golden acceptance fixture:
`fixtures/golden/scripted-two-agent-v1.json`. This encodes the *target* behavior
for the new engine, not a claim that legacy Python emitted identical IDs.

## Deliberately retired features

No FAISS adapter, Transformers/HF agent, global PyTorch seeding, legacy
`AgentFactory`/`InteractionHistory`, toy sentiment or intent metrics, or old
Python release/CI paths. Retain useful capabilities via generic Rust policies,
scripted/model backends, structured memory and explicitly reviewed prompt/tool
guards later. OpenTelemetry may be an optional OTLP exporter, never a required
runtime dependency. Parquet stays optional until needed.

## R0 evidence status and exit criteria

- [x] Hiyal package name chosen as development placeholder; publishing off.
- [x] Extension/effect/commit conceptual contracts documented.
- [x] Stable logical IDs and event ordering specified for the first slice.
- [x] Minimal scripted fixture and Rust tests for identity, staging and eligibility.
- [x] Python observer-on/off baseline **harness** created.
- [ ] Independently run Python observer-on/off and snapshot benchmarks with
      hardware/version/provenance details; publish comparable measured results.
- [ ] Verify GitHub handle/domain/trademark and crates.io publication ownership.
- [ ] Pin and test named-RNG numerical profile and RFC 8785 test vectors in R1.
- [ ] Implement durable replay/resume and confirm independent application gate.

Do not mark R0 complete based only on code review or unexecuted benchmark
scripts. The root-level Rust files are a beginning of R1, not its completion.
