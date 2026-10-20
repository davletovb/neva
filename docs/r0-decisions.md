# Issue #103 — R0 decision record and R1 starter boundary

Status: **R0 implementation and decision artifacts prepared; execution evidence pending**. This is a new Rust engine informed by the archived
Python implementation, *not* a line-by-line translation. No Python bindings,
embedded Python runtime, or required Python analysis stack. The old package
is preserved on [archive/hiyal-python-pre-r5](https://github.com/davletovb/hiyal/tree/archive/hiyal-python-pre-r5) for reference benchmarks only
(see `HIYAL_LEGACY_SOURCE` in `benchmarks/README.md`).

## Naming and scope

- Project and repository: **Hiyal**. Arabic *ḥiyal*, used for ingenious
  devices/automata, also meaning tricks; pronounced approximately "hee-YAHL".
- Root Rust library package: `hiyal`; package publishing is deliberately
  disabled until name/ownership/legal checks and R0 acceptance. The old Python
  package having the same name is an archive, not a dependency.
- `hiyal.com` is occupied by an unrelated established Uyghur computing site.
  No domain ownership or affiliation is claimed. A GitHub user/org `hiyal`,
  authoritative multi-jurisdiction trademark clearance and crates.io publisher
  ownership remain unverified. The issue's October 19 registry check is a
  planning observation, not a reservation. See [name diligence](r0-name-diligence.md).
- **Decision:** Do not publish/reserve empty crates from this prototype; avoid
  squatting or presenting unreleased APIs as stable. A verified publisher with
  registry credentials can release the smallest useful crate after acceptance.
  Only `hiyal` is currently planned for publication; `-core/-cli` are not
  reserved or promised.
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

1. Scheduler selects an eligible actor; every selected attempt ends with an
   accepted world event or an explicitly audited rejection. Both advance
   attempt ordering, but only accepted world events advance the logical
   committed-step counter.
2. World produces an immutable observation.
3. Policy may request typed model/tool effects; runner performs asynchronous
   bounded I/O *outside* the deterministic world reducer.
4. Effect results are resolved into a proposed typed action and validated.
5. Stage a transition; on success commit world changes plus a committed
   world event. On rejection, world and committed sequence stay unchanged,
   but scheduler eligibility advances and a separate rejection attempt
   is audited; this must also be reproduced during R1 resume.
6. Observer consumes the committed event. Its failure cannot roll back or
   mutate the world; it may produce an independently recorded diagnostic.
7. Charged/irreversible effects have a separate ledger even when a proposal
   is rejected. Rejection feedback is explicit and logged, not silently
   appended to an agent conversation on retry.

**Implemented in this initial slice:** a pure synchronous scripted policy,
typed world, round-robin selection, deterministic trial/event IDs, staged
world commits, rejection audit records and in-memory committed events.
JSONL export is
prototype-only. **Not yet implemented:** effect runner, durable event log,
restart, snapshots, replay, remote models, random scheduling, named RNG,
transactional file append, canonical JCS hashes, observer projections or CLI.

## Determinism and schema design

- IDs: `hiyal:id:v1` SHA-256 with each UTF-8 component **length-prefixed
  as u64 big-endian bytes**. Trial IDs derive from user-supplied trial
  label and seed; event IDs derive from trial, logical step and actor ID.
  Agent IDs are stable u64 values supplied by the application in this slice.
  Event IDs use a versioned `event-v2` domain and also incorporate a
  provisional JSON action hash plus previous committed event ID. This
  detects divergent actions and preserves history-chain identity. Full JCS
  transition/state hashing remains R1 work.
  No UUID4 or wall-clock value is inside deterministic identities.
- Clock: separate monotonically increasing committed-action and selected-attempt
  indices. Rejections advance the attempt index and schedule cursor (avoiding
  starvation), but never advance the world committed-action index.
  Later effect calls use 0-based call indices per actor per logical step.
  Events commit sequentially; later concurrent effects are *applied* in stable
  logical actor/call order, never network-completion order.
- D0 means validated identity-matched effect replay; D1 means bitwise repeat
  for a pinned same-target toolchain/dependencies/RNG/numerical profile; D2
  means statistical invariants across targets/live providers. **This slice
  demonstrates only repeated scripted actor/event ordering, not full D1**.
- **R0 RNG contract:** trial-owned named `ChaCha20Rng` streams; implementation
  candidate `rand_chacha 0.9.x` locked to an exact Cargo.lock release for D1.
  Seed bytes derive from SHA-256 of domain-separated, length-prefixed
  `(trial seed decimal string, trial ID, stream name)`. Draw `u64` values
  in explicitly defined byte order; for uniform [0,n), reject values less
  than `(-n mod 2^64) mod n` and choose `value % n`. This avoids modulo bias.
  Store full stream state and draws/word position on checkpoint. Separate
  `scheduler`, `world`, `agent/<id>` and `intervention/<id>` streams.
  The implementation must pin/test exact upstream RNG output and the
  byte-to-u64 rule before asserting D1. No `StdRng` promise or global RNG.
  This is a **design contract, not a shipped random scheduler**.
- Canonical events, manifests and request identity: **RFC 8785 JCS + SHA-256**.
  Candidate Rust implementation `serde_json_canonicalizer 0.3.2` (verify RFC
  vectors independently). Reject duplicate object keys, nonfinite numbers,
  and JSON integer values outside the exact IEEE-754 safe integer interval
  `[-9007199254740991, 9007199254740991]`. Encode larger integer/ID
  quantities as **decimal strings** before canonicalization, not silently
  rounded floating-point numbers. Forbid normalization of Unicode contents;
  sort JSON properties under JCS UTF-16 rules. Use versioned hash domains.
  Tests must include 1 vs 1.0, -0.0, exponent notation, multilingual key order,
  nonfinite rejection and the large-integer guard.
  JSONL exports encode `u64` actor IDs, committed/attempted steps, and model
  seed/usage totals as *decimal strings* to preserve their full range across
  language readers. Finite floating-point sampling checks reject NaN/Inf
  before serialization (rather than serializing them as null).
  **serde_json output here is not claimed canonical.**
  For D1, integer and fixed-point transitions are supported initially;
  floating-point transcendental algorithms, parallel nonassociative reductions
  and cross-target numerical identity remain D2 until a pinned profile exists.
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
| Failed completion hook leaves partial state | World transition staged before commit; rejected attempts auditable; observers read events only (partial staging unit tested) |
| Mismatched vector dimensions silently truncate | Reject dimensions (if native vector search is later needed) |
| Random IDs/wall-clock in hashes | Stable logical identifiers, action content hash and predecessor link (**unit tested**) |
| Omitted sampling/provider settings | Explicit requested/effective values and unsupported-capability errors (R1) |
| Replay shared completion cursor | Keyed replay by logical call ID, out-of-order completions stable (R1) |
| Unbounded observer full snapshots | Incremental metrics and bounded retention (R2) |
| Rejected actor starves other actors | Attempt cursor advances, rejection recorded without world commit (unit tested) |
| Public event vector enables ID reuse | Private step counter and read-only world, monotonic IDs across drained events (unit tested) |
| Large integer wire values lose precision | Actor IDs, logical steps, seeds and token usage serialized as decimal strings (unit tested) |
| Non-finite sampling silently becomes null | Checked serialization rejects NaN/Inf (unit tested) |

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
- [x] Python observer-on/off baseline **harness** created and acceptance
      runner authored; prior art and independent name-risk findings recorded.
- [x] TOML v1 config, effect envelopes, observer and event append boundaries
      sketched in [contracts](r0-contracts.md).
- [ ] Independently run Python observer-on/off and snapshot benchmarks with
      hardware/version/provenance details; publish comparable measured results.
- [x] Domain collision independently checked and documented; publication
      policy set to unpublished until ownership and legal review.
- [ ] Authoritative trademark clearance and verified registry publisher account
      before any public crate publication (external gate, cannot infer clearance
      from a web search).
- [x] Choose named RNG/sampling, numerical D1 scope, canonical JSON safe-number
      contract and explicit conformance fixtures; executable library integration
      and pinned golden values remain R1.
- [x] Run `cargo test --workspace` on GitHub's standard Ubuntu Rust runner:
      [run 39246204608](https://github.com/davletovb/hiyal/actions/runs/39246204608)
      verified rustfmt and the custom-world binary; subsequently
      [run 39247148177](https://github.com/davletovb/hiyal/actions/runs/39247148177)
      verified 11 Rust tests both in the normal workspace and with
      `serde_json/preserve_order` enabled at `d65680a`. The local container lacks Rust; Clippy was not run.
- [ ] Implement durable replay/resume and confirm independent application gate.

The engineering **preparation** of R0 is complete and the initial Rust
toolchain checks have now passed, but the R0 **measured acceptance gate remains
open** until independently measured Python observer-on/off and snapshot
benchmark outputs are recorded. Do not invent results, claim legal clearance or mark issue
#103 closed. The root-level Rust files are a beginning of R1, not its completion.
