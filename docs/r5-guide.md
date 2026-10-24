# R5 — Rust SDK/CLI research handbook

Use only the Rust toolchain (Rust 1.85 or newer); no Python interpreter or
binding is required to run the native engine.

## Built-in CLI

```sh
cargo test --workspace --locked
cargo run --locked -p hiyal -- run examples/counter.toml ./my-trial
cargo run --locked -p hiyal -- inspect ./my-trial
cargo run --locked -p hiyal -- resume ./my-trial 5
cargo run --locked -p hiyal -- validate ./my-trial
cargo run --locked -p hiyal -- export ./my-trial ./redacted.jsonl
cargo run --locked -p hiyal -- sweep examples/sweep.toml ./my-study
cargo run --locked -p hiyal -- fork ./my-trial ./my-forks 3 8
```

Use new paths (a fresh run refuses to overwrite an existing trial); the
export file must not exist and must be outside the trial tree. Inspect and
validate do not write/repair trial data. The CLI intentionally supports only
registered built-in counter models; user-defined worlds are independent
Rust executables using the public SDK.

## SDK contract and extension points

- Implement `core::World` with `Clone`, `observe(actor)` and pure
  `apply(actor, action)`. The world/action must implement serde for durable
  `experiment::Trial`. `apply` must not perform I/O or mutate global state.
  Worlds and actions may hold `f64`: Hiyal parses JSON floats exactly
  (serde_json's `float_roundtrip`), so replay reproduces them bit for bit.
  Keep them finite; JSON writes NaN and infinities as `null`, which does not
  parse back as a float.
- Implement `experiment::EffectPolicy<W>`: `requests` declares typed
  `runtime::EffectSpec` requests and `decide` consumes ordered receipts to
  return an action. For policies without external effects, `Rules<P>` adapts
  a pure synchronous `core::Policy`.
- Implement `runtime::Backend` to supply a local model (or explicitly
  controlled tool executor). The default `permits_tools()` is false.
  Whitelist concrete tool names and validate input; never turn on tool
  permission for arbitrary untrusted commands or side effects.
- `Trial::create` takes the world, policy, backend, stable actors, seed and
  `Scheduling`. `run_attempts` records accepted/rejected transitions;
  `checkpoint` writes a verified snapshot. `Trial::resume` recovers from
  the authoritative hash chain. `replay_only = true` forbids any new effect
  dispatch and requires recorded keyed receipts.
- For a randomized initial world (labels, graphs), `Trial::create_with_setup`
  builds it with a closure that draws only from the trial's own
  `setup`/`setup/<name>` streams. The advanced positions are recorded in
  genesis and carried into forks, so setup is reproducible from the name and
  seed; `experiment::verify_setup` re-runs the closure and checks it against
  genesis. Do not hand-build a `Genesis` for this. `verify_setup` checks
  **only** the setup: it does not authenticate the event journal, snapshots
  or fork ancestry, so also run `Store::inspect_read_only` (and
  `research::verify_fork` for forks) before treating a trial as verified.
- **Staging cost.** The engine stages each action with
  `World::apply_undoable`, whose default copies the world before applying
  it. When an action changes a small, known part of a large world, override
  it to apply in place and return `core::Undo::revert` with a closure that
  puts back exactly what changed: an old counter value, the record just
  pushed. The engine undoes the action when its event cannot be committed:
  when it is too large to record or the append fails. An override must
  leave the world unchanged when it returns `Err`, and its undo must be
  exact. The engine cannot check either, and full replay catches a fault
  only when a later accepted event or a checked snapshot hashes the world:
  a faulty `Err` or undo is recorded as a rejected event, which has no
  state hash, so a fault in the last steps of a trial can pass replay.
  Derive results from freshly replayed evidence, not from the live world of
  the process that ran the trial. Replay itself applies events in place for
  every world.
- **Do not keep the trajectory in the world.** Every event serializes and
  hashes the whole world, so a world that appends a record per event costs
  O(T²) over a trial. Keep only what the reducer needs, and derive the
  trajectory from the journal with `Store::inspect_read_only_with`. It runs
  the same full verification as `inspect_read_only` and calls a closure with
  each committed event (`sequence()`, `logical_step()`, `actor`, `outcome`)
  and the world after it. Later events and the snapshot are checked after
  the closure has run for earlier ones, so treat its output as provisional
  and publish it only once the call returns `Ok`. A trajectory already
  stored in a published world
  kind is part of its hashes: move it out under a new versioned world kind,
  never by changing an existing one.
- Build `rng::Streams` with `Streams::new` (or `paired_branch`), not a
  struct literal: it has a private, non-serialized field. Its serialized
  form, draws and positions are part of the evidence and do not change.
- `memory::MemoryBank` is optional explicit world state, not a hidden global.
  `JsonlObserver` and `CsvMetrics` are derived projections; errors are
  diagnostics and should be surfaced by the caller.

## Independently authored model and tools

The [external native example](../examples/custom_model/src/main.rs)
implements its own `Village` world, policy, `OfflineCouncil` backend,
one model request and one explicitly allowlisted *side-effect-free* tool.
The custom model is compiled in its **own crate**, not hard-coded into the
CLI, with no API key, Python or remote calls.

```sh
cargo run --locked -p hiyal-custom-model-example -- run ./my-village 4
cargo run --locked -p hiyal-custom-model-example -- resume ./my-village 3
cargo run --locked -p hiyal-custom-model-example -- inspect ./my-village
```

Provider/model versions and request hashes are recorded in the effect ledger.
Receipts are durably captured before a world action is committed. Repeat the
same request keys in strict replay instead of calling the model twice. In
production, review permissions, external billing and spend limits from
[guarded effects](r3-reliability.md).

The original [independent custom-world example](../examples/custom_world/src/main.rs)
shows policy-only actions. Both examples use the Rust API without modifying
Hiyal internals.

## Research reproducibility and counterfactuals

```sh
cargo run --locked -p hiyal -- sweep examples/sweep.toml ./study
cargo run --locked -p hiyal -- run examples/counter.toml ./baseline
cargo run --locked -p hiyal -- fork ./baseline ./pair 3 8
cargo run --locked -p hiyal -- validate ./pair/control
cargo run --locked -p hiyal -- validate ./pair/treatment
```

A sweep derives independent trial seeds and runs with bounded concurrency.
A fork captures a verified committed boundary, including scheduler/random
state and parent evidence; the control and treatment arms use independent
substreams and distinct trial IDs. Compare outcomes across replicates and
preserve treatment attribution; do not substitute sample correlations for
a valid causal identification design. See [R2 contracts](r2-engine.md).

## Stable exports and trust boundaries

Use redacted JSONL event summaries for provenance without revealing policy
or raw prompt bodies. The authoritative manifest, full event log, effects
ledger and snapshots may contain secret or personally sensitive world inputs.
Limit access at filesystem and deployment level; there is no encryption at
rest or automatic sanitization of authoritative records. For exact data types
and storage limits see [R5 release contracts](r5-release.md) and
[R4 observability and memory](r4-native.md).

## Removed or intentionally not ported

- No native `TransformerAgent`, FAISS backend, Hugging Face Transformers,
  PyTorch state seeding or mandatory OpenTelemetry.
- No PyO3, wheels, notebooks as runtime, Python CLI, native Torch weights, or
  full historical Python feature parity.
- Existing generic model/tool/summarization capabilities are represented by
  typed Rust requests and opt-in backends, not specific provider SDKs.
- The historical Python runtime, tests and CI were removed from the Rust
  mainline after checking for committed research datasets. The complete source
  plus two tutorial notebooks are frozen on
  [archive/hiyal-python-pre-r5](https://github.com/davletovb/hiyal/tree/archive/hiyal-python-pre-r5).
  The independent baseline remains a **manual** deferred benchmark using
  `HIYAL_LEGACY_SOURCE` and a separate worktree, not part of native CI.
