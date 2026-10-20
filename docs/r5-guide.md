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
- The archived Python tree remains accessible temporarily for **manual**
  baseline calibration. Delete only after preserving independent measurements
  and historical evidence; it is not part of any native build.
