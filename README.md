# Hiyal

**Hiyal** (Arabic *ḥiyal*, roughly "hee-YAHL") refers to ingenious devices
and automata. It is being built as a reproducible, extensible **Rust-native
agent-based simulation engine** for research and experimentation.

The project is a new engine informed by the historical Python library, **not
a feature-parity Python port**. Hiyal's Rust library and eventual CLI execute
models; JSONL and later open-data exports let Python, R, Julia, Rust and DuckDB
analyze the results independently. No Python runtime or bindings are required
by the Rust engine.

## R5 verification and release preparation

R5 adds bounded/streamed event-log restore, hostile/corrupt-input restart tests,
an independent custom local-model/allowlisted-tool executable, offline Rust
measurements, and cross-platform **opt-in** release qualification.
The release and comparable Python performance gates are **not yet certified**.
See [Rust SDK and CLI guide](docs/r5-guide.md),
[release and rollback](docs/r5-release.md), and
[benchmark protocol](benchmarks/r5-protocol.md).

```sh
cargo test --workspace --locked
cargo run --locked -p hiyal-custom-model-example
cargo run --locked --release -p hiyal-native-bench -- 20 100 off
```

## Native memory and redacted observation (R4)

R4 introduces bounded, deterministic structured memory with TTL and adaptive
salience for Rust applications, plus privacy-conscious derived JSONL event
summaries. The CLI can export verified redacted event summaries without model
prompts or world snapshots. See [R4 native contracts](docs/r4-native.md).

```sh
cargo run --locked -p hiyal -- run examples/counter.toml /tmp/hiyal-r4
cargo run --locked -p hiyal -- export /tmp/hiyal-r4 /tmp/hiyal-r4.jsonl
```

## Guarded provider and tool boundaries (R3)

The optional `live-http` feature supports bounded OpenAI-compatible JSON/SSE
responses, conservative HTTP retry classification, circuit breaking, and
captured model/usage metadata. By default no live network or tool execution
occurs. `guard::GovernedBackend` enforces process-local micro-USD reservations,
request/output limits, a bounded total effect count, explicit tool validators,
and call-key-scoped one-time approvals for potentially side-effecting tools.
The Rust CLI's `--live` mode requires explicit token prices and a spend cap.
See [R3 behavior and deferred contracts](docs/r3-reliability.md).

## Research workflows (R2)

R2 adds reproducible parameter sweeps, bounded Rust-native trial concurrency,
independent paired counterfactual forks, restart recovery, portable CSV metric
series and reference toy societies. See [research runner and guarantees](docs/r2-engine.md).

```sh
cargo run --locked -p hiyal -- sweep examples/sweep.toml /tmp/hiyal-study
cargo run --locked -p hiyal -- run examples/counter.toml /tmp/hiyal-parent
cargo run --locked -p hiyal -- fork /tmp/hiyal-parent /tmp/hiyal-paired 3 12
```

The native run produces per-job `events.jsonl`, `metrics.csv`, verified
`manifest.json` and `snapshot.json`, and a stable `results.csv` summary. Forks
bundle a copy of the validated parent history for portable ancestry checks.

## Native engine (R1)

Hiyal ships a typed Rust world/policy API, deterministic random and
round-robin scheduling, append-only committed/rejected events, versioned
checkpoints, strict keyed model-effect replay, opt-in OpenAI-compatible HTTP,
and a built-in `hiyal` CLI. User-defined worlds compile in independent
consumer crates. See [R1 usage and guarantees](docs/r1-engine.md).

```sh
cargo test --workspace
cargo run -p hiyal -- run examples/counter.toml /tmp/hiyal-counter
cargo run -p hiyal -- resume /tmp/hiyal-counter 10
cargo run -p hiyal-custom-world-example -- run /tmp/hiyal-custom 5
```

R1 tests are Rust-only; Python is never required by Hiyal's engine or CI.
The current package is still unpublished. Full R0 Python baseline
measurements remain an optional **separate historical measurement** and are
not part of the R1 implementation gate.

## Archived Python reference

The prior implementation, tests, docs and historical (inactive) CI are under
[`hiyal-python/`](hiyal-python/README.md). The new Rust workspace does not
depend on it. The [Python benchmark harness](benchmarks/README.md) is used
only to measure the legacy observer-on/off reference fairly. The historical
Python CI and Dependabot configuration remain archived and inactive. The
[new Rust verification workflow](docs/ci-budget.md) is deliberately minimal:
one Linux job on selected PR lifecycle events, an explicit manual request, or
a version tag—not on every agent push.
