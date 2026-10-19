# Hiyal

**Hiyal** (Arabic *ḥiyal*, roughly "hee-YAHL") refers to ingenious devices
and automata. It is being built as a reproducible, extensible **Rust-native
agent-based simulation engine** for research and experimentation.

The project is a new engine informed by the historical Python library, **not
a feature-parity Python port**. Hiyal's Rust library and eventual CLI execute
models; JSONL and later open-data exports let Python, R, Julia, Rust and DuckDB
analyze the results independently. No Python runtime or bindings are required
by the Rust engine.

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
