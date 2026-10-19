# Hiyal

**Hiyal** (Arabic *ḥiyal*, roughly "hee-YAHL") refers to ingenious devices
and automata. It is being built as a reproducible, extensible **Rust-native
agent-based simulation engine** for research and experimentation.

The project is a new engine informed by the historical Python library, **not
a feature-parity Python port**. Hiyal's Rust library and eventual CLI execute
models; JSONL and later open-data exports let Python, R, Julia, Rust and DuckDB
analyze the results independently. No Python runtime or bindings are required
by the Rust engine.

## Current implementation status

An early Rust vertical slice is available: typed `World` and `Policy`
traits, guarded round-robin scheduling, deterministic logical identities,
staged sequential commits and in-memory event export. **This is not yet a
persistent, replayable research engine.** Async effects, durable event
storage, restart/fork, seeded random scheduling and the native CLI follow
as part of [issue #103](https://github.com/davletovb/hiyal/issues/103).

```sh
cargo test --workspace
cargo run -p hiyal-custom-world-example
```

The [custom-world example](examples/custom_world/src/main.rs) is a separate
consumer crate, showing how to extend Hiyal without editing its source.
The package is currently `publish = false`; no crates.io publication or name
reservation is claimed.

Read [R0 design decisions and acceptance register](docs/r0-decisions.md)
before treating any prototypes as stable APIs.

## Archived Python reference

The prior implementation, tests, docs and historical (inactive) CI are under
[`hiyal-python/`](hiyal-python/README.md). The new Rust workspace does not
depend on it. The [Python benchmark harness](benchmarks/README.md) is used
only to measure the legacy observer-on/off reference fairly. The historical
Python CI and Dependabot configuration remain archived and inactive. The
[new Rust verification workflow](docs/ci-budget.md) is deliberately minimal:
one Linux job on selected PR lifecycle events, an explicit manual request, or
a version tag—not on every agent push.
