# R0 acceptance ledger

Issue: [#103](https://github.com/davletovb/hiyal/issues/103).
Implementation work: [PR #106](https://github.com/davletovb/hiyal/pull/106).

| Requirement | Evidence | Status |
| --- | --- | --- |
| Distinct development package and intended brand | Root `Cargo.toml`, [name diligence](r0-name-diligence.md) | Decided (unpublished) |
| Rust extension model and CLI registration limits | [contracts](r0-contracts.md), `examples/custom_world/` | Specified and code sketched; build unverified |
| Typed world, action, scheduler, model/effect and log boundary | [contracts](r0-contracts.md), `src/core.rs`, `src/effects.rs` | Specified; prototype |
| IDs, logical time, random streams, numerical profile, JSON hashing | [R0 decisions](r0-decisions.md) | Specified; conformance tests are R1 |
| Archive defect register and golden cases | `fixtures/golden/`, Rust core tests, R0 decisions | Authored; tests unexecuted |
| Python observer-on/off and snapshot benchmark plan | `benchmarks/reference_python.py`, archived checkpoint benchmark, [reviewer-only results](../benchmarks/REVIEWER_REPORTED.md) | Harness authored; independent results missing |
| Removed/retained features list | [R0 decisions](r0-decisions.md) | Done |
| Prior art/stop conditions | [prior art](r0-prior-art.md), issue #103 | Done |
| crate reservation, trademark and registry authorization | [name diligence](r0-name-diligence.md) | Release blocker, not falsely marked complete |

## Execution facts

The current agent environment has **Python 3.13.5 but no `rustc`/`cargo`**.
Its container also cannot connect to github.com to fetch the archived project.
Consequently **neither `cargo test --workspace` nor independently measured
Python observer-on/off trials ran here**. Running a simulated Python benchmark
or posting guessed speedups would not constitute R0 evidence.

The original reviewer figures are preserved with explicit attribution in
`benchmarks/REVIEWER_REPORTED.md`; they are NOT a substitute for the required
fresh measured baseline.

To finish the hard gate, a machine with Rust + Python + legacy dependencies
must execute `sh scripts/r0-verify.sh` and
`python3 hiyal-python/benchmarks/checkpoint_scaling.py --profile standard
--repeat 3 --output benchmarks/results/python-checkpoints.json`, then retain
the raw JSON and machine manifest in the evidence review (prefer an opt-in
artifact or attached issue). Run Rust checks before interpreting benchmark
results. Do **not** activate legacy GitHub Actions or Dependabot.

The project may move on to R1 implementation in parallel, but **R0 is not
verified/complete until these measurements and build checks are recorded**.
