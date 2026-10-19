# R0 acceptance ledger

Issue: [#103](https://github.com/davletovb/hiyal/issues/103).
Implementation work: [PR #106](https://github.com/davletovb/hiyal/pull/106).

| Requirement | Evidence | Status |
| --- | --- | --- |
| Distinct development package and intended brand | Root `Cargo.toml`, [name diligence](r0-name-diligence.md) | Decided (unpublished) |
| Rust extension model and CLI registration limits | [contracts](r0-contracts.md), `examples/custom_world/` | Specified; Rust custom-world builds and runs in focused CI |
| Typed world, action, scheduler, model/effect and log boundary | [contracts](r0-contracts.md), `src/core.rs`, `src/effects.rs` | Specified; prototype |
| IDs, logical time, random streams, numerical profile, JSON hashing | [R0 decisions](r0-decisions.md) | Specified; conformance tests are R1 |
| Archive defect register and golden cases | `fixtures/golden/`, Rust core tests, R0 decisions | Authored; 11 Rust tests passed on d65680a |
| Python observer-on/off and snapshot benchmark plan | `benchmarks/reference_python.py`, archived checkpoint benchmark, [reviewer-only results](../benchmarks/REVIEWER_REPORTED.md) | Harness authored; independent results missing |
| Removed/retained features list | [R0 decisions](r0-decisions.md) | Done |
| Prior art/stop conditions | [prior art](r0-prior-art.md), issue #103 | Done |
| crate reservation, trademark and registry authorization | [name diligence](r0-name-diligence.md) | Release blocker, not falsely marked complete |

## Execution facts

The local agent container has **Python 3.13.5 but no `rustc`/`cargo`** and
cannot connect directly to github.com. However the purpose-built Rust smoke
workflow **ran on GitHub's Ubuntu runner** at commit
[`e96e523`](https://github.com/davletovb/hiyal/commit/e96e52323b4aad1a6bb2ed3473ded4a3c7f546fe):
[`run 39246204608`](https://github.com/davletovb/hiyal/actions/runs/39246204608)
passed `cargo fmt --all -- --check`, `cargo test --workspace` (**10 passed,
0 failed**), and the standalone custom-world smoke. The extended Clippy
task was intentionally skipped because this was a low-cost PR verification.
The independent Python observer-on/off and checkpoint benchmarks **remain unrun**
with attributable machine/source evidence. Running a simulated Python benchmark
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

The new [low-cost Rust CI policy](ci-budget.md) provides a narrow Rust build/test
check on selected pull-request lifecycle events and version tags. It does not
run Python benchmarks or replace the measured R0 evidence requirement.


The October 19 reviewer reported testing the original branch `047b4b9`
(4 Rust tests passed, formatting and Clippy failed). The revised branch passed format, all 10 tests and standalone example in
[`run 39246204608`](https://github.com/davletovb/hiyal/actions/runs/39246204608).
New determinism fixes passed **11 tests** including feature-unification tests in
[`run 39247148177`](https://github.com/davletovb/hiyal/actions/runs/39247148177)
on `d65680a`. Subsequent documentation-only changes do not modify the
tested Rust sources. Extended Clippy remains unverified until an intentional
release/manual-extended run. CI is deliberately low-frequency; if code changes,
re-verify the resulting head before merge.

Two subsequent review threads identified downstream
`serde_json/preserve_order` feature unification and untracked-source
provenance. Commit `d65680a` fixes both. A final focused
[CI run 39247148177](https://github.com/davletovb/hiyal/actions/runs/39247148177)
**passed** formatting, **11 workspace tests**, **11 crate tests** under
`serde_json/preserve_order`, and the standalone custom-world smoke.
Extended Clippy still requires an intentional manual/release check, and
independent Python baseline figures are still outstanding.
