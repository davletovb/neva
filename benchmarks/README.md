# R0 measurement protocol

Run the preserved Python baseline **before** asserting any Rust performance
advantage. The baseline is not a dependency or part of the Rust engine.

Install the archived project's core dependencies in an isolated Python
environment (see `hiyal-python/pyproject.toml`), then from the repository root:

```sh
python benchmarks/reference_python.py --agents 4 --steps 100 500 --repeat 3 \
  --output benchmarks/results/python-baseline.json
```

The script runs observer-on and observer-off workloads in fresh subprocesses.
Every turn is a fixed offline `"ok"` response. It reports import and setup
time, median and p95 step time, throughput, process CPU time, peak RSS and
periodic retained-observer-entry counts. Current RSS checkpoints require Linux.
Outputs in `benchmarks/results/` are machine-generated and are not committed
without hardware/source/version provenance.

The existing `hiyal-python/benchmarks/checkpoint_scaling.py` remains the
snapshot/export baseline. It should be run separately on matching hardware.
Future Rust comparisons must use equivalent action semantics and reporting
windows; network-bound live-provider comparisons are a separate class.
Record `uname -a`, CPU model, memory, Python version, Rust toolchain, source
commit, flags and repeats alongside numbers. Do not substitute the reviewer's
reported Python figures for independently measured evidence.

The R0 gate remains open until measured figures and platform metadata are
committed. This script provides the reproducible harness, not invented results.

## Complete R0 evidence collection (manual; no CI)

Run `sh scripts/r0-verify.sh` on a machine with Rust and the archived
Python core dependencies installed. This runs compiled Rust workspace tests and
a standalone custom-world consumer, identity-fixture validation, plus observer
on/off baselines at 20 and 100 agents and increasing steps. Run the archived
checkpoint benchmark separately. The script does not change repository CI
configuration. See [acceptance ledger](../docs/r0-verification.md).
