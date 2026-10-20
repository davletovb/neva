# R5 benchmarking protocol: no fabricated speedups

R5 requires **independent same-hardware measurements** of Python observer
on/off and Rust observer on/off before any efficiency claim. The historical
Python and Rust event-sourced worlds are not equivalent by default:
`BasicEnvironment.step` is an in-memory synchronous Python scheduler/LLM
mock, while `Trial::step` durably fsyncs a hash-chained state transition,
even when its observer is disabled. Reporting the raw ratios as "Rust is
Nx faster" would be scientifically invalid.

## Measurement commands (manual only)

On a dedicated Linux host with Rust plus the isolated archived Python
dependencies installed. Fetch the frozen archive branch into a separate
worktree first; the archived source is intentionally absent from Rust main:

```sh
git worktree add --detach ../hiyal-python-reference archive/hiyal-python-pre-r5
export HIYAL_LEGACY_SOURCE="$(cd ../hiyal-python-reference/hiyal-python && pwd)"
```

Then record provenance and run workloads (manual only):

```sh
git rev-parse HEAD
rustc --version
cargo --version
python --version
uname -a
cargo build --release --locked --workspace --bins
python benchmarks/reference_python.py --agents 20 --steps 100 500 1000 --repeat 5 --output benchmarks/results/python-20.json
python benchmarks/reference_python.py --agents 100 --steps 100 500 1000 --repeat 5 --output benchmarks/results/python-100.json
for agents in 20 100; do
  for steps in 100 500 1000; do
    for observed in on off; do
      for iteration in 1 2 3 4 5; do
        target/release/hiyal-native-bench "$agents" "$steps" "$observed"
      done
    done
  done
done
```

Rust's `hiyal-native-bench` uses a reproducible scripted counter, committed-event
fsync on every step, and optional JSONL/CSV projections. Its JSON output
reports observer state, wall latency distribution, throughput, committed log
growth and Linux RSS checkpoints. The Python baseline measures an in-memory
scripted agent scheduler and its historical accumulating observer. Each
Python sample is a fresh process. Save *all* Rust raw JSONL samples to files,
together with CPU/memory, kernel, filesystem, git SHA, dependency and toolchain
versions, process start-up timings (shell or hyperfine), and flags. Clean
worktrees and warm/cold-cache methodology must be consistent. These are
**different engine semantics**: compare relative observer overhead *within*
each runtime; do not infer cross-language engine speedups from raw throughput.

For a genuinely comparable future same-workload benchmark, create a separate
pair of pure, no-IO decision kernels with matching agents, deterministic
scheduler, action and observer projection semantics, and instrument full
start-up and retained-memory growth with identical reporting boundaries.
Do not call this available until those independent fixtures and parity tests
exist.

Run remote/network-bound model inference as a **separate** measured tier
with provider identity, request concurrency, quotas and network latency
provenance; never mix those numbers with offline model/scheduler results.
Provider cost and retry/timeout semantics can dominate both runtimes.

**No benchmark is triggered by PR CI**, and no made-up baseline or reviewer
reported numbers are considered acceptance evidence. Until raw Python/Rust
data are published, the R5 comparative benchmark gate stays open.
