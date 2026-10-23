# Profiling an experiment built on Hiyal

The [phase benchmarks](../benchmarks/phase/README.md) measure the engine on
synthetic worlds. This guide covers the other half: finding where a real
experiment spends its time, changing something, and showing that the change is
both faster and exactly equivalent. It is the procedure used for #117.

## 1. Record the machine

Run the baseline phase report once on the machine you will measure on:

```sh
cd benchmarks/phase && cargo build --locked --release --bin phase-report
./target/release/phase-report run --cold --drop-caches --out machine-baseline.json
```

Its provenance records the CPU, its flags and whether `sha2` uses its
hardware backend. The per-phase numbers show what the engine costs on this
machine before any experiment code is involved. In particular, SHA-256 cost
differs several-fold between machines with and without SHA-NI.

## 2. Time the experiment end to end

Use the experiment's own command, release build, a fixed seed and a new output
directory each time. Time the run and verification steps separately, and
record event counts and journal bytes (`wc -lc` on each `events.jsonl`).
Repeat at least three times and alternate any variants you compare
(A, B, B, A, …) on the same machine. Report medians with min–max ranges.

If user time is close to wall time and system time is small, the run is
CPU-bound and fsync is not the bottleneck. Confirm with
`HIYAL_PHASE_BENCH_DIR` on a tmpfs, or with the experiment's output on a RAM
disk.

## 3. Profile where the time goes

Build with debug symbols in a separate target directory, then profile a small
case under callgrind. It counts instructions, so it is deterministic and
needs no idle machine:

```sh
CARGO_PROFILE_RELEASE_DEBUG=true CARGO_TARGET_DIR=/tmp/prof-target \
  cargo build --release --locked --bin <experiment-binary>
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out \
  /tmp/prof-target/release/<experiment-binary> <small-case-args>
callgrind_annotate --inclusive=yes /tmp/cg.out | head -60
callgrind_annotate --inclusive=yes --tree=caller /tmp/cg.out \
  | grep -B12 'hiyal::storage::digest$'
```

Look at inclusive cost and call counts together. For example, #117 found
4,172 world digests during replay against 1,440 while stepping. That pointed
at repeated replays, not at the hash itself. Valgrind hides the CPU's SHA
extensions from the program (checked with valgrind 3.22), so under callgrind
`sha2` always takes its software path. That overstates SHA-256 on machines
with SHA-NI, so read SHA-256's share alongside the phase report from step 1.

## 4. Prove the change is equivalent

Faster is not enough: evidence written before and after must agree.

- Run the same seed with the old and new build and compare every journal
  event field for field: actors, actions, state hashes, schedules and RNG
  positions. Only identity fields that embed build provenance (trial ids,
  chain hashes) may differ.
- Cross-verify: the new build must verify the old build's evidence, and the
  old build must verify the new evidence.
- Re-verify archived evidence, for example experiment 002's V0 release with
  `scripts/verify-002.sh` in social-emergence-lab.
- For serialization or hashing changes, add differential tests against the
  previous implementation, as `src/canonical.rs` and `src/tagged.rs` do.
  Check that they fail when the change is deliberately broken.

## 5. Record the result

Write a report under `benchmarks/results/` with the baseline and change
commits, toolchain, machine, workloads, event counts, journal bytes, raw
medians and ranges, and the equivalence checks. That directory is ignored by
default, so add a report with `git add -f`. Use
`benchmarks/phase/compare.sh` when the change is in the engine, so the phase
numbers for both revisions come from the same machine and the same session.
