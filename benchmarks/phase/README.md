# Per-phase engine benchmarks

Measures the cost of Hiyal's engine phases on synthetic worlds, separately
from any particular experiment, so that an optimization can be attributed to
the engine and checked for regressions. This crate is standalone: it is
excluded from the Hiyal workspace, has its own `Cargo.lock`, and is never
built by PR CI. For profiling a specific experiment, see
[docs/profiling.md](../../docs/profiling.md).

## World shapes

| Shape | Kind | Size parameter | Behaviour |
|---|---|---|---|
| `constant` | `phase-bench.constant.v1` | state slots | Updated in place; size never changes |
| `growing` | `phase-bench.growing.v1` | history records at genesis | One record appended per event, like a world that stores its own trace |
| `keyed` | `phase-bench.keyed.v1` | tables of 8 string-keyed counters | Many small maps, like per-agent or per-stream bookkeeping |

Each world's next action depends only on the step and actor, so every
revision commits the same deterministic history.

## Phases

| Phase | `phase-report` name | Criterion group | Unit | Notes |
|---|---|---|---|---|
| Serialization | `serialize` | `serialize` | call | Stable JSON bytes; `world_bytes` recorded |
| Hash (serialize + SHA-256) | `digest` | `digest` | call | What Hiyal does per state hash |
| SHA-256 alone | `sha256` | `sha256` | call | Over the serialized bytes; MiB/s recorded |
| Action application | `apply` | `apply` | action | The world's reducer alone (world code, not Hiyal): one action on a fresh clone of the reported world, clone and drop untimed |
| Staging and hashing | `stage` | `stage` | event | Clone, apply, hash: the staging work, rebuilt from public API |
| Journal append, buffered | `commit_buffered` | `commit_buffered` | event | Full committed step, `Durability::AtCheckpoint`, first 50 events |
| Journal append, fsynced | `commit_fsync` | `commit_fsync` | event | Full committed step, `Durability::EveryEvent`, the same first 50 events |
| Journal append, long horizon | `commit_buffered_long` | — | event | As `commit_buffered`, over the workload's full event count |
| Replay | `replay_full` | `replay_full` | event | `inspect_read_only`: every event re-applied, every hash re-checked |
| Replay, cold process | `replay_full_cold_process` | — | event | Same, in a fresh process per sample (see below) |
| Inspection, memo miss and hit | `inspect_cached_miss`, `inspect_cached_hit` | `inspect_cached_hit` | call | `inspect_read_only_cached` (feature `replay-cache`) |
| Trusted-snapshot recovery | `open_trusted_snapshot` | `open_trusted_snapshot` | call | `Store::open_trusted` |

**Durability is compared at one history length.** `commit_buffered` and
`commit_fsync` commit the same 50 events from genesis, so their difference is
the cost of durability alone. `commit_buffered_long` shows how commit cost
scales with history (it grows for the `growing` shape); compare it with
`commit_buffered`, never with `commit_fsync`. Criterion's commit groups time
one event from genesis for both durabilities.

**`apply` times one action at a time.** Each timed action is applied to a
fresh clone of the reported world straight after cloning, as Hiyal's staging
does, so it acts on exactly `world_bytes`. The median cost of an empty timer
call is subtracted and recorded, so reducers of a few nanoseconds are at the
limit of resolution. Clones are deliberately not prepared in batches: that
changes the heap layout, which decides whether a `growing` world's push can
extend its buffer in place, and moved its median by up to 100x. Allocation
counts cover 100 such actions, the same work as the timings.

**Trusted-snapshot recovery is not verification.** `open_trusted`
deliberately skips re-verifying events before the snapshot. It is measured to
show the cost of resuming, and must never be compared with `replay_full` as if
the two did the same work.

`phase-report` also records, for one further run of each phase, the number of
allocations, bytes allocated and peak live heap above the phase's starting
heap. It uses a counting global allocator, so these figures exclude the
allocator's own overhead. The cold-process phase runs in a child process, so it
has timings only.

## Running

```sh
cd benchmarks/phase
cargo build --locked --release --bin phase-report

# Fixed baseline workloads, 5 repetitions, warm only:
./target/release/phase-report run --out report.json
# Add cold-process replay; drop the page cache first when permitted:
./target/release/phase-report run --cold --drop-caches --out report.json
# Larger worlds too:
./target/release/phase-report run --suite full --out report.json
# Median-per-phase table from one or more reports:
./target/release/phase-report summarize report.json

# Criterion statistics (warm), all groups or a regex filter:
cargo bench --locked --bench phases
cargo bench --locked --bench phases -- 'replay_full/'
```

`HIYAL_PHASE_BENCH_DIR` sets where scratch trials are written (default: the
system temporary directory), for example to compare a disk with a tmpfs.

On GitHub, run **Actions → Phase benchmarks (manual) → Run workflow**. It
builds with Rust 1.89.0, runs `phase-report` (warm and cold, page cache
dropped), optionally Criterion and a comparison, and uploads everything as an
artifact with the machine record.

## Cold and warm

- **Warm:** many iterations in one process (Criterion, and every
  `phase-report` phase except the cold one). Files are in the page cache, the
  allocator is warmed up, and the replay memo holds whatever that phase says.
  `inspect_cached_miss` clears the memo before each sample;
  `inspect_cached_hit` measures a memo that already holds the journal.
- **Cold process:** `replay_full_cold_process` runs each sample in a new
  process (`phase-report cold-replay`). The replay memo and allocator start
  empty. With `--drop-caches`, `phase-report` first runs `sync` and writes
  `3` to `/proc/sys/vm/drop_caches`, directly as root or through `sudo -n`.
  Each sample records whether that succeeded (`page_cache: dropped`) or not
  (`kept: …`). A cold sample whose page cache was kept is still a cold
  process, but not a cold disk.

Keep the two apart when reporting: never mix warm and cold samples in one figure.

## Provenance

Every report records:

- **Engine build:** Hiyal's `BuildProvenance`, which gives the engine version,
  rustc, target, source commit, dirty flag, and the SHA-256 of Hiyal's
  `Cargo.lock`.
- **Benchmark build:** the SHA-256 of this crate's `Cargo.lock` as compiled
  into the binary (embedded at build time, so a later checkout change cannot
  alter it), and its enabled features.
- **Workloads:** the suite version, repetition count, world shape, size, event
  count, world bytes and journal bytes. Durability is the phase name.
- **Machine:** CPU model, logical CPUs and selected CPU flags, plus
  `sha2_hardware_backend`. The kernel and a timestamp are also recorded.
- **Samples:** every raw sample, not only the median.

**SHA-NI.** A `sha_ni` CPU flag only says what the CPU offers.
`sha2_hardware_backend` repeats the runtime test that `sha2` 0.10 uses to
choose its hardware backend (`sha`, `sse2`, `ssse3` and `sse4.1` all
detected). The manual workflow also counts `sha256rnds2` instructions in the
built binary, which shows the hardware path is compiled in. Together they show
which path `sha2` takes. They are not an instruction trace; for that, profile
the `sha256` phase with `perf` or a similar tool on the machine in question.
The `sha256` phase's MiB/s is a useful cross-check: hardware SHA-256 is
typically several times faster than the software fallback.

## Baselines and comparing revisions

The workloads in `BASELINE_V1` (`src/bin/phase-report.rs`) are fixed. Never
change one in place; add a new suite version instead, so old reports stay
comparable.

To compare two revisions, run both on the same machine, alternating:

```sh
benchmarks/phase/compare.sh BASE_REV HEAD_REV [ROUNDS] [OUT_DIR]
PHASE_COLD=1 benchmarks/phase/compare.sh BASE_REV HEAD_REV   # add cold replay
```

The script builds this crate's current benchmark code against each revision,
in temporary git worktrees. It alternates base and head for each round and
writes every raw report plus `summary.md`, which gives the median of medians
per phase. Both builds disable the `replay-cache` feature, so revisions
before #118 compile. Comparing numbers from two different machines or GitHub
runners is not a comparison; rerun both revisions on one machine instead.

## Instruction-count regression gate (not yet enabled)

An instruction-count check on every PR, for example with `iai-callgrind`,
would be more stable than wall-clock timing. It is deliberately not part of
this crate yet. It needs valgrind and a version-matched runner binary
installed in CI, and its runner cost is unmeasured. Its counts still depend
on the toolchain and compiler flags. They also cannot see hardware SHA-256:
valgrind hides the SHA extensions, so `sha2` always takes its software path
under it (checked with valgrind 3.22). Add it only after measuring that cost, with a pinned toolchain,
fixed inputs and a generous threshold, as [docs/ci-budget.md](../../docs/ci-budget.md)
requires.
