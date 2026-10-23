# Issue #117 prototype: redundant replay, staging and RNG work (2026-10-23)

Exploratory engineering measurement, not a controlled benchmark. One machine,
one seed, same-session alternating runs.

## What changed

Commit `aeb79f28ff2067b9b8d04b9f6a2d669ed583e53e` against baseline
`e38419f857afbb53428284a312b0eb926a2320e2`. No committed byte, hash or
verification rule changes.

1. `Store::inspect_read_only_cached`: in-process memo of successful full
   read-only replays, keyed by SHA-256 of the exact manifest, snapshot and
   log bytes replayed (plus world/action types). Used by `verify_fork`,
   `fork_at_head` and the paired helpers.
2. `Store::append_staged`: trial steps commit the world they already staged
   and hash it once, instead of the store cloning, re-applying and re-hashing.
3. RNG: per-stream ChaCha20 seed (a pure SHA-256 of domain, root seed and
   name) is cached instead of re-derived on every draw.

The consumer, social-emergence-lab experiment 002, also switched its
`inspect_read_only` calls in `reputation_pilot.rs` and
`reputation_prior_pair.rs` to the cached variant.

## Setup

- Linux x86-64, Intel Xeon @ 2.80GHz, 4 cores, no SHA-NI (software SHA-256).
- Rust 1.89.0, `--release`.
- Workload: `reputation-paired-prior run <dir> 72100 N d tau T`, then
  `verify <dir>` (two societies, parent plus four forks each, with the
  lab's own reference replay during verify).
- Runs alternated baseline/new in each repetition; output on local disk.

| Case | N, d, tau, T | Journal lines (incl. ancestry copies) | Journal bytes |
|---|---|---|---|
| A | 10, 3, 40, 120 | 1,040 | 2,549,549 |
| B | 20, 4, 80, 240 | 2,080 | 7,643,506 |
| C | 20, 4, 200, 600 | 5,200 | 21,207,185 |

## Wall-clock (median [min–max])

| Case | Step | n | Baseline | New | Speedup |
|---|---|---|---|---|---|
| A | run | 5 | 3.37 s [3.12–3.64] | 1.47 s [1.41–1.70] | 2.29× |
| A | verify | 5 | 0.93 s [0.82–1.18] | 0.50 s [0.47–0.52] | 1.88× |
| B | run | 5 | 20.89 s [20.47–21.74] | 7.76 s [7.45–8.08] | 2.69× |
| B | verify | 5 | 6.63 s [6.57–7.30] | 2.75 s [2.59–2.84] | 2.42× |
| C | run | 3 | 55.40 s [55.21–57.78] | 21.40 s [20.88–21.67] | 2.59× |
| C | verify | 3 | 17.42 s [17.32–17.87] | 8.24 s [8.14–8.39] | 2.11× |

## Instruction counts (callgrind, case A `run`)

| Function (inclusive) | Baseline | New |
|---|---|---|
| Program total | 19.62 G | 7.22 G (−63%) |
| `Store::replay_history` | 11.68 G | 2.76 G |
| `Trial::step` | 4.44 G | 2.55 G |
| store append (`append_typed` / `append_inner`) | 2.42 G | 0.54 G |
| `Streams::draw_u64` | 2.91 G | 0.64 G |
| `experiment::verify_setup` | 1.94 G | 0.47 G |
| `storage::digest` | 11.95 G | 3.83 G |

## Equivalence checks

- Case B: all 2,080 events match baseline output field-for-field after
  removing identity fields (actions, state hashes, schedules, RNG positions).
  The identity fields are trial ids and chain hashes, which differ only
  because build provenance is hashed into the genesis. `paired-report.json`
  is equal apart from `hiyal_revision`.
- The new binary verifies the baseline's evidence and the baseline binary
  verifies the new evidence.
- The original experiment 002 V0 archive (Hiyal `4bb53ac`, SHA256SUMS
  `80666d6f…d8ec`) passes `scripts/verify-002.sh` with the new engine.
- Hiyal: workspace tests, the `serde_json/preserve_order` variant,
  clippy `-D warnings` and fmt all pass. A new test covers cached
  inspection after appends, log tampering and snapshot tampering.

## Not addressed

Per-event world hashing still dominates what remains (`digest` is 53% of the
new profile). That includes `serde_json::Value` construction and, on this
CPU, software SHA-256. The O(T²) growth from re-hashing the full trace is
also untouched.
