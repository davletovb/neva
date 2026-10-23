# Issue #117 idea 4: direct canonical JSON writer (2026-10-23)

Exploratory engineering measurement, not a controlled benchmark. Same
machine, seed and workload as
[2026-10-23-issue-117-redundant-work.md](2026-10-23-issue-117-redundant-work.md).

## What changed

`stable_bytes`, `digest` and the engine's `action_digest` now use
`src/canonical.rs`. Before, they built a `serde_json::Value` tree and then
walked it to emit recursively key-sorted JSON.

- The new serializer writes each object's entries straight into one
  buffer, then reorders them in place only when serde emitted them out of
  order. Keys sort byte-wise, and for a repeated key the last value wins,
  as in a `Value` map.
- Scalars and map keys are still converted by serde_json's own `Value`
  serializer. Number, float, out-of-range integer and map-key rules are
  therefore unchanged by construction.
- The hashed bytes are intended to be identical, and the checks below
  found no difference. No schema or profile version changes.

## Equivalence checks

- **Differential tests** (`canonical::tests`) compare the new writer with
  the previous `Value` route, which is kept test-only as the reference.
  Coverage:
  - 20,000 random JSON values with escape, control, non-ASCII and astral
    characters in keys and strings, and extreme floats including `-0.0`;
  - maps keyed by integers, u128, strings, bool, char, newtypes, tuples,
    unit and byte vectors;
  - every serde enum representation, flatten with a duplicate key,
    `skip_serializing_if`, f32, NaN/inf, and out-of-range i128/u128;
  - engine records (`Streams`, `BuildProvenance`).

  Two deliberate mutations, skipping the sort and keeping the first
  duplicate, both fail these tests.
- **Experiment 002**, N=20, d=4, τ=80, T=240: all 2,080 events match the
  original baseline field for field after removing identity fields.
- **Cross-verification:** this build verifies evidence written by `e38419f`
  and by `main` (`55d30ea`), and both of those builds verify evidence
  written by this one. Every state and chain hash is therefore recomputed
  by both serializers over each other's bytes.
- **V0 archive:** the experiment 002 V0 raw-evidence archive (Hiyal
  `4bb53ac`) passes `scripts/verify-002.sh` with this engine.
- **Test suites:** workspace tests, the `serde_json/preserve_order` and
  `live-http` variants, clippy `-D warnings` and fmt all pass.

## Wall-clock (median [min–max], 3-way rotation per repetition)

`main` is `55d30ea`: ideas 1–3, merged in #118.

| Case | Step | n | Baseline `e38419f` | `main` | This change | vs `main` | vs baseline |
|---|---|---|---|---|---|---|---|
| A (10,3,40,120) | run | 5 | 3.38 s [3.35–3.67] | 1.70 s [1.47–1.82] | 1.43 s [1.31–1.53] | 1.19× | 2.37× |
| A | verify | 5 | 0.93 s [0.87–1.09] | 0.55 s [0.46–0.72] | 0.43 s [0.38–0.59] | 1.30× | 2.17× |
| B (20,4,80,240) | run | 5 | 22.68 s [21.01–24.06] | 8.62 s [7.87–9.11] | 7.27 s [6.95–7.56] | 1.19× | 3.12× |
| B | verify | 5 | 6.79 s [6.75–7.17] | 2.93 s [2.82–3.62] | 2.56 s [2.39–2.67] | 1.14× | 2.65× |
| C (20,4,200,600) | run | 3 | 61.16 s [60.75–61.96] | 23.15 s [21.77–24.23] | 17.96 s [16.96–19.52] | 1.29× | 3.40× |
| C | verify | 3 | 19.37 s [18.24–19.83] | 8.99 s [8.96–9.13] | 6.81 s [6.64–7.46] | 1.32× | 2.84× |

Case A ranges overlap between `main` and this change, so its gain is less
certain than B's and C's.

## Instruction counts (callgrind, case A `run`)

| Function (inclusive) | `main` | This change |
|---|---|---|
| Program total | 7.54 G | 6.25 G (−17%) |
| `storage::digest` | 4.06 G | 2.99 G |
| ⤷ `core::stable_json_bytes` (old route) | 1.28 G | — |
| `sha2::sha256::compress256` | 2.75 G | 2.75 G |
| `__rust_alloc` | 0.74 G | 0.31 G |

Serialization inside `digest` fell from about 1.3 G to about 0.25 G
instructions. What remains of `digest` is SHA-256 compression. This CPU
has no SHA-NI, so here that is software SHA at 44% of the run. On SHA-NI
hardware it should cost several times less.

## Not addressed

The O(T²) cost of re-hashing the whole growing trace at every step (idea 5)
is untouched. With serialization overhead mostly gone, total bytes hashed is
now what determines `digest` cost. Reducing it needs a new versioned world
kind or commitment scheme.
