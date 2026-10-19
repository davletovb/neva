# R2: reproducible research runner, paired interventions, and CSV

Issue: [#103](https://github.com/davletovb/hiyal/issues/103).
Scope: a pure-Rust, single-machine research workflow. No Python benchmarks,
embedded scripting runtime, Parquet dependency, or scheduled CI.

## Quickstart

```bash
# Cargo.lock is committed; always use --locked for reproducible runs.
cargo test --workspace --locked
cargo run --locked -p hiyal -- sweep examples/sweep.toml /tmp/r2-study
cargo run --locked -p hiyal -- sweep examples/sweep.toml /tmp/r2-study
cat /tmp/r2-study/results.csv

# Create a stable prefix and fork into control and treatment.
cargo run --locked -p hiyal -- run examples/counter.toml /tmp/r2-parent
cargo run --locked -p hiyal -- fork /tmp/r2-parent /tmp/r2-pair 3 12
cargo run --locked -p hiyal -- fork /tmp/r2-parent /tmp/r2-pair 3 12
```

The second sweep/fork command should have no additional committed actions.
Each trial has its own writer/effect lock and isolated manifest, event log,
snapshot, and effects. Do not run concurrent *processes* against the same
trial directory; the runner bounds concurrency **between** independent jobs.

### Sweep parameters

`examples/sweep.toml` sets a safe study name, base seed, a list of unique
parameter values, replicates, actor count, target attempts, maximum parallel
jobs, and random/round-robin scheduling. Jobs are deterministic and sorted by
stable SHA-256-derived identity. The per-replicate seed is fixed, independent
of worker completion order. Job directories use their exact hash IDs.
Scientific settings are frozen in `sweep-plan.json`; attempts to restart with
changed seeds, parameters, actors, step targets or scheduler fail. The
operational `max_parallel` worker limit may change between restarts without
altering a trial's scientific identity or results.

Resource limits: maximum 512 jobs, maximum 16 parallel jobs, maximum 1,000
actors/job and 100,000 attempts/job. The worker pool drains already running
jobs after an error, without scheduling further work.

`results.csv` columns are `job_id,trial_id,parameter,replicate,seed,attempts,
total,chain_head`. All numeric IDs and results are in decimal text;
`chain_head` permits byte-level deterministic comparison. Each trial has
`metrics.csv` columns `trial_id,sequence,logical_step,agent_id,metric,value`.
The metrics are emitted **after** each accepted world event. The metrics file
is explicitly non-authoritative and is reconstructed from the verified
committed event log on restart, then appended incrementally during execution.
This avoids using a possibly partially written CSV as simulation state.

The generic Rust interface is `Sweep::execute` (bounded parallel jobs) or
`Sweep::run_counter` (built-in deterministic society). Custom experiments
can compile their own runner against Hiyal without editing the library.

## Forks and common random numbers

`research::fork_at_head` accepts a validated parent trial and creates a new
child trial **at the last fully committed event**. `paired_forks` creates
control and treatment branches with independently hashed IDs, shared
post-prefix scheduler positions, and the same named ChaCha20 random streams.
`Streams::paired_branch` records the inherited stream seed domain
separately from the child trial ID; neither child shares mutable state,
events or effect receipts with the other.

The intervention is applied **once** to the copied child genesis. It is
attributed by `fork.json` with an intervention name, arm, parameters,
parent trial/head/sequence, exact before/after world hashes, and source-file
digests. The `ancestry/` directory contains immutable copies of the
parent's manifest and prefix log. `verify_fork` independently reconstructs
the captured ancestor, validates provenance hashes and the child's initial
world/scheduler/RNG boundary. `resume_fork` performs verification before
resuming and never reapplies the intervention. The fork remains portable even
if the original parent directory is deleted.

Important limits: currently forks operate at the **current committed head**
rather than an arbitrary historical offset. The attribution data records
parameters and before/after world hashes; it is **not a signed, independently
executable intervention program**. The fork verifier validates content
integrity against its stored digests, not cryptographic authorship or
empirical causal identification. Branch effects after the fork use distinct
trial/call identities. Common random numbers are preserved when both
branches consume the same named streams in comparable order; divergent
policies can cause their draws to cease corresponding.

The CLI `fork` demonstrates an additive counter intervention. For more
realistic sociological models, compile independent `World`, `Policy` and
intervention types in a consumer crate and invoke the generic fork APIs.

## Scientific reference cases and invariants

`src/toy.rs` contains simple, explicitly checkable model references:

- Uniform random selection: rejection-sampled, trial-owned draws; seeded
  frequency sanity check at five symmetric categories.
- Conservation: ring wealth transfers are checked against an **independently
  written vector-loop reference**, not simply the same world reducer twice.
- Symmetry: equal initial balances and complete round-robin cycles return
  every actor to equal wealth, without any hidden scheduler privilege.
- Preferential attachment: a cumulative-weight sampler is compared to an
  independent expanded-urn implementation for identical named draws.
- Paired shock: two branches share verified ancestry and stream identity
  but have different intervention-attributed initial states. Test verifies
  restart, no cross-branch mutation and tamper rejection.

These are software correctness checks, **not empirical model validation**.
D1 applies to integer-state, pinned-rand-ChaCha, fixed-toolchain workflows.
Real asynchronous LLM sampling remains D2.

## Scope of the milestone

R2 deliberately does not implement distributed runners, external policy
processes, arbitrary historical forks, provider-side exactly-once billing,
causal estimators, Parquet, dashboards, or live-vs-simulated economic claims.
Parquet can be added when an actual analytical use case needs a columnar
format; CSV and verified JSONL already remain interoperable with R, Julia,
Python and DuckDB without a Python runtime in Hiyal.

## R2 review hardening

- Fork publication uses a unique sibling staging directory. The public child appears only after valid ancestry, attribution and journals are confirmed. A killed process may leave an ignorable pending folder. The CLI can safely recreate one valid, unstarted arm after an interrupted paired publication. It never auto-deletes progressed or mismatched arms.
- On restart, the CLI checks parent identity, committed head/sequence, roles, intervention and parameters, rejecting another source or swapped arms.
- The generic paired-fork API accepts a caller-selected model backend. The CLI deliberately uses the scripted backend for its fixed reference counter.
- Sweeps exclude counter-overflow configurations before running and distinguish accepted logical steps from attempt counts. Worker parallelism can change without changing scientific identity.
- Observer/CSV projection failures are not reported as successful runs; repeat execution rebuilds CSV from verified events.
- Shared atomic writing now handles plan, result, metrics and fork metadata. Rebuilt CSV is streamed rather than buffered. Metric append no longer calls fsync on every committed step; world events are still durable. Paired checks compare previously verified fork metadata without redundant full-ancestry replay.

- Fork IDs incorporate a canonical hash of the committed parent head, intervention/arm parameters and before/after state. Identical intervention names applied at different boundaries cannot alias in trial/receipt identity.
- Generic Sweep::execute continues to accept arbitrary u64 parameter categories. Overflow rejection is intentionally specific to Sweep::run_counter, which performs checked integer addition.

- Fork placement checks canonicalized source and prospective child paths, including existing symlink ancestors and lexical aliases. A path that resolves under its own source ancestry is rejected before creating a child.
