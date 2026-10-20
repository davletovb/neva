# R6: storage scale and deterministic research API (Issue #103 review)

This is a **corrective milestone**, not an assertion that every social simulation
feature is finished. The review identified a real mismatch between a small
counter benchmark and growing-world event-sourcing costs.

## Decisions and implemented interfaces

| Review | R6 change | Important boundary |
| --- | --- | --- |
| Repeated full world in every event | New accepted events contain action, state hash and replay RNG state; legacy events with embedded world still deserialize and hash the same way | World reducer must be replayable; per-event state hashing and schedule serialization still incur work |
| Full-history resume | `Store::open_trusted`, `Trial::resume_trusted` and CLI `--trusted-snapshot` seek to an explicit snapshot boundary | **Opt-in trust**: verifies the suffix but does **not** detect prior-log tampering. `open`, `inspect`, `validate` always independently replay all history |
| fsync every step | `Durability::EveryEvent` (default) or `Durability::AtCheckpoint`, set using `Store::set_durability` and TOML `durability = "at_checkpoint"`; resume optionally accepts `--checkpoint-durability` | Crash before checkpoint can lose accepted events; source of truth after restart is what actually reached disk, not a promised number of attempts |
| Crash marker leases | OS-backed `std::fs::File::try_lock`; requires Rust 1.89+ | Marker files remain on disk. **Never unlink** active lock inodes. Lease releases after exit, signal or kernel process death |
| Randomness in user policies/world | Default-compatible `propose_with_streams`, `requests_with_streams`, `decide_with_streams`, `apply_with_streams`. Stable named streams, pinned ChaCha20 profile, before-reducer RNG checkpoint in accepted events | Policies may use `agent/<id>`, worlds `world`, interventions `intervention/<id>`. Scheduler stream is engine-only. Replay checks reducer draws and committed RNG positions; no promise of byte-identical unversioned provider sampling |
| Multi-agent effects | `Trial::run_round(max_actors)` freezes per-agent observations, launches model/tool effects in bounded concurrent waves, and commits accepted/rejected actions in deterministic scheduler order | This is a **two-phase deterministic round**, not simultaneous conflict-free world mutation. For stochastic policy draws use `step` until a formal round RNG contract is verified |
| Trial budget reset | `GovernedBackend::recover_admission` under both trial/effect writer leases; durable `trial-budget.json` fixes declared pricing/cap and reconstructs conservative remaining reservations from keyed receipts | Budget changes between processes fail closed; historical live trials with receipts but no cap sidecar fail closed. Provider billing is uncertain; requires reconciliation; no multi-process cross-trial shared wallet |
| Missing executable provenance | Manifest's hash-bound `build` field records engine version, exact compiling rustc, target triple, engine Cargo.lock SHA256 and source SHA **when verifiable** | Consumer's independent lockfile and custom world git SHA remain its responsibility; no signature/bit-for-bit build reproducibility claimed |
| APIs/readers | Internal typed append rejection (`AppendError::TooLarge`), streaming `Store::visit_events`, bounded `read_events` compatibility method | Other `Result<_, String>` APIs and the deprecated-in-concept R0 prototype coexist until a separately versioned breaking API |
| CI | One essential Linux PR/main job, including synchronize pushes and Clippy, existing Rust examples and feature checks | No always-on 3-OS matrix or Python benchmark CI; branch protection rules must be enabled by a repository administrator |

## Usage

```rust
// Durable action-only events (the default) and optional batched durability:
trial.store.set_durability(hiyal::storage::Durability::AtCheckpoint)?;
let events = trial.run_round(8).await?; // model effects may overlap
trial.checkpoint()?;                     // fsync prior journal before checkpoint

// Normal/full validation is the default on resume:
let verified = hiyal::storage::Store::<MyWorld,MyAction>::inspect_read_only(path)?;
// Trusted mode is a separately chosen integrity-performance tradeoff:
let trial = hiyal::experiment::Trial::resume_trusted(path, "world.kind",
    policy, backend, false)?;
```

```toml
# built-in counter example
durability = "at_checkpoint"
```

`hiyal resume <trial> 100 --trusted-snapshot --checkpoint-durability`
can skip the historical authoritative replay, but **building derived observation
and CSV histories from scratch still requires a full scan**. Use
`hiyal validate` before treating a snapshot as trusted and restrict write
access to the snapshot directory. Trust mode is never a substitute for
independent integrity verification.

## Remaining explicit limits

- Process-isolated cancellation (as distinct from cancelling an async future)
  is not implemented. Avoid allowing irreversible tools without sandboxed
  execution and external approval. Never claim exactly-once remote billing.
- Nested multi-step model/tool loops are not implemented, only bounded
  per-attempt effects and two-phase rounds; follow-up effect loops need their
  own deterministic idempotency and retry contract.
- The external-process policy protocol and shared SQLite budget coordinator
  remain **deferred decisions**, not working implementations.
- Rust property/regression checks are not a substitute for a sustained fuzz
  campaign. The optional fuzz harness must be run explicitly; do not claim
  fuzz coverage from CI if it has not executed.
- Artifacts have source/compiler/checksum evidence but are not demonstrated
  bit-for-bit reproducible across machines; public crate publication and
  trademark/ownership clearance also remain separate gates.
- Real scaling/retention measurements with agent-dependent world state, a
  varied transcript window, and checkpoint-economics evaluation must drive
  whether event-sourcing overhead remains appropriate. The original scalar
  counter benchmark cannot answer that question.

See [R5 evidence](r5-verification-evidence.md), [benchmark protocol](../benchmarks/r5-protocol.md)
and [Issue #103](https://github.com/davletovb/hiyal/issues/103).
