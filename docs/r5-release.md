# R5 — release qualification and rollback

This is the release **procedure**, not evidence that a production release
or crates.io publication has already happened. Hiyal remains at 0.1.0 and
`publish = false` until its name, security, provenance, and compatibility
checks are complete.

## Source and binary qualification

On an immutable candidate commit:

1. Verify `cargo fmt --all -- --check`, `cargo test --workspace --locked`,
   `cargo test -p hiyal --locked --features serde_json/preserve_order`,
   `cargo test -p hiyal --locked --features live-http`,
   and `cargo clippy --workspace --all-targets -- -D warnings` (manual).
2. Run the built-in and independent custom-model executable in new directories;
   execute run, resume, validate/inspect and export without any Python runtime,
   model API key or remote provider.
3. Launch **Native distribution** via a manual GitHub Actions dispatch against
   the exact candidate branch/commit. It runs on Linux x86_64, macOS arm64 and
   Windows x86_64, each with workspace tests and release binary smoke. Do not
   make ordinary PRs run three OS matrices; there is a separate, inexpensive
   Linux PR gate. A `v*` tag also triggers the distribution workflow.
4. Download all three uploaded artifacts. Check `BUILD.txt` source SHA,
   target/triple/toolchain and `SHA256SUMS` using `sha256sum -c` or `shasum
   -a 256 -c`. Checksums detect transfer damage; without signatures or
   attestations they are **not** independent authenticity proofs.
5. Record the run URLs, source commit, tested platform, toolchain identity,
   SHA256 manifests, benchmark hardware and any known failures under a
   release evidence issue. Don't label an unrun platform green.
6. Check crates.io package ownership and potential collisions manually, and
   run `cargo package --locked --list` before lifting `publish = false`.
   No reservation or public package existence is claimed by this RFC.
   Version/publish only after review and an explicit publication decision.

For release-channel reproducibility, commit `Cargo.lock` and fix source
revisions/toolchains, compare two builds on identical machines, and record
hashes. Full bit-for-bit reproducibility is not yet demonstrated: compiler,
linker and build environment can affect hashes. The three-OS distribution
workflow preserves **source provenance**, not a false cross-OS equal binary
hash claim. No automatic publishing or registry secrets in CI.

## Portable contracts

- `storage::SCHEMA = 1`: `manifest.json`, hash-chained `events.jsonl`,
  `snapshot.json`; 64-bit event sequence, logical-step and agent IDs in
  JSON use decimal strings. Nested maps are recursively sorted for hashing.
  This profile is **not RFC 8785** or cross-CPU floating-point reproducibility.
- `observe::OBSERVATION_SCHEMA = 1`: non-authoritative redacted
  `observations.jsonl` and the CLI's `export` output. No raw prompts,
  tool arguments, full world, rejection reasons or model responses.
- `memory::MEMORY_SCHEMA = 1`: application-owned bounded memory. Embedding
  memory in world state also exposes it in authoritative logs and snapshots;
  never treat a trial directory as safe public export.
- Manifest and snapshots are limited to 32 MiB each; committed event lines
  (including delimiter) to 32 MiB. Replay streams committed lines in bounded
  memory. These are explicit portable storage limits; a previously unbounded
  custom world larger than this requires conversion or a separately reviewed
  new contract. Effect-journal retention remains an external quota concern.
- `effects.jsonl` contains potentially sensitive request/response receipts
  and still requires restricted filesystem access, retention policy and an
  external provider billing/reconciliation plan.

## Rollback and disaster recovery

1. Stop all trial writers and external effect execution. Never clear a lock
   while another writer is alive.
2. Preserve a byte-for-byte copy of the original trial directory (including
   manifests, receipts, full event logs, snapshots and fork ancestry) before
   attempting any repair. Restrict permissions and avoid public upload.
3. Validate with the *same or newer compatible schema-aware binary*; a
   version that cannot understand a schema must refuse it rather than guess.
   `inspect`/`validate` do not repair, while `resume` under the exclusive
   writer lease can truncate an uncommitted tail only.
4. If a committed record, snapshot or receipt is corrupt, quarantine and
   restore a separately verified backup. Do **not** delete or rehash damaged
   accepted events, invent provider receipts, or replay live external actions
   to patch history. Repair through a documented new trial/fork if needed.
5. To roll back software, redeploy the previous artifact **only if** it can
   read the on-disk schema. Otherwise restore a compatible backup and treat
   future-schema data as read-only, never downgrade in place.
6. If a failed external effect may have been billed, reconcile with the
   provider before retrying; Hiyal cannot guarantee exactly-once remote work.

## Gate ledger

Passing the Linux PR job means essential local behavior is checked; it is
**not** a signed stable release or a full security audit. R5 remains open until
measured benchmark evidence, historical Python retirement criteria, verified
cross-platform artifacts, and a clean registry publication decision are linked.
