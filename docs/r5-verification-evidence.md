# R4/R5 non-benchmark release qualification — 20 October 2026

This log captures actual GitHub-run native verification **without inventing a
cross-platform single-SHA result**. PR: [#112](https://github.com/davletovb/hiyal/pull/112).

## Native distribution evidence

| Target | Run | Verified source commit | Outcome | Uploaded artifact digest (ZIP, SHA-256) |
| --- | --- | --- | --- | --- |
| Linux x86_64 | [39372929619](https://github.com/davletovb/hiyal/actions/runs/39372929619) | `a1b92c74ec87ac2ae2790f6524141dd5a1f5f9d5` | workspace tests, release build, CLI run/resume/export and custom model passed | `86f2783e381b6fd7476b11df913581d7ccf1020e7db952fb28c9d5a3ca81740f` |
| macOS arm64 | [39372929619](https://github.com/davletovb/hiyal/actions/runs/39372929619) | `a1b92c74ec87ac2ae2790f6524141dd5a1f5f9d5` | workspace tests, release build, CLI and custom model passed | `d0c7e76159e0e320bdf48ba9e2d78c2248472805143c6837989c7430f17d3509` |
| Windows x86_64 MSVC | [39373486868](https://github.com/davletovb/hiyal/actions/runs/39373486868) | `6416a6473af2e379a25a995a0a762804b9026a5c` | workspace tests, release build, CLI and custom model passed **after fixing fork evidence flush via writable handle** | `da85a41d13a1a52f00242365a044402619f7d3114407471bf4294bb9f3df0f58` |

The initial full-platform matrix discovered five Windows fork tests failing
with `Access is denied`; diagnosis narrowed them to read-only evidence-file
fsync during fork creation. [Follow-up Windows-only qualification](https://github.com/davletovb/hiyal/actions/runs/39373486868)
passed. The Linux/macOS qualifications are at a preceding SHA; we do not
claim that a final merged SHA has been tested simultaneously on all platforms.
All artifacts are uploaded by GitHub Actions with provenance `BUILD.txt`,
a local binary `SHA256SUMS` manifest and GitHub-recorded archive digest.
This establishes independently verifiable source/CI lineage, **not** signed
or bit-for-bit reproducible builds across compilers and operating systems.

Normal PR CI remains one Linux smoke job. The three-OS workflow has been
restored to **manual or version-tag triggers**, not per-push or per-PR matrix
cost. Users may manually run a final same-SHA qualification from the
latest release candidate before public publication.

## Adversarial checks and data policy

- Event/metadata limits: 32 MiB per committed event and manifest/snapshot;
  bounded streamed authoritative replay, no automatic corrupt committed
  record repair, deterministic torn-tail truncation under a writer lease.
- Effect receipt replay: bounded streamed reader, keyed identity/hashes,
  32 MiB per record, 1 GiB total cap, pre-dispatch reservation of a compact
  failure receipt per new effect, and compact receipts (billing status kept)
  when oversized results occur.
- Rust unit/integration tests: integrity mutation matrix, malformed metadata,
  versioned snapshots, crash recovery, fork ancestry and branch isolation,
  writer lease concurrency, deterministic RNG ordering and memory invariants.
- Privacy: redacted JSONL/CSV outputs are separate from full authoritative
  world/event/effect ledgers; never publicly upload trial sources with raw
  prompts, world state or user-level memory without explicit sanitization.
- Rollback steps are specified in [R5 release guide](r5-release.md).

## Archived Python removal and excluded benchmark

The **159 legacy files** (approximately 1.76 MB) under `hiyal-python/`,
including its tests, CI, Dependabot and two notebooks, were removed from Rust
main and preserved at exact source commit
[`d056e228`](https://github.com/davletovb/hiyal/tree/d056e228dab3d42162f1568d91b2595b17b8e8a7/hiyal-python)
and [archive/hiyal-python-pre-r5](https://github.com/davletovb/hiyal/tree/archive/hiyal-python-pre-r5).
Git tree examination found no committed historical CSV/JSONL/Parquet/database
research datasets requiring conversion. The historical notebooks are
preserved in the archive rather than silently discarded. Manual historical
benchmarks read an explicitly set `HIYAL_LEGACY_SOURCE` from a separate
archive worktree. No Python runtime, tests or CI run as part of Hiyal's
native build.

## Public package name and residual external gates

`hiyal` was selected and documented as the development name. It remains
`publish = false`; there is **no** verified crates.io owner/registration,
legal trademark clearance or authorized publisher credential in this
implementation. The separate [name-gate assessment](r5-name-gate.md)
prevents conflating a name decision with ownership. No crate was publicly
published or reserved, and no benchmark numbers were collected.

The only R0/R5 *benchmark* acceptance requirements remain intentionally
deferred by request. Public package publication/rights and optional final
single-SHA, signed or bit-for-bit release attestation require external
authorization/evidence rather than fabricated checkboxes.
