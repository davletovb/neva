# GitHub Actions: low-spend policy for Hiyal

**Effective:** 2026-10-19. Applies only to the Rust rewrite. The historical
multi-job Python workflow is archived under `hiyal-python/.github/workflows/`
and remains inactive; the archived Dependabot file is not active. Do not
restore those jobs just to verify Rust changes.

## Trigger budget

| Event | Runs | Rationale |
| --- | --- | --- |
| A code-affecting PR first opens ready for review, reopens, or is marked ready | **One small Rust job** | Minimal merge-time confidence |
| Every subsequent coding-agent commit / PR synchronization | **No run** | Avoid dozens or hundreds of repeated runs |
| Draft PR | **No runner job** | Work-in-progress need not consume minutes |
| Documentation-only PR | **No run** | Path-filtered; no compilation needed |
| Every push to `main` | **No run** | Do not pay to repeat a merge check |
| Manual `workflow_dispatch` | **One Rust job** | Deliberate final pre-merge verification or debugging |
| Published version tag matching `v*` (push) | **One job with Clippy** | Release boundary gets additional lint |
| Nightly/scheduled benchmarks, coverage, Python compatibility matrix | **No run** | No recurring default expense |

**Important tradeoff:** a PR check from initial opening becomes stale after
later commits because `synchronize` is intentionally **excluded**. Before a
merge, verify the **latest commit SHA**. If the workflow exists on the default
branch, use **Actions → Rust smoke → Run workflow** and select the PR branch.
For early PRs before the workflow is on default, toggle **Convert to draft →
Ready for review** once on the final revision; the second transition triggers
a fresh run for the current merge ref. If checks or changed files are unclear,
prefer the manual local command `cargo fmt --all -- --check &&
cargo test --workspace && cargo run -p hiyal-custom-world-example`.
Never treat a green check from an earlier SHA as validation of a newer commit.
Do not make this workflow a required continuously up-to-date PR check without
first changing its trigger strategy.

## Job cost controls

- One **standard Ubuntu** runner; no platform, toolchain or feature matrix.
- **10-minute job timeout**. Stale concurrent PR/manual runs are cancelled using the same branch name
  as their group key (`github.head_ref || github.ref_name`).
- One Rust-only job: formatting, unit/feature checks (including opt-in loopback HTTP), registered CLI run/resume and standalone consumer smoke; no Python benchmark jobs.
- Clippy only for manual extended verification and version-tag releases.
- No automatic benchmarks, LLM/network smoke, deprecated Python dependencies,
  heavy optional integrations, docs builds, coverage, artifacts or cache churn.
- Read-only `GITHUB_TOKEN`, no secrets, and no privileged `pull_request_target`.
- No `schedule` and no auto-trigger from a push to `main`; benchmarks run
  manually on designated hardware at meaningful architecture/release gates.
- A Cargo lockfile is not yet checked in. The workflow enforces `--locked`
  **when a lockfile exists**. Commit the lockfile as soon as Rust dependencies
  can be resolved; dependency resolution on each run is not ideal.
- Review median run durations and action activity after enough samples; decide
  whether even the one PR lifecycle run is justified. Avoid making unmeasured
  claims about minutes saved.

## Billing detail and safety

As of 2026-10-19, `davletovb/hiyal` is a **public repository**. GitHub's
documented policy is that standard hosted-runner compute is free on public
repositories; larger runners can still cost money. Charges across **other
private repositories** are not explained away by this fact. Do not generalize
the public-repo exemption to all of the owner's projects. See:
https://docs.github.com/en/billing/concepts/product-billing/github-actions

PRs cannot claim verified green CI until the new workflow has actually run.
No CI can substitute for the independent observer-on/off and checkpoint
measurements required by [R0](r0-verification.md). Those are manual, targeted
scientific measurements and must have recorded hardware and raw samples.
