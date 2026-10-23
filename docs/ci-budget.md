# GitHub Actions: low-spend policy for Hiyal

**Effective:** 2026-10-19; **reconciled with the workflow files:** 2026-10-23.
Applies only to the Rust rewrite. The historical multi-job Python workflow is
preserved only on
[archive/hiyal-python-pre-r5](https://github.com/davletovb/hiyal/tree/archive/hiyal-python-pre-r5)
and remains inactive; the archived Dependabot file is not active. Do not
restore those jobs just to verify Rust changes.

This page describes what the workflows in `.github/workflows/` actually do.
If you change a trigger, update this table in the same PR.

## Trigger budget

### `rust-smoke.yml`: the only automatic Rust job

| Event | Runs | Notes |
| --- | --- | --- |
| PR opened, reopened, marked ready, or a new commit pushed (`synchronize`) | **One Rust job** | Only when a path under `Cargo.toml`, `Cargo.lock`, `src/`, `tests/`, `fixtures/`, `examples/`, `benchmarks/native/` or the workflow file changed |
| Draft PR | **No runner job** | The job is skipped; GitHub allocates no runner |
| Documentation-only or `benchmarks/phase/` change | **No run** | Outside the path filter |
| Push to `main` | **One Rust job** | No path filter: every push, including merges |
| Version tag matching `v*` | **One Rust job, plus a second Clippy pass** | Never cancelled |
| Manual `workflow_dispatch` | **One Rust job** | `extended: true` adds the second Clippy pass |

Concurrent runs for the same branch share a group, and a newer run cancels a
stale one (never for tags). So a burst of PR pushes costs about one complete
run per pause, not one per commit. Before merging, check that the green run
is for the **latest commit SHA**; a green check on an earlier SHA does not
validate a newer one.

### Manual and one-off workflows

| Workflow | Triggers | Cost |
| --- | --- | --- |
| `phase-benchmarks.yml` | Manual `workflow_dispatch` only | One Ubuntu job, at most 60 minutes; uploads results as an artifact kept 90 days |
| `native-distribution.yml` | Manual `workflow_dispatch`, `v*` tags | Linux, macOS and Windows matrix, at most 25 minutes each; artifacts kept 14 days |
| `one-time-r0-r5-benchmark.yml` | PR events, but its job runs only on branch `bench/r0-r5-evidence-20261020` | A skipped job on every other PR (no runner) |

No workflow has a `schedule`. Benchmarks, coverage, docs builds and Python
compatibility matrices never run automatically.

## Job cost controls (`rust-smoke.yml`)

- One **standard Ubuntu** runner; no platform or toolchain matrix.
- **10-minute job timeout**.
- **Steps:**
  - formatting;
  - `cargo test --workspace --locked`;
  - Clippy with `-D warnings`;
  - the hiyal test suite again under feature unifications:
    `serde_json/preserve_order`, `serde_json/arbitrary_precision` plus
    `raw_value`, and `live-http` (loopback only);
  - CLI run, resume, sweep and fork smokes;
  - standalone external-world and custom-model consumers;
  - a 3-step native benchmark provenance smoke.
- `Cargo.lock` is committed, and every step uses `--locked`.
- `benchmarks/phase/` is excluded from the workspace and has its own lockfile.
  This job never compiles it or Criterion.
- No benchmarks beyond the 3-step smoke, LLM or network calls, coverage, docs
  builds, artifacts or caches.
- Read-only `GITHUB_TOKEN`, no secrets, and no privileged `pull_request_target`.
- Review median run durations after enough samples. Avoid unmeasured claims
  about minutes saved.

## Benchmarks

Engine benchmarks run only by hand, through `phase-benchmarks.yml` or locally
(see [benchmarks/phase](../benchmarks/phase/README.md) and
[profiling](profiling.md)). Any automatic benchmark gate, such as an
instruction-count regression check on PRs, must first have its runner cost
measured and be added to the tables above.

## Billing detail and safety

As of 2026-10-19, `davletovb/hiyal` is a **public repository**. GitHub's
documented policy is that standard hosted-runner compute is free on public
repositories; larger runners can still cost money. Charges across **other
private repositories** are not explained away by this fact. Do not generalize
the public-repo exemption to all of the owner's projects. See:
https://docs.github.com/en/billing/concepts/product-billing/github-actions

PRs cannot claim verified green CI until the workflow has actually run on
their latest commit. No CI can substitute for the independent observer-on/off
and checkpoint measurements required by [R0](r0-verification.md). Those are
manual, targeted scientific measurements and must have recorded hardware and
raw samples.
