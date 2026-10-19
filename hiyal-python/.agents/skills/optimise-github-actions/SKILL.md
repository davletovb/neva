---
name: optimise-github-actions
description: Audit and optimise GitHub Actions for runner usage and wall-clock speed without weakening meaningful checks. Use when CI is slow or expensive, Actions usage is high, a workflow/job/matrix/cache/trigger is being changed, or before proposing GitHub Actions cost or speed improvements.
---

# Optimise GitHub Actions

Use a measurement-first process. Do not optimise CI from YAML inspection alone.

This repository-local workflow is adapted from the approach in:
https://github.com/enesgules/dotfiles/blob/main/skills/optimise-github-actions/SKILL.md

## 1. Measure first

From the repository root, run:

```bash
python3 .agents/skills/optimise-github-actions/scripts/measure.py --days 14 --out /tmp/github-actions-jobs.json
```

The script auto-detects `OWNER/REPO` with `gh repo view`. Pass `--repo OWNER/REPO` when needed.

Record at least:

- standard hosted runner-minute estimate and monthly projection;
- rounding overhead;
- cancelled and failed runner minutes;
- usage by workflow/event;
- top jobs and the share of workflow runs in which they execute;
- successful-run median and p90 wall-clock time;
- pull-request run churn.

For public repositories, treat hosted minutes primarily as runner usage/speed, not direct cost.

If the bundled script cannot run, reproduce the same measurements with `gh api` before editing workflows.

## 2. Learn what every affected check protects

Before changing a workflow:

1. Read its triggers, `concurrency`, path filters, matrices, `needs` graph, caches, timeouts, permissions, artifacts, and deployment/release dependencies.
2. Inspect repository rulesets and branch protection. Identify required status checks and whether the protected branch requires branches to be up to date.
3. Search the repository for tests or scripts that read files under `.github/workflows/`.
4. Identify release, deploy, packaging, or artifact consumers that rely on the workflow or a specific job/check name.
5. Identify genuinely platform-specific coverage before changing Windows or macOS jobs.

Never rename or remove a required check until its protection/ruleset dependency is deliberately handled.

Useful commands:

```bash
gh api repos/{owner}/{repo}/rulesets
gh api repos/{owner}/{repo}/branches/main/protection
grep -RIn --include='*.test.*' --include='*.spec.*' --include='*.py' --include='*.ts' --include='*.js' --include='*.rs' '.github/workflows\|workflows/' .
```

A missing branch-protection endpoint can simply mean the repository uses rulesets or has no classic protection; verify rather than assume.

## 3. Rank waste by measured impact

Look for these patterns, in roughly this order:

- heavy suites repeated on both pull requests and the post-merge push;
- path filters that still select almost every pull request;
- many tiny jobs whose setup/checkout/install time and per-job rounding dominate useful work;
- slow dependency installation without effective caching;
- repeated runs on the same PR branch and large amounts of cancelled work;
- missing `concurrency` cancellation for superseded PR runs;
- jobs with no sensible `timeout-minutes` despite a stable p90;
- Docker builds or cache exports that dominate the critical path;
- Windows/macOS jobs that are not protecting platform-specific behavior;
- schedules, bots, or manually retriggered workflows that consume disproportionate time.

Do not assume that more parallelism is an optimisation: it can reduce wall-clock time while increasing runner minutes because setup work is duplicated.

## 4. Estimate before editing

For each proposed change, estimate:

| Change | Runner minutes saved | Wall-clock effect | Risk / coverage tradeoff |
| --- | ---: | ---: | --- |

Use measured historical runs where possible. Keep estimates conservative. Mark workflow-dependent savings (for example, draft-PR skipping) as unmeasured unless the repository's actual usage supports the estimate.

Prefer changes that reduce both duplicated work and critical-path time. If cost and speed move in opposite directions, state that explicitly.

## 5. Change workflows safely

Apply the smallest coherent change that captures the measured saving.

- Preserve meaningful test, build, security, packaging, and deployment coverage.
- Preserve required check names unless protection is intentionally updated.
- Add path filtering only when real changed-file history shows it will exclude meaningful work.
- Use dependency caches keyed by the relevant lockfile/input.
- Add PR-scoped concurrency with cancellation when superseded runs are actually a source of waste.
- Set timeouts a little above observed normal/p90 behavior, not arbitrarily low.
- Consolidate tiny jobs only when doing so does not hide independent failures or lengthen the critical path materially.
- Keep long independent jobs parallel when speed matters.
- Keep platform runners for real platform-specific behavior.
- Do not move jobs to self-hosted runners without explicit confirmation of capacity, architecture, and registry/network access.
- Do not enable or depend on merge queue without verifying repository/plan support and merge policy.

When the reason for a non-obvious optimisation would otherwise be easy to undo later, leave a short YAML comment explaining it.

## 6. Verify

Before considering the optimisation complete:

1. Run `actionlint` when available.
2. Run repository tests that assert workflow structure or behavior.
3. Run the relevant local CI/test commands.
4. Inspect the PR's first Actions run for unexpected skipped or renamed required checks.
5. Re-measure after enough representative runs exist before claiming realised savings.

## Report

Summarise the work with:

```markdown
## Baseline
Measured period, runner minutes, monthly projection, median/p90 wall-clock.

## Main sources of waste
Evidence-backed causes, largest first.

## Changes
| Change | Expected runner-minute effect | Expected wall-clock effect | Risk |

## Verification
Checks/tests run and required-check/deployment constraints preserved.

## Not changed
Potential optimisations deliberately skipped and why.
```

The goal is not the fewest CI minutes. The goal is the least waste while preserving the checks that protect what the repository ships.
