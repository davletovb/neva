## What changed

<!-- One or two sentences per behavioural change. Link the tracked gap if this closes one. -->

## Verification

<!-- Commands you ran and what they printed. State explicitly what you could not run locally. -->

- [ ] Full local suite: `pytest -q` (report the passed/skipped counts)
- [ ] Quality gates: `black --check .`, `isort --check-only .`, `flake8`, `mypy neva benchmarks`,
      `bandit -r neva benchmarks`, `pre-commit run --all-files`
- [ ] `ROBUSTNESS_GAPS.md` and `CHANGELOG.md` updated if this changes tracked status or behavior

## Notes for reviewers

<!-- Known limits, deferred follow-ups, or anything that needs a decision. -->
