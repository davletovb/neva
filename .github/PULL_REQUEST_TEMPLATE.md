## What changed

<!-- One or two sentences per behavioural change. Link the relevant issue or design discussion when useful. -->

## Verification

<!-- Commands you ran and what they printed. State explicitly what you could not run locally. -->

- [ ] Full local suite: `pytest -q` (report the passed/skipped counts)
- [ ] Quality gates: `black --check .`, `isort --check-only .`, `flake8`, `mypy neva benchmarks`,
      `bandit -r neva benchmarks`, `pre-commit run --all-files`
- [ ] `CAPABILITIES.md` updated if this changes supported library behavior
- [ ] `CHANGELOG.md` updated when the change belongs in release history

## Notes for reviewers

<!-- Known limits, deferred follow-ups, or anything that needs a decision. -->
