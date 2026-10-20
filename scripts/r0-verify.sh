#!/bin/sh
# Manual-only R0 evidence collection. No CI or Dependabot triggers.
set -eu
# Resolve a relative archive path from the caller's directory before the cd.
if [ -n "${HIYAL_LEGACY_SOURCE:-}" ] && [ -d "$HIYAL_LEGACY_SOURCE" ]; then
  HIYAL_LEGACY_SOURCE=$(cd "$HIYAL_LEGACY_SOURCE" && pwd)
  export HIYAL_LEGACY_SOURCE
fi
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
printf 'Repository revision: '
git rev-parse HEAD
printf 'Platform: '
uname -a
printf 'Python version: '
python3 --version
printf 'Rust compiler: '
rustc --version
printf 'Cargo: '
cargo --version
python3 -m compileall -q benchmarks/reference_python.py scripts/r0-fixture-check.py
python3 scripts/r0-fixture-check.py
python3 -m json.tool fixtures/golden/scripted-two-agent-v1.json >/dev/null
# Initial repository PR does not contain a machine-generated lockfile.
# Generate one once dependencies are accessible and retain it for repeatability.
if [ ! -f Cargo.lock ]; then
  cargo generate-lockfile
fi
cargo fmt --all -- --check
cargo test --workspace --locked
cargo run --locked -p hiyal-custom-world-example >benchmarks/results/custom-world.jsonl
# The Python reference benchmarks alone need the archived implementation.
if [ -z "${HIYAL_LEGACY_SOURCE:-}" ] || [ ! -d "$HIYAL_LEGACY_SOURCE" ]; then
  printf '%s\n' 'Set HIYAL_LEGACY_SOURCE to an archive-branch hiyal-python directory to run the Python benchmarks.' >&2
  exit 1
fi
python3 benchmarks/reference_python.py --agents 20 --steps 100 500 1000 \
  --repeat 3 --output benchmarks/results/python-20.json
python3 benchmarks/reference_python.py --agents 100 --steps 100 500 1000 \
  --repeat 3 --output benchmarks/results/python-100.json
# 1,000-agent observer comparison and retained-RSS test is opt-in;
# it may require substantial RAM on the Python implementation.
printf '%s\n' 'R0 evidence collected. Record host CPU/RAM, versions, commit and lockfile hash.'
