#!/bin/sh
# Manual-only R0 evidence collection. No CI or Dependabot triggers.
set -eu
cd "$(dirname "$0")/.."
: "${HIYAL_LEGACY_SOURCE:?Set HIYAL_LEGACY_SOURCE to an archive-branch hiyal-python directory before running Python benchmarks}"
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
python3 benchmarks/reference_python.py --agents 20 --steps 100 500 1000 \
  --repeat 3 --output benchmarks/results/python-20.json
python3 benchmarks/reference_python.py --agents 100 --steps 100 500 1000 \
  --repeat 3 --output benchmarks/results/python-100.json
# 1,000-agent observer comparison and retained-RSS test is opt-in;
# it may require substantial RAM on the Python implementation.
printf '%s\n' 'R0 evidence collected. Record host CPU/RAM, versions, commit and lockfile hash.'
