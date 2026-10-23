#!/usr/bin/env bash
# Compare two Hiyal revisions on the fixed baseline workloads, on this machine,
# with alternating runs so drift (thermal, noisy neighbours, cache state)
# affects both sides alike.
#
#   benchmarks/phase/compare.sh BASE_REV HEAD_REV [ROUNDS] [OUT_DIR]
#
# Both builds use the benchmark code of the current checkout; only the Hiyal
# source differs. They are built without the `replay-cache` feature, so revisions
# before #118 compile too. Set PHASE_COLD=1 to add fresh-process replay with
# a page-cache drop (needs root or passwordless sudo; the outcome is recorded).
# Raw JSON reports and summary.md are written to OUT_DIR.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,13p' "$0" >&2
  exit 2
fi
base_rev=$1
head_rev=$2
rounds=${3:-5}
repo=$(git rev-parse --show-toplevel)
out=${4:-"$repo/benchmarks/results/phase-compare-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$out"
work=$(mktemp -d)
cleanup() {
  for side in base head; do
    git -C "$repo" worktree remove --force "$work/src-$side" 2>/dev/null || true
  done
  rm -rf "$work"
}
trap cleanup EXIT

extra=()
if [ "${PHASE_COLD:-0}" = 1 ]; then
  extra=(--cold --drop-caches)
fi

declare -A bin label
for side in base head; do
  rev=$base_rev
  [ "$side" = head ] && rev=$head_rev
  sha=$(git -C "$repo" rev-parse --verify "$rev^{commit}")
  git -C "$repo" worktree add --quiet --detach "$work/src-$side" "$sha"
  mkdir -p "$work/bench-$side"
  (cd "$repo/benchmarks/phase" && tar --exclude=./target -cf - .) | tar -xf - -C "$work/bench-$side"
  sed -i "s#hiyal = { path = \"../..\" }#hiyal = { path = \"$work/src-$side\" }#" \
    "$work/bench-$side/Cargo.toml"
  grep -q "path = \"$work/src-$side\"" "$work/bench-$side/Cargo.toml"
  echo "compare: building $side ($sha)" >&2
  CARGO_TARGET_DIR="$work/target-$side" cargo build --quiet --release --no-default-features \
    --bin phase-report --manifest-path "$work/bench-$side/Cargo.toml"
  bin[$side]="$work/target-$side/release/phase-report"
  label[$side]="$side:${sha:0:12}"
done

for round in $(seq 1 "$rounds"); do
  order=(base head)
  [ $((round % 2)) -eq 0 ] && order=(head base)
  for side in "${order[@]}"; do
    echo "compare: round $round/$rounds $side" >&2
    "${bin[$side]}" run --suite baseline --reps 3 "${extra[@]}" \
      --label "${label[$side]}" --out "$out/round-$round-$side.json"
  done
done

"${bin[head]}" summarize "$out"/round-*.json > "$out/summary.md"
echo "compare: wrote $out/summary.md and $((rounds * 2)) raw reports" >&2
