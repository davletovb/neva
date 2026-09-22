"""Measure checkpoint creation, save, and load scaling.

This benchmark intentionally has no pass/fail performance threshold. It records
wall-clock time, peak Python allocations reported by tracemalloc, and serialized
checkpoint bytes so persistence changes can be compared on the same machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from neva.utils.state_management import (
    ConversationState,
    create_snapshot,
    load_snapshot,
    save_snapshot,
)

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkCase:
    """Deterministic checkpoint workload description."""

    name: str
    agents: int
    turns_per_agent: int
    message_chars: int
    environment_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.agents <= 0:
            raise ValueError("agents must be positive")
        for field_name in ("turns_per_agent", "message_chars", "environment_bytes"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


QUICK_CASES: Tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="smoke",
        agents=2,
        turns_per_agent=10,
        message_chars=64,
        environment_bytes=1024,
    ),
)

STANDARD_CASES: Tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="small",
        agents=2,
        turns_per_agent=100,
        message_chars=256,
        environment_bytes=16 * 1024,
    ),
    BenchmarkCase(
        name="medium",
        agents=4,
        turns_per_agent=500,
        message_chars=512,
        environment_bytes=128 * 1024,
    ),
    BenchmarkCase(
        name="large",
        agents=8,
        turns_per_agent=1000,
        message_chars=1024,
        environment_bytes=1024 * 1024,
    ),
)


def _measure(operation: Callable[[], T]) -> Tuple[T, Dict[str, float]]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = operation()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {
        "elapsed_ms": elapsed_ms,
        "peak_python_bytes": float(peak_bytes),
    }


def _fixture(case: BenchmarkCase) -> Tuple[Dict[str, object], List[ConversationState]]:
    environment_state: Dict[str, object] = {
        "payload": "e" * case.environment_bytes,
        "benchmark_case": case.name,
    }
    states: List[ConversationState] = []
    message = "m" * case.message_chars
    for agent_index in range(case.agents):
        state = ConversationState(agent_name=f"agent-{agent_index}")
        for turn_index in range(case.turns_per_agent):
            state.record_turn(f"speaker-{turn_index % 2}", message)
        states.append(state)
    return environment_state, states


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def run_case(
    case: BenchmarkCase,
    *,
    repeat: int = 3,
    workdir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one checkpoint workload and return raw samples plus medians."""

    if repeat <= 0:
        raise ValueError("repeat must be positive")

    environment_state, states = _fixture(case)
    benchmark_root: Optional[Path] = None
    if workdir is not None:
        benchmark_root = Path(workdir)
        benchmark_root.mkdir(parents=True, exist_ok=True)

    tempdir = tempfile.TemporaryDirectory(
        prefix=f"neva-checkpoint-{case.name}-",
        dir=str(benchmark_root) if benchmark_root is not None else None,
    )
    benchmark_dir = Path(tempdir.name)

    samples: List[Dict[str, Any]] = []
    try:
        for sample_index in range(repeat):
            snapshot, create_metrics = _measure(
                lambda: create_snapshot(
                    environment_state=environment_state,
                    agent_states=states,
                )
            )
            checkpoint_path = benchmark_dir / f"{case.name}-{sample_index}.json"
            _, save_metrics = _measure(lambda: save_snapshot(snapshot, checkpoint_path))
            loaded, load_metrics = _measure(lambda: load_snapshot(checkpoint_path))
            if loaded.environment_state != snapshot.environment_state:
                raise RuntimeError("checkpoint roundtrip changed environment_state")

            samples.append(
                {
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "create_snapshot": create_metrics,
                    "save_snapshot": save_metrics,
                    "load_snapshot": load_metrics,
                }
            )
            checkpoint_path.unlink()
    finally:
        tempdir.cleanup()

    stage_names = ("create_snapshot", "save_snapshot", "load_snapshot")
    summary = {
        stage: {
            "median_ms": _median(sample[stage]["elapsed_ms"] for sample in samples),
            "median_peak_python_bytes": _median(
                sample[stage]["peak_python_bytes"] for sample in samples
            ),
        }
        for stage in stage_names
    }

    return {
        "case": asdict(case),
        "repeat": repeat,
        "median_checkpoint_bytes": int(
            statistics.median(sample["checkpoint_bytes"] for sample in samples)
        ),
        "stages": summary,
        "samples": samples,
    }


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    repeat: int = 3,
    workdir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a sequence of checkpoint workloads with machine metadata."""

    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "measurement_notes": {
            "elapsed": "wall-clock milliseconds from time.perf_counter",
            "memory": (
                "peak Python allocations from tracemalloc during each stage; "
                "excludes OS page cache and temporary/destination file space"
            ),
            "thresholds": "none; compare results on equivalent hardware",
        },
        "cases": [run_case(case, repeat=repeat, workdir=workdir) for case in cases],
    }


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("quick", "standard"),
        default="standard",
        help="quick is a smoke workload; standard measures three increasing sizes",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="samples per case; medians are reported",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; stdout is always printed",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cases = QUICK_CASES if args.profile == "quick" else STANDARD_CASES
    result = run_benchmark(cases, repeat=args.repeat)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
