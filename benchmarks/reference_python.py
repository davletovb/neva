#!/usr/bin/env python3
"""R0 archived-Python reference: scripted turns, observers ON versus OFF.

This runs only the preserved hiyal-python/ implementation. It is not part
of the new Rust package and requires the legacy requests dependency.
All measurements are machine-specific and *not* Rust performance claims.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "hiyal-python"


def _git_revision() -> str:
    """Fail closed on missing source provenance for independent R0 runs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    fallback = os.environ.get("GITHUB_SHA")
    if fallback:
        return fallback
    raise RuntimeError("Cannot determine git revision; run from a Git checkout")


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _rss_bytes() -> int | None:
    """Current RSS on Linux; other platforms report peak RSS separately."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(amount if sys.platform == "darwin" else amount * 1024)
    except (ImportError, OSError):
        return None


def _percentile(values: list[float], quantile: float) -> float:
    ranked = sorted(values)
    return ranked[max(0, math.ceil(len(ranked) * quantile) - 1)]


def _worker(agents: int, steps: int, observer_on: bool) -> dict:
    if not LEGACY.is_dir():
        raise RuntimeError("Archived Python implementation not found at hiyal-python/")
    # Import inside the subprocess, so every sample has independent allocations.
    sys.path.insert(0, str(LEGACY))
    started_import = time.perf_counter_ns()
    from hiyal.agents import AgentManager
    from hiyal.environments import BasicEnvironment
    from hiyal.schedulers import RoundRobinScheduler
    imported_ms = (time.perf_counter_ns() - started_import) / 1e6

    started_setup = time.perf_counter_ns()
    scheduler = RoundRobinScheduler()
    if not observer_on:
        # Disable before registering agents: watcher hooks otherwise skew setup.
        scheduler.simulation_observer = None
    env = BasicEnvironment("R0 scripted", "Fixed response, no network.", scheduler)
    manager = AgentManager()

    for index in range(agents):
        agent = manager.create_agent(
            "transformer",
            name=f"actor-{index}",
            llm_backend=lambda _prompt: "ok",
        )
        env.register_agent(agent)
    setup_ms = (time.perf_counter_ns() - started_setup) / 1e6

    durations_ms: list[float] = []
    checkpoints: list[dict] = []
    # The Python observer copies its accumulating metric history on each step.
    checkpoint_at = set([1, steps, *[min(steps, i) for i in (100, 500, 1000)]])
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    for decision in range(1, steps + 1):
        before = time.perf_counter_ns()
        output = env.step()
        durations_ms.append((time.perf_counter_ns() - before) / 1e6)
        if output != "ok":
            raise RuntimeError(f"unexpected scripted backend output: {output!r}")
        if decision in checkpoint_at:
            observer = scheduler.simulation_observer
            retained = (
                sum(len(values) for values in observer.data.values())
                if observer is not None
                else 0
            )
            checkpoints.append({
                "steps": decision,
                "rss_current_bytes": _rss_bytes(),
                "observer_metric_entries": retained,
            })
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    cpu_ms = (time.process_time_ns() - cpu_start) / 1e6

    return {
        "observer": "on" if observer_on else "off",
        "agents": agents,
        "steps": steps,
        "import_ms": imported_ms,
        "setup_ms": setup_ms,
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "steps_per_second": steps * 1000 / wall_ms,
        "latency_p50_ms": statistics.median(durations_ms),
        "latency_p95_ms": _percentile(durations_ms, 0.95),
        "peak_rss_bytes": _peak_rss_bytes(),
        "checkpoints": checkpoints,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--observer", choices=("on", "off"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.agents < 1 or args.repeat < 1 or any(n < 1 for n in args.steps):
        parser.error("agents, repeat and step counts must all be positive")
    if args.worker:
        if len(args.steps) != 1 or args.observer is None:
            parser.error("worker requires one --steps and --observer")
        print(json.dumps(_worker(args.agents, args.steps[0], args.observer == "on")))
        return 0

    result = {
        "schema_version": 1,
        "baseline": "archived_python",
        "source": "hiyal-python/",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_revision": _git_revision(),
        "worktree_dirty": _git_dirty(),
        "notes": [
            "One fresh subprocess per sample; no provider calls; observer toggled before registration",
            "Latency samples include Python per-step timing overhead in both modes",
            "Current RSS checkpoint series available on Linux; peak RSS from process resource usage",
            "Record machine and source revisions before comparing with Rust using equivalent semantics",
            "No stdout from third-party providers is expected; no network backend is used",
        ],
        "cases": [],
    }
    for steps in args.steps:
        for mode in ("on", "off"):
            samples = []
            for _ in range(args.repeat):
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--agents", str(args.agents), "--steps", str(steps),
                    "--observer", mode,
                ]
                proc = subprocess.run(command, text=True, capture_output=True)
                if proc.returncode:
                    raise RuntimeError(
                        f"baseline worker failed: {proc.returncode}\n{proc.stderr}"
                    )
                samples.append(json.loads(proc.stdout))
            result["cases"].append({
                "agents": args.agents,
                "steps": steps,
                "observer": mode,
                "repeat": args.repeat,
                "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
                "median_throughput": statistics.median(
                    s["steps_per_second"] for s in samples
                ),
                "samples": samples,
            })
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
