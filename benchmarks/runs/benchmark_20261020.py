#!/usr/bin/env python3
"""Independent one-time R0/R5 measurements; no Python runtime in Rust core.

The workloads differ: Python is an in-memory LLM scheduler, Rust a durably
fsynced event-sourced counter. Calculate only *within-runtime* overhead.
"""
from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark-results"
OUT.mkdir(exist_ok=True)
ARCHIVE = ROOT / "archived"
LEGACY = ARCHIVE / "hiyal-python"
RUST_BIN = ROOT / "target" / "release" / "hiyal-native-bench"
AGENTS = (20, 100)
STEPS = (100, 500, 1000)
REPEATS = 5
ERRORS: list[str] = []


def shell(command: list[str], *, timeout: int = 900, env: dict[str, str] | None = None) -> str:
    started = time.monotonic()
    run = subprocess.run(
        command, cwd=ROOT, env=env, encoding="utf-8", capture_output=True,
        timeout=timeout, check=False,
    )
    if run.returncode:
        raise RuntimeError(
            f"cmd={command!r} returncode={run.returncode}; stderr_tail={run.stderr[-3500:]!r}"
        )
    print(f"COMPLETE {' '.join(command[:3])}: {time.monotonic() - started:.1f}s", flush=True)
    return run.stdout


def version(*command: str) -> str:
    try:
        return shell(list(command), timeout=20).strip().splitlines()[0]
    except Exception as error:
        return f"unavailable: {error!s}"


def stats(samples: list[dict], key: str) -> dict:
    data = [float(s[key]) for s in samples]
    return {
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
        "sample_count": len(data),
    }


machine = {
    "main_git_sha": version("git", "rev-parse", "HEAD"),
    "archive_git_sha": version("git", "-C", str(ARCHIVE), "rev-parse", "HEAD"),
    "python_version": platform.python_version(),
    "rustc": version("rustc", "--version"),
    "cargo": version("cargo", "--version"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "cpu_model": next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text()
         .splitlines() if line.startswith("model name")), "unknown"
    ),
    "mem_total": next(
        (line for line in Path("/proc/meminfo").read_text().splitlines()
         if line.startswith("MemTotal:")), "unknown"
    ),
    "kernel": platform.release(),
    "runner": os.environ.get("RUNNER_NAME"),
    "gha_run_id": os.environ.get("GITHUB_RUN_ID"),
    "gha_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    "fs_type": version("stat", "-f", "-c", "%T", "/tmp"),
    "python_workload": "archived in-memory scripted BasicEnvironment.step()",
    "rust_workload": "fsynced hash-chained Counter Trial::step(), with or without JSONL/CSV",
    "comparable": False,
    "repeat": REPEATS,
    "case_agents": list(AGENTS),
    "case_steps": list(STEPS),
}
(OUT / "environment.json").write_text(json.dumps(machine, indent=2) + "\n")
env = dict(os.environ)
env["HIYAL_LEGACY_SOURCE"] = str(LEGACY)
env["PYTHONPATH"] = str(LEGACY)

python_results = []
for agents in AGENTS:
    try:
        output = OUT / f"python-{agents}.json"
        stdout = shell([
            sys.executable, "benchmarks/reference_python.py",
            "--agents", str(agents), "--steps", *map(str, STEPS),
            "--repeat", str(REPEATS), "--output", str(output),
        ], timeout=1100, env=env)
        if not output.exists():
            raise RuntimeError("missing Python benchmark result")
        python_results.extend(json.loads(output.read_text())["cases"])
        print(f"PYTHON DATA {agents}: {len(stdout)} bytes", flush=True)
    except Exception as exc:
        ERRORS.append(f"Python agents={agents}: {exc}")

try:
    path = OUT / "python-checkpoint.json"
    stdout = shell([
        sys.executable, str(LEGACY / "benchmarks/checkpoint_scaling.py"),
        "--profile", "standard", "--repeat", "3", "--output", str(path),
        "--git-sha", machine["archive_git_sha"],
    ], timeout=480, env=env)
    print(f"PYTHON CHECKPOINT {len(stdout)} bytes", flush=True)
except Exception as exc:
    ERRORS.append(f"Python checkpoint: {exc}")

# Deterministic interleaved order minimizes systematic on/off thermal bias.
jobs = [(a, n, state, repeat)
        for a in AGENTS for n in STEPS for state in ("on", "off")
        for repeat in range(REPEATS)]
random.Random(103).shuffle(jobs)
rust_samples = []
native_path = OUT / "rust-raw.jsonl"
with native_path.open("w") as file:
    for index, (agents, steps, observer, repetition) in enumerate(jobs, 1):
        started = time.monotonic()
        try:
            output = shell([str(RUST_BIN), str(agents), str(steps), observer], timeout=180)
            data = json.loads(output.strip())
            if (data["actors"], data["steps"], data["observer"]) != (agents, steps, observer):
                raise ValueError("Rust process reported mismatched case")
            data.update(repetition=repetition, process_elapsed_ms=(time.monotonic() - started) * 1000)
            rust_samples.append(data)
            file.write(json.dumps(data, sort_keys=True) + "\n")
            file.flush()
        except Exception as exc:
            ERRORS.append(f"Rust case={agents}/{steps}/{observer}/{repetition}: {exc}")
            break
        if index % 10 == 0:
            print(f"RUST SAMPLES {index}/{len(jobs)}", flush=True)

summary = {
    "metadata": machine,
    "caveat": "Different units of work. Compare ON/OFF *within* each runtime only.",
    "python": [],
    "rust": [],
    "python_checkpoint": None,
    "errors": ERRORS,
}
for agents in AGENTS:
    for steps in STEPS:
        cases = {mode: next((c for c in python_results
                             if c["agents"] == agents and c["steps"] == steps
                             and c["observer"] == mode), None)
                 for mode in ("on", "off")}
        if all(cases.values()):
            p_on = cases["on"]["median_wall_ms"]
            p_off = cases["off"]["median_wall_ms"]
            summary["python"].append({
                "actors": agents, "steps": steps,
                "on_wall_median_ms": p_on, "off_wall_median_ms": p_off,
                "observer_overhead_ratio": p_on / p_off if p_off else None,
                "on_throughput_median": cases["on"]["median_throughput"],
                "off_throughput_median": cases["off"]["median_throughput"],
                "on_peak_rss_median_bytes": statistics.median(
                    s["peak_rss_bytes"] for s in cases["on"]["samples"]
                    if s["peak_rss_bytes"] is not None
                ),
                "off_peak_rss_median_bytes": statistics.median(
                    s["peak_rss_bytes"] for s in cases["off"]["samples"]
                    if s["peak_rss_bytes"] is not None
                ),
            })
        rs = {mode: [s for s in rust_samples if s["actors"] == agents
                     and s["steps"] == steps and s["observer"] == mode]
              for mode in ("on", "off")}
        if all(len(v) == REPEATS for v in rs.values()):
            on = stats(rs["on"], "wall_ms")
            off = stats(rs["off"], "wall_ms")
            summary["rust"].append({
                "actors": agents, "steps": steps,
                "on_wall_median_ms": on["median"], "off_wall_median_ms": off["median"],
                "observer_overhead_ratio": on["median"] / off["median"],
                "on_throughput_median": stats(rs["on"], "throughput_steps_per_second")["median"],
                "off_throughput_median": stats(rs["off"], "throughput_steps_per_second")["median"],
                "on_rss_median_bytes": stats(rs["on"], "rss_current_bytes")["median"],
                "off_rss_median_bytes": stats(rs["off"], "rss_current_bytes")["median"],
            })
checkpoint = OUT / "python-checkpoint.json"
if checkpoint.exists():
    cp = json.loads(checkpoint.read_text())
    summary["python_checkpoint"] = [
        {"case": c["case"]["name"], "size_bytes": c["median_checkpoint_bytes"],
         "create_ms": c["stages"]["create_snapshot"]["median_ms"],
         "save_ms": c["stages"]["save_snapshot"]["median_ms"],
         "load_ms": c["stages"]["load_snapshot"]["median_ms"]}
        for c in cp["cases"]
    ]
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("=== BENCHMARK_SUMMARY_JSON_BEGIN ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("=== BENCHMARK_SUMMARY_JSON_END ===", flush=True)
sys.exit(0 if not ERRORS else 1)
