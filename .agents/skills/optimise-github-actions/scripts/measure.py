#!/usr/bin/env python3
"""Measure GitHub Actions runner usage and wall-clock time.

Requires Python 3.9+ and an authenticated GitHub CLI (`gh`) with Actions read
access. This reports a standard-hosted-runner minute estimate for comparison;
GitHub plan details, larger runners, and pricing should be checked separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def run_gh(*args: str) -> str:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "gh command failed"
        raise RuntimeError(message)
    return proc.stdout


def gh_api(path: str) -> list[Any]:
    raw = run_gh("api", "--paginate", "--slurp", path)
    value = json.loads(raw)
    if not isinstance(value, list):
        raise RuntimeError(f"Unexpected gh api response for {path}")
    return value


def detect_repo() -> str:
    return run_gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner").strip()


def flatten_pages(pages: Iterable[Any], key: str | None = None) -> list[Any]:
    out: list[Any] = []
    for page in pages:
        if key is None:
            if isinstance(page, list):
                out.extend(page)
            else:
                out.append(page)
            continue
        if isinstance(page, dict):
            values = page.get(key, [])
            if isinstance(values, list):
                out.extend(values)
    return out


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_minutes(start: str | None, end: str | None) -> float:
    a, b = parse_time(start), parse_time(end)
    if a is None or b is None:
        return 0.0
    return max(0.0, (b - a).total_seconds() / 60.0)


def standard_hosted(job: dict[str, Any]) -> bool:
    labels = {str(x).lower() for x in job.get("labels", [])}
    if "self-hosted" in labels:
        return False
    group = job.get("runner_group_name")
    return group in (None, "", "GitHub Actions")


def multiplier(job: dict[str, Any]) -> int:
    labels = [str(x).lower() for x in job.get("labels", [])]
    if any(x.startswith("macos") for x in labels):
        return 10
    if any(x.startswith("windows") for x in labels):
        return 2
    return 1


def billed_estimate(job: dict[str, Any]) -> int:
    minutes = duration_minutes(job.get("started_at"), job.get("completed_at"))
    if minutes <= 0 or not standard_hosted(job):
        return 0
    return math.ceil(minutes) * multiplier(job)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(q * len(ordered)))
    return ordered[index]


def fmt_int(value: float | int) -> str:
    return f"{round(value):,}"


def pct(part: float, total: float) -> str:
    return f"{(100.0 * part / total) if total else 0.0:.1f}%"


UTC = dt.timezone.utc
MAX_FILTERED_RUNS = 1000


def first_object(pages: Iterable[Any]) -> dict[str, Any]:
    for page in pages:
        if isinstance(page, dict):
            return page
    return {}


def github_timestamp(value: dt.datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def filtered_runs(
    repo: str,
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[dict[str, Any]], int]:
    query = f"{github_timestamp(start)}..{github_timestamp(end)}"
    pages = gh_api(f"repos/{repo}/actions/runs?per_page=100&created={query}")
    runs = [run for run in flatten_pages(pages, "workflow_runs") if isinstance(run, dict)]
    first = first_object(pages)
    total_count = int(first.get("total_count") or len(runs))
    return runs, total_count


def collect_interval(repo: str, start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]]:
    runs, total_count = filtered_runs(repo, start, end)
    if total_count < MAX_FILTERED_RUNS:
        return runs

    span_seconds = int((end - start).total_seconds())
    if span_seconds <= 0:
        raise RuntimeError(
            f"GitHub returned at least {MAX_FILTERED_RUNS} runs for one second "
            f"({github_timestamp(start)}); cannot measure without truncation"
        )

    midpoint = start + dt.timedelta(seconds=span_seconds // 2)
    left = collect_interval(repo, start, midpoint)
    right_start = midpoint + dt.timedelta(seconds=1)
    right = collect_interval(repo, right_start, end) if right_start <= end else []
    return left + right


def collect_runs(repo: str, days: int) -> tuple[list[dict[str, Any]], dt.datetime, dt.datetime]:
    # Use completed UTC days so --days N always covers exactly N full days.
    end_exclusive = dt.datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end_exclusive - dt.timedelta(days=days)
    end_inclusive = end_exclusive - dt.timedelta(seconds=1)
    dates = [start.date() + dt.timedelta(days=i) for i in range(days)]

    def one_day(day: dt.date) -> list[dict[str, Any]]:
        day_start = dt.datetime.combine(day, dt.time.min, tzinfo=UTC)
        day_end = day_start + dt.timedelta(days=1) - dt.timedelta(seconds=1)
        return collect_interval(repo, day_start, day_end)

    runs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for day_runs in pool.map(one_day, dates):
            runs.extend(day_runs)

    # Split queries are adjacent, but de-duplicate defensively by workflow-run id.
    by_id = {run.get("id"): run for run in runs if run.get("id") is not None}
    return list(by_id.values()), start, end_inclusive


def attempt_conclusion(repo: str, run: dict[str, Any], attempt: int, attempts: int) -> str | None:
    if attempts == 1:
        return run.get("conclusion")
    pages = gh_api(f"repos/{repo}/actions/runs/{run['id']}/attempts/{attempt}")
    return first_object(pages).get("conclusion")


def workflow_identity(run: dict[str, Any]) -> tuple[str, str, str]:
    workflow_id = str(run.get("workflow_id") or run.get("path") or run.get("name") or "unknown")
    name = str(run.get("name") or workflow_id)
    path = str(run.get("path") or "")
    return workflow_id, name, path


def pr_identity(run: dict[str, Any]) -> str:
    pull_requests = run.get("pull_requests")
    if isinstance(pull_requests, list) and pull_requests:
        first = pull_requests[0]
        if isinstance(first, dict) and first.get("number") is not None:
            return f"PR #{first['number']}"

    head_repo = run.get("head_repository")
    if isinstance(head_repo, dict):
        full_name = head_repo.get("full_name")
        if full_name:
            return f"{full_name}:{run.get('head_branch') or '?'}"
    return f"unknown-repo:{run.get('head_branch') or '?'}"


def collect_jobs(repo: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def one_run(run: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = run["id"]
        attempts = max(1, int(run.get("run_attempt") or 1))
        workflow_id, workflow_name, workflow_path = workflow_identity(run)
        result: list[dict[str, Any]] = []

        for attempt in range(1, attempts + 1):
            path = f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
            try:
                pages = gh_api(path)
            except RuntimeError:
                if attempts == 1:
                    pages = gh_api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")
                else:
                    raise

            run_conclusion = attempt_conclusion(repo, run, attempt, attempts)
            for job in flatten_pages(pages, "jobs"):
                if not isinstance(job, dict):
                    continue
                result.append(
                    {
                        "run": run_id,
                        "attempt": attempt,
                        "run_conclusion": run_conclusion,
                        "workflow_id": workflow_id,
                        "workflow": workflow_name,
                        "workflow_path": workflow_path,
                        "event": run.get("event", ""),
                        "branch": run.get("head_branch", ""),
                        "pr_identity": pr_identity(run),
                        "job": job.get("name", ""),
                        "conclusion": job.get("conclusion"),
                        "labels": job.get("labels", []),
                        "runner_group_name": job.get("runner_group_name"),
                        "started_at": job.get("started_at"),
                        "completed_at": job.get("completed_at"),
                    }
                )
        return result

    jobs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one_run, run) for run in runs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            jobs.extend(future.result())
            if i % 50 == 0:
                print(f"  read jobs for {i}/{len(runs)} runs", file=sys.stderr)
    return jobs


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    alignment = [
        "---:" if i < len(headers) - 1 else "---" for i in range(len(headers))
    ]
    print("|" + "|".join(alignment) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="OWNER/REPO; defaults to current gh repository")
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of completed UTC days to measure",
    )
    parser.add_argument("--out", help="Optional path for raw job JSON")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be >= 1")

    try:
        repo = args.repo or detect_repo()
        repo_info_pages = gh_api(f"repos/{repo}")
        repo_info = first_object(repo_info_pages)
        runs, window_start, window_end = collect_runs(repo, args.days)
        window = (
            f"{github_timestamp(window_start)} through {github_timestamp(window_end)}"
        )
        print(
            f"{len(runs)} runs in {args.days} completed UTC days ({window}); reading jobs...",
            file=sys.stderr,
        )
        jobs = collect_jobs(repo, runs)
    except (RuntimeError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    ran = [
        job
        for job in jobs
        if job.get("conclusion") != "skipped"
        and duration_minutes(job.get("started_at"), job.get("completed_at")) > 0
    ]
    hosted = [job for job in ran if standard_hosted(job)]
    other = [job for job in ran if not standard_hosted(job)]

    total = sum(billed_estimate(job) for job in hosted)
    raw_weighted = sum(
        duration_minutes(job.get("started_at"), job.get("completed_at")) * multiplier(job)
        for job in hosted
    )
    rounding = max(0.0, total - raw_weighted)
    cancelled = sum(
        billed_estimate(job)
        for job in hosted
        if job.get("run_conclusion") == "cancelled"
    )
    failed = sum(
        billed_estimate(job)
        for job in hosted
        if job.get("run_conclusion") == "failure"
    )

    print(f"# GitHub Actions usage: {repo}, {args.days} completed UTC days\n")
    print(f"- Window: {github_timestamp(window_start)} through {github_timestamp(window_end)}")
    if repo_info.get("private") is False:
        print(
            "> Public repository: standard hosted runners are generally not a direct minutes "
            "charge; use these figures mainly to compare runner usage and speed.\n"
        )

    print(f"- Runs: {len(runs)}; jobs that ran: {len(ran)}")
    print(
        "- Standard hosted runner-minute estimate: "
        f"**{fmt_int(total)}** (about {fmt_int(total * 30 / args.days)} per 30 completed UTC days)"
    )
    print(f"- Per-job minute rounding estimate: {fmt_int(rounding)} ({pct(rounding, total)})")
    print(
        f"- Jobs in cancelled runs: {fmt_int(cancelled)}; jobs in failed runs: "
        f"{fmt_int(failed)} estimated runner minutes"
    )
    if other:
        groups = sorted({str(job.get("runner_group_name") or "self-hosted/other") for job in other})
        raw_other = sum(
            duration_minutes(job.get("started_at"), job.get("completed_at"))
            for job in other
        )
        print(
            f"- Other runner groups ({', '.join(groups)}): {fmt_int(raw_other)} raw minutes, "
            "excluded from the hosted estimate"
        )

    by_flow: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"billed": 0, "count": 0, "raw": 0.0, "name": "", "path": ""}
    )
    by_job: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"billed": 0, "count": 0, "raw": 0.0, "runs": set(), "name": "", "path": ""}
    )

    for job in hosted:
        flow_key = (str(job.get("workflow_id", "unknown")), str(job.get("event", "")))
        job_key = (*flow_key, str(job.get("job", "")))
        duration = duration_minutes(job.get("started_at"), job.get("completed_at"))
        estimate = billed_estimate(job)

        by_flow[flow_key]["billed"] += estimate
        by_flow[flow_key]["count"] += 1
        by_flow[flow_key]["raw"] += duration
        by_flow[flow_key]["name"] = str(job.get("workflow", ""))
        by_flow[flow_key]["path"] = str(job.get("workflow_path", ""))

        by_job[job_key]["billed"] += estimate
        by_job[job_key]["count"] += 1
        by_job[job_key]["raw"] += duration
        by_job[job_key]["runs"].add((job.get("run"), job.get("attempt")))
        by_job[job_key]["name"] = str(job.get("workflow", ""))
        by_job[job_key]["path"] = str(job.get("workflow_path", ""))

    print("\n## By workflow and event\n")
    flow_rows: list[list[str]] = []
    top_flows = sorted(
        by_flow.items(),
        key=lambda item: item[1]["billed"],
        reverse=True,
    )[:20]
    for (workflow_id, event), values in top_flows:
        label = values["name"] or workflow_id
        if values["path"]:
            label = f"{label} [{values['path']}]"
        flow_rows.append(
            [
                fmt_int(values["billed"]),
                pct(values["billed"], total),
                str(values["count"]),
                f'{values["raw"] / values["count"]:.1f}',
                f"{label} / {event}",
            ]
        )
    print_table(["Est. min", "Share", "Jobs", "Avg raw min", "Workflow / event"], flow_rows)

    print("\n## Top jobs\n")
    job_rows: list[list[str]] = []
    top_jobs = sorted(
        by_job.items(),
        key=lambda item: item[1]["billed"],
        reverse=True,
    )[:30]
    for (workflow_id, event, name), values in top_jobs:
        run_count = sum(
            max(1, int(run.get("run_attempt") or 1))
            for run in runs
            if workflow_identity(run)[0] == workflow_id
            and str(run.get("event", "")) == event
        )
        ran_in = len(values["runs"])
        workflow_label = values["name"] or workflow_id
        if values["path"]:
            workflow_label = f"{workflow_label} [{values['path']}]"
        job_rows.append(
            [
                fmt_int(values["billed"]),
                pct(values["billed"], total),
                str(values["count"]),
                f'{values["raw"] / values["count"]:.1f}',
                pct(ran_in, run_count),
                f"{workflow_label} / {event} :: {name}",
            ]
        )
    print_table(["Est. min", "Share", "Jobs", "Avg raw min", "Ran in", "Name"], job_rows)

    durations: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"values": [], "name": "", "path": ""}
    )
    for run in runs:
        if run.get("conclusion") != "success":
            continue
        minutes = duration_minutes(run.get("run_started_at"), run.get("updated_at"))
        if minutes <= 0:
            continue
        workflow_id, workflow_name, workflow_path = workflow_identity(run)
        key = (workflow_id, str(run.get("event", "")))
        durations[key]["values"].append(minutes)
        durations[key]["name"] = workflow_name
        durations[key]["path"] = workflow_path

    print("\n## Wall-clock time of successful runs\n")
    duration_rows: list[list[str]] = []
    for (workflow_id, event), info in sorted(
        durations.items(), key=lambda item: len(item[1]["values"]), reverse=True
    )[:15]:
        values = info["values"]
        label = info["name"] or workflow_id
        if info["path"]:
            label = f"{label} [{info['path']}]"
        duration_rows.append(
            [
                str(len(values)),
                f"{statistics.median(values):.1f}",
                f"{quantile(values, 0.90):.1f}",
                f"{label} / {event}",
            ]
        )
    print_table(["Runs", "Median min", "p90 min", "Workflow / event"], duration_rows)

    pr_counts: Counter[tuple[str, str]] = Counter()
    for run in runs:
        if run.get("event") != "pull_request":
            continue
        workflow_id, _, _ = workflow_identity(run)
        pr_counts[(workflow_id, pr_identity(run))] += max(1, int(run.get("run_attempt") or 1))

    if pr_counts:
        values = sorted(pr_counts.values())
        print("\n## Pull-request churn\n")
        print(
            "- Executions per PR/workflow pair (rerun attempts included): "
            f"median {statistics.median(values):g}, max {max(values)} "
            f"across {len(values)} PR/workflow pairs"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
