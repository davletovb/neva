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


def collect_runs(repo: str, days: int) -> list[dict[str, Any]]:
    today = dt.datetime.now(dt.timezone.utc).date()
    dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(days)]

    def one_day(date: str) -> list[dict[str, Any]]:
        pages = gh_api(f"repos/{repo}/actions/runs?per_page=100&created={date}")
        return flatten_pages(pages, "workflow_runs")

    runs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for day_runs in pool.map(one_day, dates):
            runs.extend(day_runs)
    return runs


def collect_jobs(repo: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def one_run(run: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = run["id"]
        attempts = max(1, int(run.get("run_attempt") or 1))
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
            for job in flatten_pages(pages, "jobs"):
                if not isinstance(job, dict):
                    continue
                result.append(
                    {
                        "run": run_id,
                        "workflow": run.get("name", ""),
                        "event": run.get("event", ""),
                        "branch": run.get("head_branch", ""),
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
    print("|" + "|".join("---:" if i < len(headers) - 1 else "---" for i in range(len(headers))) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="OWNER/REPO; defaults to current gh repository")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", help="Optional path for raw job JSON")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be >= 1")

    try:
        repo = args.repo or detect_repo()
        repo_info_pages = gh_api(f"repos/{repo}")
        repo_info = repo_info_pages[0] if repo_info_pages else {}
        runs = collect_runs(repo, args.days)
        print(f"{len(runs)} runs in {args.days} days; reading jobs...", file=sys.stderr)
        jobs = collect_jobs(repo, runs)
    except (RuntimeError, json.JSONDecodeError) as exc:
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
    cancelled = sum(billed_estimate(job) for job in hosted if job.get("conclusion") == "cancelled")
    failed = sum(billed_estimate(job) for job in hosted if job.get("conclusion") == "failure")

    print(f"# GitHub Actions usage: {repo}, last {args.days} days\n")
    if repo_info.get("private") is False:
        print("> Public repository: standard hosted runners are generally not a direct minutes charge; use these figures mainly to compare runner usage and speed.\n")

    print(f"- Runs: {len(runs)}; jobs that ran: {len(ran)}")
    print(
        "- Standard hosted runner-minute estimate: "
        f"**{fmt_int(total)}** (about {fmt_int(total * 30 / args.days)} per 30 days)"
    )
    print(f"- Per-job minute rounding estimate: {fmt_int(rounding)} ({pct(rounding, total)})")
    print(f"- Cancelled: {fmt_int(cancelled)}; failed: {fmt_int(failed)} estimated runner minutes")
    if other:
        groups = sorted({str(job.get("runner_group_name") or "self-hosted/other") for job in other})
        raw_other = sum(duration_minutes(job.get("started_at"), job.get("completed_at")) for job in other)
        print(f"- Other runner groups ({', '.join(groups)}): {fmt_int(raw_other)} raw minutes, excluded from the hosted estimate")

    workflow_runs = Counter((run.get("name", ""), run.get("event", "")) for run in runs)
    by_flow: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"billed": 0, "count": 0, "raw": 0.0})
    by_job: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"billed": 0, "count": 0, "raw": 0.0, "runs": set()}
    )

    for job in hosted:
        flow_key = (str(job.get("workflow", "")), str(job.get("event", "")))
        job_key = (*flow_key, str(job.get("job", "")))
        duration = duration_minutes(job.get("started_at"), job.get("completed_at"))
        estimate = billed_estimate(job)

        by_flow[flow_key]["billed"] += estimate
        by_flow[flow_key]["count"] += 1
        by_flow[flow_key]["raw"] += duration

        by_job[job_key]["billed"] += estimate
        by_job[job_key]["count"] += 1
        by_job[job_key]["raw"] += duration
        by_job[job_key]["runs"].add(job.get("run"))

    print("\n## By workflow and event\n")
    flow_rows: list[list[str]] = []
    for (workflow, event), values in sorted(by_flow.items(), key=lambda item: item[1]["billed"], reverse=True)[:20]:
        flow_rows.append(
            [
                fmt_int(values["billed"]),
                pct(values["billed"], total),
                str(values["count"]),
                f'{values["raw"] / values["count"]:.1f}',
                f"{workflow} / {event}",
            ]
        )
    print_table(["Est. min", "Share", "Jobs", "Avg raw min", "Workflow / event"], flow_rows)

    print("\n## Top jobs\n")
    job_rows: list[list[str]] = []
    for (workflow, event, name), values in sorted(by_job.items(), key=lambda item: item[1]["billed"], reverse=True)[:30]:
        run_count = workflow_runs[(workflow, event)]
        ran_in = len(values["runs"])
        job_rows.append(
            [
                fmt_int(values["billed"]),
                pct(values["billed"], total),
                str(values["count"]),
                f'{values["raw"] / values["count"]:.1f}',
                pct(ran_in, run_count),
                f"{workflow} / {event} :: {name}",
            ]
        )
    print_table(["Est. min", "Share", "Jobs", "Avg raw min", "Ran in", "Name"], job_rows)

    durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        if run.get("conclusion") != "success":
            continue
        minutes = duration_minutes(run.get("run_started_at"), run.get("updated_at"))
        if minutes > 0:
            durations[(str(run.get("name", "")), str(run.get("event", "")))].append(minutes)

    print("\n## Wall-clock time of successful runs\n")
    duration_rows: list[list[str]] = []
    for (workflow, event), values in sorted(durations.items(), key=lambda item: len(item[1]), reverse=True)[:15]:
        duration_rows.append(
            [
                str(len(values)),
                f"{statistics.median(values):.1f}",
                f"{quantile(values, 0.90):.1f}",
                f"{workflow} / {event}",
            ]
        )
    print_table(["Runs", "Median min", "p90 min", "Workflow / event"], duration_rows)

    pr_counts = Counter(
        (str(run.get("name", "")), str(run.get("head_branch", "")))
        for run in runs
        if run.get("event") == "pull_request"
    )
    if pr_counts:
        values = sorted(pr_counts.values())
        print("\n## Pull-request churn\n")
        print(
            "- Runs per branch/workflow pair: "
            f"median {statistics.median(values):g}, max {max(values)} "
            f"across {len(values)} branch/workflow pairs"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
