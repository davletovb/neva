import json

import pytest

import benchmarks.checkpoint_scaling as checkpoint_scaling
from benchmarks.checkpoint_scaling import BenchmarkCase, main, run_benchmark, run_case


def test_run_case_reports_checkpoint_stage_metrics_and_cleans_files(tmp_path):
    case = BenchmarkCase(
        name="unit",
        agents=2,
        turns_per_agent=3,
        message_chars=32,
        environment_bytes=128,
    )

    result = run_case(case, repeat=2, workdir=tmp_path)

    assert result["case"] == {
        "name": "unit",
        "agents": 2,
        "turns_per_agent": 3,
        "message_chars": 32,
        "environment_bytes": 128,
    }
    assert result["repeat"] == 2
    assert result["median_checkpoint_bytes"] > 0
    assert len(result["samples"]) == 2
    assert list(tmp_path.iterdir()) == []

    for stage in ("create_snapshot", "save_snapshot", "load_snapshot"):
        assert result["stages"][stage]["median_ms"] >= 0
        assert result["stages"][stage]["median_peak_python_bytes"] >= 0


def test_run_benchmark_includes_machine_and_measurement_metadata(tmp_path):
    case = BenchmarkCase(
        name="unit",
        agents=1,
        turns_per_agent=1,
        message_chars=8,
        environment_bytes=8,
    )

    result = run_benchmark(
        [case],
        repeat=1,
        workdir=tmp_path,
        git_sha="deadbeef",
    )

    assert result["schema_version"] == 1
    assert result["python_version"]
    assert result["platform"]
    assert result["git_sha"] == "deadbeef"
    assert "untraced" in result["measurement_notes"]["elapsed"]
    assert "separate tracemalloc pass" in result["measurement_notes"]["memory"]
    assert result["measurement_notes"]["thresholds"] == (
        "none; compare results on equivalent hardware"
    )
    assert [entry["case"]["name"] for entry in result["cases"]] == ["unit"]


def test_quick_cli_writes_json_output(tmp_path, capsys):
    output = tmp_path / "benchmark.json"

    workdir = tmp_path / "checkpoint-storage"
    assert (
        main(
            [
                "--profile",
                "quick",
                "--repeat",
                "1",
                "--workdir",
                str(workdir),
                "--git-sha",
                "cafebabe",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["cases"][0]["case"]["name"] == "smoke"
    assert file_payload["git_sha"] == "cafebabe"
    assert workdir.exists()
    assert list(workdir.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    [
        {"name": "", "agents": 1, "turns_per_agent": 0, "message_chars": 0, "environment_bytes": 0},
        {
            "name": "bad",
            "agents": 0,
            "turns_per_agent": 0,
            "message_chars": 0,
            "environment_bytes": 0,
        },
        {
            "name": "bad",
            "agents": 1,
            "turns_per_agent": -1,
            "message_chars": 0,
            "environment_bytes": 0,
        },
        {
            "name": "bad",
            "agents": 1,
            "turns_per_agent": 0,
            "message_chars": -1,
            "environment_bytes": 0,
        },
        {
            "name": "bad",
            "agents": 1,
            "turns_per_agent": 0,
            "message_chars": 0,
            "environment_bytes": -1,
        },
    ],
)
def test_invalid_benchmark_cases_are_rejected(case):
    with pytest.raises(ValueError):
        BenchmarkCase(**case)


def test_measure_uses_separate_untraced_timing_and_traced_memory_passes():
    tracing_states = []

    def operation():
        tracing_states.append(checkpoint_scaling.tracemalloc.is_tracing())
        return "result"

    result, metrics = checkpoint_scaling._measure(operation)

    assert result == "result"
    assert tracing_states == [False, True]
    assert metrics["elapsed_ms"] >= 0
    assert metrics["peak_python_bytes"] >= 0


def test_invalid_repeat_is_rejected():
    case = BenchmarkCase(
        name="unit",
        agents=1,
        turns_per_agent=0,
        message_chars=0,
        environment_bytes=0,
    )
    with pytest.raises(ValueError, match="repeat must be positive"):
        run_case(case, repeat=0)


def test_run_case_preserves_existing_files_in_workdir(tmp_path):
    existing = tmp_path / "unit-0.json"
    existing.write_text("keep me", encoding="utf-8")
    case = BenchmarkCase(
        name="unit",
        agents=1,
        turns_per_agent=1,
        message_chars=8,
        environment_bytes=8,
    )

    run_case(case, repeat=1, workdir=tmp_path)

    assert existing.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.iterdir()) == [existing]


def test_cli_rejects_non_positive_repeat():
    with pytest.raises(SystemExit) as exc_info:
        main(["--profile", "quick", "--repeat", "0"])

    assert exc_info.value.code == 2


def test_git_sha_defaults_to_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "from-env")
    case = BenchmarkCase(
        name="unit",
        agents=1,
        turns_per_agent=1,
        message_chars=8,
        environment_bytes=8,
    )

    result = run_benchmark([case], repeat=1, workdir=tmp_path)

    assert result["git_sha"] == "from-env"
