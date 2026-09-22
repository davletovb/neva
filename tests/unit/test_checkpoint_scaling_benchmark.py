import json

import pytest

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

    result = run_benchmark([case], repeat=1, workdir=tmp_path)

    assert result["schema_version"] == 1
    assert result["python_version"]
    assert result["platform"]
    assert result["measurement_notes"]["thresholds"] == (
        "none; compare results on equivalent hardware"
    )
    assert [entry["case"]["name"] for entry in result["cases"]] == ["unit"]


def test_quick_cli_writes_json_output(tmp_path, capsys):
    output = tmp_path / "benchmark.json"

    assert main(["--profile", "quick", "--repeat", "1", "--output", str(output)]) == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["cases"][0]["case"]["name"] == "smoke"


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
