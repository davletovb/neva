import importlib
import runpy
from pathlib import Path


def test_quickstart_module_imports():
    module = importlib.import_module("examples.quickstart_conversation")
    assert hasattr(module, "main")


def test_quickstart_script_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).resolve().parents[2] / "examples" / "quickstart_conversation.py"
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "quickstart_metrics.json").exists()
