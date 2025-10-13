import json
from types import SimpleNamespace

import pytest

from neva.utils import security
from neva.utils.exceptions import DependencyScanError, MissingDependencyError


class DummyCompletedProcess(SimpleNamespace):
    pass


def test_run_dependency_scan_requires_pip_audit(monkeypatch) -> None:
    monkeypatch.setattr(security.shutil, "which", lambda _: None)

    with pytest.raises(MissingDependencyError):
        security.run_dependency_scan(["requirements.txt"])


def test_run_dependency_scan_parses_vulnerabilities(monkeypatch) -> None:
    fake_output = json.dumps(
        [
            {
                "name": "example",
                "version": "1.0.0",
                "vulns": [
                    {
                        "id": "PYSEC-0001",
                        "advisory": "Example vulnerability",
                        "cve": "CVE-0000-0001",
                        "severity": "high",
                        "fix_versions": ["1.0.1"],
                    }
                ],
            }
        ]
    )

    monkeypatch.setattr(security.shutil, "which", lambda _: "/usr/bin/pip-audit")
    monkeypatch.setattr(
        security.subprocess,
        "run",
        lambda *args, **kwargs: DummyCompletedProcess(
            stdout=fake_output, stderr="", returncode=1
        ),
    )

    findings = security.run_dependency_scan(["requirements.txt"])

    assert len(findings) == 1
    finding = findings[0]
    assert finding.package == "example"
    assert finding.version == "1.0.0"
    assert finding.cve == "CVE-0000-0001"
    assert finding.fix_versions == ("1.0.1",)


def test_run_dependency_scan_raises_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(security.shutil, "which", lambda _: "/usr/bin/pip-audit")
    monkeypatch.setattr(
        security.subprocess,
        "run",
        lambda *args, **kwargs: DummyCompletedProcess(
            stdout="", stderr="unexpected failure", returncode=2
        ),
    )

    with pytest.raises(DependencyScanError):
        security.run_dependency_scan(["requirements.txt"])
