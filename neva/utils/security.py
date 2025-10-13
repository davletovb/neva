"""Utilities for performing security checks within Neva projects."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from neva.utils.exceptions import DependencyScanError, MissingDependencyError


@dataclass(frozen=True)
class VulnerabilityFinding:
    """Represents a single vulnerability identified during dependency scanning."""

    package: str
    version: str
    advisory: str
    cve: Optional[str] = None
    severity: Optional[str] = None
    fix_versions: Sequence[str] = field(default_factory=tuple)


def run_dependency_scan(requirements: Iterable[str] = ("requirements.txt",)) -> List[VulnerabilityFinding]:
    """Scan dependency requirement files for known vulnerabilities.

    Parameters
    ----------
    requirements:
        An iterable of requirement file paths that should be analysed. Each file is passed to
        :command:`pip-audit` and the JSON output is aggregated.

    Returns
    -------
    list[VulnerabilityFinding]
        A list of vulnerability findings. The list will be empty when no vulnerabilities are
        reported by :command:`pip-audit`.

    Raises
    ------
    MissingDependencyError
        If the ``pip-audit`` executable is not available on the current PATH.
    DependencyScanError
        If ``pip-audit`` returns an unexpected exit code or emits invalid JSON output.
    """

    pip_audit_executable = shutil.which("pip-audit")
    if pip_audit_executable is None:
        raise MissingDependencyError(
            "pip-audit is required to run dependency vulnerability scans. "
            "Install it with 'pip install pip-audit'."
        )

    findings: List[VulnerabilityFinding] = []
    for requirement_file in requirements:
        command = [pip_audit_executable, "-r", requirement_file, "--format", "json"]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode not in (0, 1):
            message = result.stderr.strip() or result.stdout.strip() or "pip-audit execution failed"
            raise DependencyScanError(
                f"pip-audit failed when analysing '{requirement_file}': {message}"
            )

        output = result.stdout.strip()
        if not output:
            continue

        try:
            audit_results = json.loads(output)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise DependencyScanError(
                f"pip-audit returned invalid JSON for '{requirement_file}'"
            ) from exc

        for package_report in audit_results:
            package_name = package_report.get("name", "")
            package_version = package_report.get("version", "")
            for vulnerability in package_report.get("vulns", []):
                findings.append(
                    VulnerabilityFinding(
                        package=package_name,
                        version=package_version,
                        advisory=(
                            vulnerability.get("advisory")
                            or vulnerability.get("id")
                            or "Unknown advisory"
                        ),
                        cve=vulnerability.get("cve"),
                        severity=vulnerability.get("severity"),
                        fix_versions=tuple(vulnerability.get("fix_versions") or ()),
                    )
                )

    return findings
