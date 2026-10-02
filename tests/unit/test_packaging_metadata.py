"""Packaging metadata consistency between pyproject extras and the pinned requirements.

The optional dependency surface is declared twice: ``[tool.poetry.extras]`` in
``pyproject.toml`` (what users install) and ``requirements-optional.txt`` (what
CI installs). These tests keep the two manifests in agreement, reject optional
dependencies that nothing imports, and keep the pinned-without-import list
explicit so that a new declaration has to make a deliberate choice.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_OPTIONAL = REPO_ROOT / "requirements-optional.txt"
PACKAGE_ROOT = REPO_ROOT / "neva"

# Distribution name -> the module that must appear in an import statement or an
# `importlib.import_module(...)` call. A new optional dependency must be added
# here, which forces the author to name the module the package actually loads.
IMPORT_NAME = {
    "wikipedia": "wikipedia",
    "deep-translator": "deep_translator",
    "transformers": "transformers",
    "torch": "torch",
    "mlflow": "mlflow",
    "anthropic": "anthropic",
    "httpx": "httpx",
    "google-genai": "google.genai",
    "faiss-cpu": "faiss",
    "numpy": "numpy",
    "opentelemetry-api": "opentelemetry",
    "opentelemetry-sdk": "opentelemetry",
    "tiktoken": "tiktoken",
}

# Optional dependencies that are deliberately pinned without a direct import in
# `neva/`, with the reason they are declared. Everything else must be imported.
PINNED_WITHOUT_DIRECT_IMPORT = {
    "torch": "runtime backend required by the declared transformers extra",
    "httpx": "compatibility pin for the google-genai SDK transport",
}


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _declared_optionals() -> dict[str, str]:
    dependencies = _load_pyproject()["tool"]["poetry"]["dependencies"]
    return {
        name: str(spec.get("version", ""))
        for name, spec in dependencies.items()
        if isinstance(spec, dict) and spec.get("optional") is True
    }


def _extras() -> dict[str, list[str]]:
    return _load_pyproject()["tool"]["poetry"]["extras"]


def _requirement_names() -> list[str]:
    names: list[str] = []
    for line in REQUIREMENTS_OPTIONAL.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*==", stripped)
        assert match, f"requirements-optional.txt must pin with '==': {stripped!r}"
        names.append(match.group(1))
    return names


def _package_source() -> str:
    return "\n".join(path.read_text() for path in sorted(PACKAGE_ROOT.rglob("*.py")))


def _is_imported(module_name: str, source: str) -> bool:
    direct = re.search(rf"^\s*(?:import|from)\s+{re.escape(module_name)}\b", source, re.M)
    dynamic = re.search(rf'import_module\(\s*"{re.escape(module_name)}', source)
    return bool(direct or dynamic)


def test_extras_reference_declared_optional_dependencies():
    optionals = _declared_optionals()
    for extra, names in _extras().items():
        for name in names:
            assert name in optionals, f"extra {extra!r} names undeclared dependency {name!r}"


def test_all_extra_is_the_union_of_the_other_extras():
    extras = _extras()
    expected = {name for extra, names in extras.items() if extra != "all" for name in names}
    assert set(extras["all"]) == expected


def test_every_optional_dependency_is_reachable_from_an_extra():
    covered = {name for names in _extras().values() for name in names}
    orphans = sorted(set(_declared_optionals()) - covered)
    assert not orphans, f"optional dependencies not exposed by any extra: {orphans}"


def test_budgeting_extra_declares_a_tokenizer_floor_with_the_gpt4o_mapping():
    """`openai_chat_counter` only accepts gpt-4o-mini, which needs tiktoken>=0.7.0."""

    declared = _declared_optionals()["tiktoken"]
    match = re.search(r"(\d+)\.(\d+)", declared)
    assert match, declared
    assert (int(match.group(1)), int(match.group(2))) >= (0, 7), declared

    pinned = re.search(r"^tiktoken==([\d.]+)$", REQUIREMENTS_OPTIONAL.read_text(), re.M)
    assert pinned, "requirements-optional.txt must pin tiktoken"
    assert tuple(int(part) for part in pinned.group(1).split(".")[:2]) >= (0, 7), pinned.group(1)


def test_requirements_optional_matches_the_extras():
    from_extras = {name for names in _extras().values() for name in names}
    from_requirements = set(_requirement_names())
    only_pyproject = sorted(from_extras - from_requirements)
    only_requirements = sorted(from_requirements - from_extras)
    assert not only_pyproject and not only_requirements, (
        f"only in pyproject extras: {only_pyproject}; "
        f"only in requirements-optional.txt: {only_requirements}"
    )


def test_requirements_optional_labels_every_extra():
    text = REQUIREMENTS_OPTIONAL.read_text()
    missing = sorted(
        extra for extra in _extras() if extra != "all" and f"# {extra} extra" not in text
    )
    assert not missing, f"requirements-optional.txt has no group comment for: {missing}"


def test_every_optional_dependency_maps_to_an_import_name():
    unmapped = sorted(set(_declared_optionals()) - set(IMPORT_NAME))
    assert not unmapped, f"add the import name for: {unmapped}"


def test_pinned_without_import_entries_stay_declared():
    stale = sorted(set(PINNED_WITHOUT_DIRECT_IMPORT) - set(_declared_optionals()))
    assert not stale, f"remove stale pinned-without-import entries: {stale}"


def test_optional_dependencies_are_imported_or_explicitly_pinned():
    source = _package_source()
    unused = []
    for name in sorted(_declared_optionals()):
        if name in PINNED_WITHOUT_DIRECT_IMPORT:
            continue
        if not _is_imported(IMPORT_NAME[name], source):
            unused.append(name)
    assert not unused, (
        "declared optional dependencies that nothing imports (drop them or add "
        f"them to PINNED_WITHOUT_DIRECT_IMPORT with a reason): {unused}"
    )
