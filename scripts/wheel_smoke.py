"""Smoke-test an installed Neva wheel.

Run by the ``package`` CI job against a wheel installed into a clean virtual
environment. The script asserts that the distribution ships the expected
package data and extras metadata and that a small offline simulation runs end to
end, so packaging regressions fail in CI instead of at first import for users.

Run it with the interpreter of the environment that has the wheel installed::

    python scripts/wheel_smoke.py

The script never imports the repository checkout: it fails if ``neva`` resolves
to a source tree instead of an installed distribution.
"""

from __future__ import annotations

import pathlib
import re
import sys
from importlib import metadata
from typing import NoReturn

EXPECTED_EXTRAS = {"all", "budgeting", "memory", "mlops", "observability", "providers", "tools"}


def fail(message: str) -> NoReturn:
    print(f"wheel smoke: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_installed_location() -> pathlib.Path:
    import neva

    root = pathlib.Path(neva.__file__).resolve().parent
    if (root.parent / "pyproject.toml").is_file() or (root.parent / "setup.py").is_file():
        fail(f"imported Neva from a source checkout, not an installed wheel: {root}")
    return root


def check_package_data(root: pathlib.Path) -> None:
    if not (root / "py.typed").is_file():
        fail("py.typed is missing from the installed package")


def check_wheel_contents() -> None:
    files = metadata.files("neva")
    if not files:
        fail("distribution metadata lists no files")
    unexpected = sorted(
        str(path)
        for path in files
        if not str(path).startswith("neva/") and ".dist-info/" not in str(path)
    )
    if unexpected:
        fail(f"wheel ships files outside neva/: {unexpected[:5]}")


def check_extras_metadata() -> None:
    requires = metadata.requires("neva") or []
    optional = [requirement for requirement in requires if "extra ==" in requirement]
    provided = set()
    for requirement in optional:
        provided.update(re.findall(r'extra == "([^"]+)"', requirement))
    missing = EXPECTED_EXTRAS - provided
    if missing:
        fail(f"extras missing from distribution metadata: {sorted(missing)}")

    names = {requirement.split(";")[0].split("(")[0].strip().lower() for requirement in optional}
    if "tiktoken" not in names:
        fail("the budgeting extra does not declare tiktoken")


def run_offline_simulation() -> None:
    from neva.agents import AgentManager
    from neva.environments import BasicEnvironment
    from neva.memory import ShortTermMemory
    from neva.schedulers import RoundRobinScheduler
    from neva.utils.context_budget import ModelContextBudget

    def backend(prompt: str) -> str:
        return f"ack: {prompt.split(':')[-1].strip()[:20]}"

    manager = AgentManager()
    scheduler = RoundRobinScheduler()
    environment = BasicEnvironment("smoke", "wheel smoke environment", scheduler)
    agent = manager.create_agent(
        "transformer",
        name="Smoke",
        llm_backend=backend,
        memory=ShortTermMemory(capacity=2, label="smoke"),
    )
    environment.register_agent(agent)
    steps = list(environment.run(2))
    if len(steps) != 2 or not all(steps):
        fail(f"offline simulation produced unexpected output: {steps}")

    def smoke_tokens(request) -> int:
        assert isinstance(request, list)
        return 3 + sum(3 + len(message["role"]) + len(message["content"]) for message in request)

    budget = ModelContextBudget(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=8192,
        count_request_tokens=smoke_tokens,
        counter_id="wheel-smoke-v1",
    )
    if budget.provider != "openai" or budget.max_tokens != 8192:
        fail("ModelContextBudget did not preserve its configuration")


def main() -> None:
    root = check_installed_location()
    check_package_data(root)
    check_wheel_contents()
    check_extras_metadata()
    run_offline_simulation()
    print(f"wheel smoke: OK (neva {metadata.version('neva')} at {root})")


if __name__ == "__main__":
    main()
