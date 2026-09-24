# Neva: Multi-Agent Simulations with Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-brightgreen.svg)](#development)
[![Issues](https://img.shields.io/github/issues/davletovb/neva)](https://github.com/davletovb/neva/issues)
[![Last Commit](https://img.shields.io/github/last-commit/davletovb/neva)](https://github.com/davletovb/neva/commits/main)
[![Contributors](https://img.shields.io/github/contributors/davletovb/neva)](https://github.com/davletovb/neva/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/davletovb/neva?style=social)](https://github.com/davletovb/neva/fork)
[![Stars](https://img.shields.io/github/stars/davletovb/neva?style=social)](https://github.com/davletovb/neva/stargazers)

<p align="center">
  <img src="https://github.com/davletovb/neva/assets/43503037/e9a2627b-e328-4986-a669-9ac13ad438b4" alt="Neva logo">
</p>

Neva is an open-source Python library for building simulations in which
configurable AI agents interact with one another and with shared environments.
It is designed for agent-based modeling, multi-agent experiments, tool-using
workflows, education, games, and research into emergent human/AI and AI/AI
behavior.

Neva keeps the main simulation model simple—agents, environments, schedulers,
tools, memory, and observers—while providing optional provider integrations,
streaming, persistence, recovery, reproducibility, observability, and execution
controls when a project needs them.

## Contents

- [Why Neva?](#why-neva)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Basic usage](#basic-usage)
- [Features](#features)
- [Examples](#examples)
- [Documentation](#documentation)
- [Development](#development)
- [Get involved](#get-involved)
- [License](#license)

## Why Neva?

- **Agent-based by design** — model interactions among multiple specialized
  agents rather than wrapping a single chat call.
- **Composable** — combine agents, schedulers, tools, environments, memory, and
  observers without committing to one application shape.
- **Provider-flexible** — use OpenAI-compatible APIs, Anthropic, Gemini, Grok,
  local/custom model backends, or deterministic stubs.
- **Simulation-friendly** — advance a world one turn at a time, inspect state,
  collect metrics, persist runs, and reproduce captured model-boundary behavior.
- **Useful offline** — the bundled quickstart and showcases can run with
  scripted backends and do not require API credentials.

## Quickstart

The minimal offline example creates two agents in a round-robin environment and
prints the collected simulation metrics:

```bash
python examples/quickstart_conversation.py
```

For the larger set of bundled scenarios:

```bash
python examples/run_showcase.py
```

These examples use deterministic or scripted backends by default, making them
safe to explore without provider credentials or API charges.

## Installation

Neva requires Python 3.11 or newer.

With Poetry:

```bash
git clone https://github.com/davletovb/neva.git
cd neva
poetry install
```

Optional extras keep the core installation lean:

- `tools` — Wikipedia, translation, Transformers/PyTorch tools;
- `providers` — additional provider SDK support;
- `memory` — FAISS/NumPy vector memory;
- `mlops` — MLflow integration;
- `observability` — OpenTelemetry SDK support;
- `budgeting` — model-aware token counting with `tiktoken`;
- `all` — all optional runtime extras.

For example:

```bash
poetry install --extras "providers memory observability"
```

or with pip:

```bash
pip install .[providers,memory,observability]
```

Install development dependencies with:

```bash
poetry install --with dev
```

## Basic usage

```python
from neva.agents import AgentManager
from neva.environments import BasicEnvironment
from neva.schedulers import RoundRobinScheduler


class ClassroomEnvironment(BasicEnvironment):
    def __init__(self, scheduler):
        super().__init__("Classroom", "A simple maths lesson", scheduler)
        self.transcript = []

    def context(self):
        if not self.transcript:
            return "Welcome students!"
        return "Recent discussion: " + " | ".join(self.transcript[-2:])

    def step(self):
        reply = super().step()
        if reply:
            self.transcript.append(reply)
        return reply


manager = AgentManager()
scheduler = RoundRobinScheduler()
environment = ClassroomEnvironment(scheduler)

teacher = manager.create_agent(
    "transformer",
    name="Teacher",
    llm_backend=lambda prompt: f"Teacher reflects on {prompt.split(':')[-1].strip()}",
)
student = manager.create_agent(
    "transformer",
    name="Student",
    llm_backend=lambda prompt: f"Student considers {prompt.split(':')[-1].strip()}",
)

environment.register_agent(teacher)
environment.register_agent(student)

for _ in range(4):
    print(environment.step())
```

Each `environment.step()` asks the scheduler for the next agent, builds the
environment context, invokes the agent, and records the turn through Neva's
observer lifecycle.

## Features

Neva currently includes:

- **Multiple agent types and backends** with a common simulation model.
- **Pluggable scheduling** including round robin, random, priority,
  least-recently-used, weighted, event-driven, conditional, and composite
  strategies.
- **Stateful conversations and checkpoints** for persisting and restoring
  supported simulation state.
- **Memory integrations** including optional FAISS-backed vector memory.
- **Provider coordination** for shared request-rate, concurrency, and optional
  spend budgets.
- **Streaming provider responses** with synchronous and asynchronous
  consumption.
- **Failure recovery** with configurable retry policies and durable failure
  records.
- **Tool guardrails** for schemas, permissions, approvals, concurrency, timeout,
  output, and optional isolated execution limits.
- **Bounded model/tool loops** for structured tool-using agent workflows.
- **Context controls** for character limits, retained conversation size, and
  optional provider/model-aware token envelopes.
- **Reproducibility utilities** for manifests, seeding, model-boundary recording,
  and deterministic offline replay.
- **Observability** through built-in simulation metrics, export helpers, MLflow,
  and optional OpenTelemetry instrumentation.
- **Privacy-conscious telemetry defaults** that omit raw model content unless
  explicitly enabled.

For configuration details, behavior boundaries, and caveats, see
[CAPABILITIES.md](CAPABILITIES.md).

## Examples

The `examples/` directory includes demonstrations of:

- multi-agent conversations;
- research/tool workflows;
- hierarchical coordination;
- structured debates;
- community planning;
- NPC and game-like interactions;
- productivity swarms;
- customer-support escalation;
- an opt-in live-provider smoke test.

The live-provider example is deliberately opt-in:

```bash
export OPENAI_API_KEY="your-key"
python examples/live_provider_smoke.py --live --max-spend-usd 0.01
```

Provider prices, token counts, and local spend controls are estimates; verify
current provider limits and billing before running live workloads.

## Documentation

- [Library capabilities](CAPABILITIES.md) — current runtime features,
  configuration concepts, and important boundaries.
- [Production-readiness acceptance plan](PRODUCTION_READINESS.md) —
  release-candidate checklist covering artifact, live-provider, operational,
  and deployment checks.
- [Documentation source](docs/) — guides and API documentation.
- [Examples](examples/) — runnable simulations and integrations.
- [Changelog](CHANGELOG.md) — chronological release and implementation history.
- [Contributing](CONTRIBUTING.md) — development and contribution workflow.
- [Security policy](SECURITY.md) — security reporting and supported practices.

## Development

After installing development dependencies, the main local checks are:

```bash
pytest
black --check .
isort --check-only .
flake8
mypy neva benchmarks
bandit -r neva benchmarks
pre-commit run --all-files
```

Build the Sphinx documentation with:

```bash
sphinx-build -b html docs docs/_build/html
```

A Dockerfile is also included for a reproducible containerized runtime.

## Get involved

Contributions are welcome through issues and pull requests. You can help by
reporting bugs, proposing features, improving documentation, adding examples,
or strengthening test coverage.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Neva is released under the MIT License. See [LICENSE](LICENSE).
