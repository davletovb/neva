# Neva: Creating Multi-Agent Simulations with Large Language Models!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/davletovb/neva)](https://github.com/davletovb/neva/issues)
[![Last Commit](https://img.shields.io/github/last-commit/davletovb/neva)](https://github.com/davletovb/neva/commits/master)
[![Contributors](https://img.shields.io/github/contributors/davletovb/neva)](https://github.com/davletovb/neva/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/davletovb/neva?style=social)](https://github.com/davletovb/neva/fork)
[![Stars](https://img.shields.io/github/stars/davletovb/neva?style=social)](https://github.com/davletovb/neva/stargazers)

<p align="center">
  <img src="https://github.com/davletovb/neva/assets/43503037/e9a2627b-e328-4986-a669-9ac13ad438b4" alt="Intellibot Logo">
</p>

Want to create worlds where AI agents come alive? 🤖 Our open-source library lets you easily build captivating simulations where customizable agents interact using natural language.💬 Immerse yourself in emergent behaviors as your intelligent assistants, characters, and communities cooperate in exciting new ways! 🔥 Watch the dynamics unfold as your agents chat, explore, and work together powered by state-of-the-art LLMs. 💡 Join our open-source movement to make agent-based modeling more accessible, from classrooms to cutting-edge research. 🧑‍🏫👩‍🔬  

**We seek Python developers passionate about integrating LLMs to collaborate on an open-source library for modeling complex emergent behaviors and human-AI cooperation dynamics.**

## Table of Contents
- [Why Neva?](#why-neva)
- [Quickstart 🚀](#quickstart-)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Get Involved 🤝](#get-involved-)
- [License](#license)

## Why Neva?
Neva makes it simple to unlock the power of large language models through specialized AI agents. Benefits include:

- **Rapid Prototyping** - Quickly build conversational agents, productivity tools, game characters, and more.
- **Modular Components** - Switch between different types of LLMs. Mix and match configurable agents, tools, and environments.
- **Scalable Systems** - Develop complex ecosystems of hierarchical, coordinating agents.
- **Simulation Capabilities** - Model and evaluate agent behaviors under various conditions.
- **Community Resources** - Leverage shared agents, tools, and examples from our open-source community.

Neva empowers developers, researchers, educators, and hobbyists to create the next generation of AI interactions for gaming, automation, education, research, and beyond!

## Quickstart 🚀
The `examples/quickstart_conversation.py` script showcases a minimal
conversation between two agents scheduled in an environment. It uses the
high-level `AgentManager`, the concrete `TransformerAgent` class with stubbed
language-model backends, and the round-robin scheduler to coordinate the
interaction loop.

```bash
python examples/quickstart_conversation.py
```

Each call to `environment.step()` advances the simulation by a single message so
you can observe the emergent collaboration unfold in just a few lines of code.
When the script finishes it prints a snapshot of the automatically collected
metrics (turn counts, participation rates, dialogue length, tool usage, and
lightweight sentiment/intent analysis) and writes them to
`quickstart_metrics.json` for later inspection.

### Expanded showcases

Looking for richer demonstrations? The `examples/showcase_use_cases.py`
module strings together four self-contained simulations that spotlight core
Neva capabilities:

```bash
python examples/showcase_use_cases.py
```

The showcase highlights tool-augmented research with a stubbed Wikipedia
integration, hierarchical leader–follower coordination, emergent roleplay
between game NPCs, and a productivity swarm that self-organises around shared
tasks. Each vignette runs fully offline using scripted LLM backends while still
triggering the observer’s tool-usage metrics.

### Observability & Experiment Tracking

The :class:`observer.SimulationObserver` now registers a suite of metrics out of
the box, removing the need to manually wire analytics into every experiment. It
tracks turn counts, per-agent participation, response latencies, recent
sentiment/intent, and tool-usage frequency. Metrics can be exported to CSV/JSON
or logged to MLflow with `SimulationObserver.log_to_mlflow()` to integrate with
your preferred dashboarding stack.

### OpenTelemetry Observability

Neva ships with a vendor-neutral instrumentation layer powered by
OpenTelemetry. Call `neva.utils.telemetry.configure_telemetry()` once during
initialisation to emit traces, metrics, and structured logs for every
conversation turn, LLM API invocation, and tool call. The helper exposes the
standard OpenTelemetry providers so you can attach any exporter supported by
your observability stack:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk._logs.export import ConsoleLogExporter
from neva.utils.telemetry import configure_telemetry

telemetry = configure_telemetry(
    span_exporter=ConsoleSpanExporter(),
    log_exporter=ConsoleLogExporter(),
)
```

Once configured you can trace entire conversation flows, monitor response
latency and token usage, and inspect chain-of-thought reasoning and tool
sequences across complex multi-agent simulations without being tied to a single
vendor.

## Installation
Clone the repository and install dependencies with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/davletovb/neva.git
cd neva
poetry install
```

Poetry keeps the default installation lean—only the always-on dependencies ship
with the core project. Install additional bundles as needed:

- **Development tools** – `poetry install --with dev`
- **Research tools (translation, summarisation, encyclopedia lookups)** –
  `poetry install --extras "tools"`
- **ML experiment tracking** – `poetry install --extras "mlops"`
- **Additional LLM providers (Anthropic Claude, Google Gemini, xAI Grok)** –
  `poetry install --extras "providers"`
- **Everything** – `poetry install --extras "all" --with dev`

Prefer `pip`? The same extras are available via `pip install .[tools]`,
`pip install .[providers]`, or `pip install .[all]`. Updated
`requirements-*.txt` files are provided for
environments that cannot yet adopt Poetry.

### Developer Setup

To contribute or run the full suite of examples you may need a few additional
configuration steps:

- **API keys** – supply the relevant key (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or an xAI token) when using
  `GPTAgent`. Select a backend by passing `provider="openai"`,
  `"anthropic"`, `"gemini"`, or `"grok"`. Community members often rely on
  [OpenAI compatible endpoints](https://platform.openai.com/docs/api-reference/introduction), but any drop-in replacement that matches the Chat
  Completions API works.
- **Transformers cache** – the `TransformerAgent` loads Hugging Face models on
  demand. Install the optional `torch` dependency and authenticate with
  `huggingface-cli login` if you plan to download private models.
- **Environment variables** – place sensitive credentials (API keys, database
  URLs, etc.) in a `.env` file and load them via `python-dotenv` or your
  preferred secrets manager when running examples.
- **Optional heavyweight dependencies** – translation is powered by
  `deep-translator` (a maintained wrapper around the Google Translate service) and
  summarisation leverages Hugging Face's `transformers` (T5 by default). Install
  the `tools` extra—or the individual packages—only when you require these
  capabilities to keep the core installation lightweight.
- **Structured logging** – call ``logging_utils.configure_logging()`` at the
  beginning of your experiment to emit JSON logs ready for ingestion by ELK,
  Loki, or any observability platform.
- **Graceful fallbacks** – the built-in tools surface actionable error messages
  when optional packages such as `wikipedia`, `deep-translator`, or
  `transformers` are unavailable. You can provide lightweight factories to
  `TranslatorTool` and `SummarizerTool` (or monkeypatch the
  Wikipedia backend) to keep experiments fully offline.

### Docker

Need a reproducible runtime? Build the included container image:

```bash
docker build -t neva:latest .
docker run --rm -it neva:latest python examples/quickstart_conversation.py
```

The Dockerfile uses Poetry to install the project and accepts optional build
arguments (`WITH_EXTRAS=tools,mlops`) to bake in additional extras when
required. Mount your local workspace with `-v $PWD:/workspace` for rapid
iteration.

After installing dependencies run `pytest` to confirm the environment is ready
for development.

## Usage
Create and simulate AI agents effortlessly:
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
            return "Welcome students!"  # initial observation
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

Each call to `environment.step()` asks the scheduler for an agent, collects
metrics via the observer system, and invokes the agent's `respond` method with
the current environmental context. This mirrors the simulation lifecycle used
throughout the library, the quickstart script, and the accompanying tests.

## Features
- **Flexible & Adaptable**: Adapt to various types of LLMs, tasks, and tools.
- **Stateful Agents**: Built-in conversation state tracking and snapshot/restore
  helpers let you persist simulations mid-run and resume them later.
- **Long-Term Memory Integrations**: Plug in semantic vector stores like FAISS
  to give agents durable recall of historical conversations and research notes.
- **Robust Safety Rails**: Prompt validation, sanitisation, rate limiting, and
  automatic retry logic keep API usage safe and predictable.
- **Observability First**: Structured logging, response-time metrics, token and
  cost tracking, and conversation summaries surface actionable insights out of
  the box.
- **Intuitive Interfaces**: Simple interfaces for agent creation and management.
- **Environment Simulation**: Simulate environments for agent interactions and collaborations.

## Get Involved 🤝

Neva thrives on community collaboration, and we welcome your participation through issues and pull requests! Whether you're fixing a bug, proposing a new feature, or enhancing our documentation, your contributions are the heartbeat of our project.

Before jumping in, take a moment to read our [contribution guidelines.](https://github.com/davletovb/neva/blob/main/CONTRIBUTING.md)

- **Report bugs** by opening a GitHub issue.
- **Suggest enhancements** through the issues tracker.
- **Improve documentation** with pull requests.
- **Share examples** of Neva projects and use cases.
- **Add tests** to improve coverage.

We value each contribution, no matter how big or small, and we're excited to see what you'll bring to our growing community!

## License
This project is licensed under the MIT License. See the [License File](https://github.com/davletovb/neva/blob/main/LICENSE) for more details.
