# Neva: Creating Multi-Agent Simulations with Large Language Models!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-brightgreen.svg)](#testing--quality-assurance)
[![Issues](https://img.shields.io/github/issues/davletovb/neva)](https://github.com/davletovb/neva/issues)
[![Last Commit](https://img.shields.io/github/last-commit/davletovb/neva)](https://github.com/davletovb/neva/commits/main)
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
- [Testing & Quality Assurance](#testing--quality-assurance)
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

Looking for richer demonstrations? Each advanced simulation now lives in its
own module under `examples/`, making it easier to explore or extend a single
scenario. You can run everything at once with the convenience wrapper:

```bash
python examples/run_showcase.py
```

Or execute an individual vignette directly, for example the customer support
triage flow:

```bash
python examples/customer_support_demo.py
```

The showcase lineup covers tool-augmented research with a stubbed Wikipedia
integration, hierarchical leader–follower coordination, structured debates,
community planning dynamics, emergent NPC roleplay, tactical party combat,
productivity swarms, and a customer support escalation. Every script runs fully
offline using scripted LLM backends while still triggering the observer’s
tool-usage metrics.

### Opt-in live-provider smoke example

`examples/live_provider_smoke.py` makes the difference between scripted and
generated behavior explicit. Its default run is a deterministic offline stub:

```bash
python examples/live_provider_smoke.py
```

To generate one answer using OpenAI, provide your own API key and an explicit
ceiling on Neva's **estimated** spend (USD):

```bash
export OPENAI_API_KEY="your-key-from-openai"
python examples/live_provider_smoke.py --live --max-spend-usd 0.01
```

The live path uses `gpt-4o-mini`, at most 128 output tokens, a 2,000-character
formatted context cap, a 15-second request timeout, and zero retries. It makes
one API call and prints the generated answer and estimated spend. The script
requires both the `--live` flag and the budget; no API call occurs in the
default mode. Neva's built-in `CostTracker` prices are static estimates, and
its spend budget reserves based on estimated prompt tokens and configured
maximum output tokens. Neither the budget nor the context character cap is a
provider billing or token-context guarantee. Check
[current OpenAI pricing](https://platform.openai.com/docs/pricing) and your
account's usage before running the live path; provider charges may differ.
The test suite stubs the HTTP response and never uses a real key or incurs a
provider charge.

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
from neva.utils.telemetry import configure_telemetry

telemetry = configure_telemetry(
    span_exporter=ConsoleSpanExporter(),
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

## Testing & Quality Assurance

Comprehensive automated checks keep the codebase healthy and maintain a minimum of 80% test coverage. The following commands mirror the CI pipeline and can be executed locally after installing the development dependencies with `pip install -r requirements-dev.txt` or `poetry install --with dev`:

```bash
# Run the complete pytest suite with coverage reports (HTML in htmlcov/ and XML in coverage.xml)
pytest

# Execute formatting, import-sorting, linting, typing, and security checks individually
black --check .
isort --check-only .
flake8
mypy neva benchmarks
bandit -r neva benchmarks

# Run the equivalent pre-commit hooks
pre-commit run --all-files

# Build the documentation locally
sphinx-build -b html docs docs/_build/html
```

Generated coverage reports appear in `coverage.xml` and the `htmlcov/` directory. The coverage badge above reflects the guaranteed baseline enforced by CI.


### Checkpoint scaling benchmark

Use the repository-local benchmark to measure checkpoint creation, streamed save,
and load behavior on the machine where Neva will run:

```bash
python -m benchmarks.checkpoint_scaling --profile standard --repeat 3 \
  --workdir /path/to/checkpoint-storage --git-sha "$(git rev-parse HEAD)" \
  --output checkpoint-benchmark.json
```

The standard profile runs three increasing deterministic conversation sizes. Use
`--profile quick --repeat 1` for a smoke run. Results include checkpoint bytes,
wall-clock milliseconds from an untraced pass, peak Python allocations from a
separate `tracemalloc` pass for `create_snapshot`, `save_snapshot`, and
`load_snapshot`, and each stage's peak-Python-allocation / serialized-byte
amplification ratio. Each stage therefore executes twice per sample. Use the
ratio to identify workloads where application-specific externalization of large
payloads may be worthwhile. The benchmark has no performance pass/fail
threshold: compare runs on equivalent hardware and use `--workdir` to measure
the filesystem that will hold real checkpoints.
`--git-sha` records a local revision explicitly (otherwise `GITHUB_SHA` is
used when available). `tracemalloc` does not include OS page cache or
temporary/destination file space.

### Developer Setup

To contribute or run the full suite of examples you may need a few additional
configuration steps:

- **API keys** – supply the relevant key (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or an xAI token) when using
  `GPTAgent`. The providers extra constrains `httpx<0.28` because the
  declared Anthropic 0.26 client still uses the pre-0.28 proxy constructor API. Select a backend by passing `provider="openai"`,
  `"anthropic"`, `"gemini"`, or `"grok"`. Community members often rely on
  [OpenAI compatible endpoints](https://platform.openai.com/docs/api-reference/introduction), but any drop-in replacement that matches the Chat
  Completions API works.
- **Transformers runtime and cache** – the `TransformerAgent` loads Hugging
  Face models on demand. The `tools` extra now includes both `transformers`
  and its PyTorch backend; authenticate with `huggingface-cli login` only if
  you plan to download private models. CI exercises the agent with locally
  generated/saved tiny T5 weights, so the test suite itself needs no model
  download.
- **Environment variables** – place sensitive credentials (API keys, database
  URLs, etc.) in a `.env` file and load them via `python-dotenv` or your
  preferred secrets manager when running examples.
- **Optional heavyweight dependencies** – translation is powered by
  `deep-translator` (a maintained wrapper around the Google Translate service) and
  summarisation leverages Hugging Face's `transformers` (T5 by default). Install
  the `tools` extra—or the individual packages—only when you require these
  capabilities to keep the core installation lightweight. The `memory` extra
  installs FAISS/NumPy; CI has a dependency-enabled FAISS job so that path is
  exercised rather than skipped in every normal lightweight test job.
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

### Streaming provider responses

Built-in `GPTAgent` providers can stream text without changing the ordinary
`respond()` and environment turn APIs. Use a single session once; consume
`delta` events for display and use the `complete` event as the committed answer:

```python
from neva.agents import GPTAgent
from neva.agents.streaming import StreamInterruptedError

agent = GPTAgent(provider="openai", api_key="YOUR_API_KEY")
session = agent.stream_response("Explain the result", max_queue_size=8)
try:
    for event in session:
        if event.kind == "delta":
            print(event.text, end="", flush=True)
        else:
            print("\nfirst token:", event.first_token_seconds)
            print("completion:", event.completion_seconds)
except StreamInterruptedError as exc:
    print("Uncommitted partial response:", exc.partial_text)
finally:
    session.close()
```

The same session supports `async for event in session`; call `await
session.aclose()` when stopping early. The producer blocks when the bounded
queue fills. `close()` signals cancellation, but a synchronous provider read
can take up to `request_timeout` to return. The response enters history and
cache only when the consumer accepts the `complete` event. A provider call
that finishes after the consumer closes can still incur and record spend.
A failure after text is emitted raises `StreamInterruptedError` with its
partial text; it is never retried or cached and is not added to conversation
history. Failures before output can
use configured retries. Completed output is capped by `max_response_chars`
(default one million characters) in addition to the provider's output-token
setting. Streaming uses the same shared provider admission, circuit breaker,
spend reservations, and token/cost trackers as ordinary responses. The
first-token and full-completion durations are exposed on the terminal event
and in LLM telemetry (`neva.llm.first_token.latency` and
`neva.llm.api.latency`); interrupted streams emit their available latency
measurements with an interrupted status. Cached completions have zero provider
latency. Custom synchronous `llm_backend` implementations and model-backend
wrappers require their own streaming adapters and are rejected by this API.

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

### Built-in scheduling strategies

Neva ships with a pluggable scheduler registry (`neva.schedulers.register_scheduler`)
and a suite of ready-to-use strategies:

* `RoundRobinScheduler` – cycle through agents in a fixed order.
* `RandomScheduler` – pick an available agent at random.
* `PriorityScheduler` – prefer agents with higher priority values.
* `LeastRecentlyUsedScheduler` – activate the agent that has waited the longest.
* `WeightedRandomScheduler` – random sampling with configurable weights.
* `EventDrivenScheduler` – run agents when they emit events.
* `ConditionalScheduler` – run agents only when a predicate on their state is met.
* `CompositeScheduler` – nest schedulers to manage agent sub-groups.

Custom schedulers can be registered at runtime:

```python
from neva.schedulers import register_scheduler, create_scheduler, Scheduler


class MyScheduler(Scheduler):
    ...


register_scheduler("my_scheduler", MyScheduler)
scheduler = create_scheduler("my_scheduler")
```

## Features
- **Flexible & Adaptable**: Adapt to various types of LLMs, tasks, and tools.
- **Stateful Agents**: Built-in conversation state tracking and snapshot/restore
  helpers let you persist simulations mid-run and resume them later. Stored
  history is unlimited by default, or can be bounded explicitly with
  `ConversationState(max_turns=..., max_turn_bytes=..., max_history_bytes=...)`.
  `max_turn_bytes` bounds each stored message; `max_history_bytes` bounds the
  aggregate UTF-8 bytes of retained messages and evicts oldest turns as needed.
  A single newest message larger than either byte ceiling is safely truncated
  with a `...[truncated]` marker, while the live response returned by the agent
  remains unchanged. Direct edits to the public `turns` list are reconciled on
  the next `record_turn()` or serialization call so aggregate accounting cannot
  stay stale. Live providers receive a recent-turn window as chat messages, not
  the full stored transcript. Checkpoint callers can additionally
  opt into `CheckpointLimits(max_depth=..., max_nodes=...,
  max_string_bytes=..., max_total_string_bytes=...)` on
  `Environment.snapshot()/restore()` and `create/save/load_snapshot()`.
  Loads preflight structure and JSON string-token size before UTF-8 decode and
  parsing; `max_bytes=` remains the separate exact serialized-file ceiling.
  Limit violations raise `ValueError` with the violated ceiling named in the
  message. These are resource ceilings for the existing monolithic JSON
  checkpoint format, not an OS memory limit or a streaming JSON parser.
- **Reproducible experiments**: `neva.utils.reproducibility` provides one
  run-manifest/seeding/offline-replay workflow. Call
  `prepare_reproducible_run(environment, seed=..., prompts=...)` before a run
  to seed Python, optional NumPy/PyTorch runtimes, Neva-owned Python-random
  schedulers (including nested Composite schedulers), and explicit custom
  `set_seed` hooks. Unknown custom `_rng` objects are never replaced.
  `PYTHONHASHSEED` is **not** mutated by default; opt in with
  `set_child_hash_seed=True` when child/future interpreters should inherit the
  derived hash seed. The current interpreter's hash randomization cannot be
  changed after startup.
  Run manifests record exact caller prompts, behavior-affecting environment
  configuration/state, scheduler state/order, provider/model/endpoint settings,
  generation settings, full prompt-validator policy, initial conversation and
  supported memory state (including FAISS index state), agent attributes,
  tool argument schemas and guard policies, cache policy/state fingerprints,
  dependency/runtime versions, seed application, and audit metadata. Unsupported
  custom memories must expose `checkpoint_state()` or
  `reproducibility_config()`; manifest creation fails rather than silently
  claiming complete state. API keys and cached prompt/response contents are not
  serialized.
  `RunManifest.fingerprint()` is the **replay-compatibility fingerprint**:
  it excludes audit-only dependency/runtime/seed-report/metadata fields so an
  otherwise compatible tape can replay after a Python patch, kernel, or unused
  dependency update. `audit_fingerprint()` includes those audit fields.
  `ReplayTape.for_manifest(manifest)` stores both the compatibility fingerprint
  and its payload, so a mismatch reports the differing configuration paths.
  Recording attaches through `tape.attach_recording(env.agents)`. GPT recording
  validates the effective provider-request identity—including provider/model,
  endpoint, raw-vs-history mode, and the history window actually used—not only
  the bare prompt. Each call reserves its tape position before backend work, so
  nested calls preserve invocation order and slow provider calls do not hold the
  tape lock or serialize other agents. Fresh agents can then use
  `tape.attach_replay(env.agents, manifest=...)`; replay validates exact
  request-identity order/content and refuses mismatched manifests, altered
  records, exhausted sequences, or unconsumed calls. Each record has both the
  request-identity digest and a whole-record digest covering response/failure
  data; edited responses therefore fail on load. Recorded failures replay as
  `RecordedReplayError` rather than reconstructing arbitrary provider
  exception classes.
  This does **not** make live providers deterministic: server-side revisions,
  routing, sampling/runtime implementations, hidden service state, and
  provider-side changes remain outside Neva's control. NumPy/PyTorch seeding
  alone also does not guarantee deterministic hardware kernels. Wall-clock
  timestamps, generated UUIDs, and telemetry timing are runtime metadata and
  are not rewritten by replay. Manifests may contain caller prompts,
  conversation/memory contents, tool descriptions and metadata; replay tapes
  contain effective request identities plus model responses/failures. Treat both
  artifact types as potentially sensitive.
- **Long-Term Memory Integrations**: Plug in semantic vector stores like FAISS
  to give agents durable recall of historical conversations and research notes.
- **Input hygiene, not a security boundary**: Prompts are length-capped and
  stripped of control characters, with a small regex denylist (`<script`,
  `DROP TABLE`). Built-in `GPTAgent` providers automatically share one
  provider/account FIFO admission scope inside a process (60 requests/minute
  and 8 concurrent calls by default; both configurable). Set
  `provider_coordination_path=` or `NEVA_PROVIDER_COORDINATION_DB` to share
  the same rate, concurrency, and optional `provider_spend_limit` across
  processes through SQLite. Explicit legacy `RateLimiter` instances remain
  supported; they are FIFO and cooperatively cancellable. Async agent
  cancellation automatically reaches Neva-owned admission waits and retry
  backoff, though an already-running synchronous provider call is governed by
  its request timeout and cannot be forcibly killed by Python.
  `CircuitBreaker` fails fast after consecutive retryable failures. Default
  `CostTracker` prices cover gpt-4o-mini, grok-4.5, and the other built-in
  models; override them for billing. `SpendBudget` supports atomic
  reserve/settle/release semantics so concurrent calls cannot oversubscribe a
  local ceiling, while `provider_spend_limit=` applies the same reservation
  model to the shared provider/account scope. An optional
  `billing_reconciler=` can reconcile completed estimates with an
  authoritative account-spend total. Unpriced models still fail closed when
  spend enforcement is enabled. Failed turns can be persisted durably with
  `Environment(failure_log=FailureLog(path))` (`neva.utils.failures`): handled
  failures (both policies, plus scheduler-selection failures) are appended as
  JSON lines. Optional `RecoveryPolicy(max_retries=..., backoff=...)`
  adds bounded automatic retries for scheduler selection and selected agent
  turns; final escalation can inherit or force `raise`/`return`, while
  deliberate cancellation is never retried. `Environment.recovery_state()`
  exposes retry/recovery/escalation counters and the last recovery event;
  `retries_exhausted` counts only configured retry sequences that actually
  consume their final attempt, so the default `max_retries=0` path leaves it
  at zero.
  Long-running jobs can opt into retention with
  `FailureLog(path, rotate_bytes=..., backup_count=...)`; append/load/rotation
  are coordinated across processes sharing the same path when the sibling lock
  file is writable. Read-only logs remain loadable: if that lock cannot be
  opened because the storage is read-only, `load()` falls back to an unlocked
  snapshot read rather than requiring write access. A reader must use a
  matching rotation configuration (and sufficient `backup_count`) to load
  retained backups oldest-first. `max_record_bytes=` independently enforces a
  strict UTF-8 record ceiling by dropping raw context first and then truncating
  the diagnostic message with an explicit marker. `replay_failure(record)`
  re-dispatches a recorded turn; raw context is stored only with
  `include_context=True`. Automatic retries cover context construction and
  agent execution only; after `agent.step()` succeeds, completion-hook failures
  are escalated without rerunning the completed agent call. Agent execution is
  still at-least-once when retries are enabled, so retry-safe failures or
  idempotent external side effects are recommended. Recovery policy/state are
  runtime-only and remain configured on the receiving environment across
  checkpoint restore rather than being serialized. `replay_failure(record)`
  uses the same configured recovery policy as a normal selected turn. A
  successfully recovered turn's recorded scheduler latency covers the complete
  turn attempt window, including retry backoff.
  Telemetry omits raw prompts and
  completions unless `include_content=True`. This is not protection against
  prompt injection, unauthorized tool use, or account-wide overspend. Tools
  run with the process's network identity.
- **Tool guardrails**: `ToolGuard` enforces code-level tool policy independent
  of prompt content: allowlists, approval hooks, argument schemas, per-tool
  concurrency quotas, and execution ceilings. Agent-level guards are passed
  with `tool_guard=`; a tool can also carry a guard through
  `tool.set_tool_guard(...)`, which applies even to direct `Tool.use()`
  calls. Direct calls enforce schemas/guards and normalize returned values to
  `str`; normal subclass `super().use(...)` delegation remains supported.
  `AIAgent.call_tool()` composes agent and tool guards and uses the stricter
  timeout/output/memory ceilings. `ToolLimits(max_concurrency=N)` bounds
  simultaneous calls per tool object; stale slot state is weak-reference
  pruned, and a configured `timeout` also bounds time spent waiting for a
  concurrency slot. The default timeout mode remains thread-based for
  compatibility and cannot forcibly stop arbitrary synchronous code; a timed-
  out worker retains its slot until it exits. Opt into
  `isolate_process=True` for a hard process timeout. Isolation uses
  `forkserver` where available, otherwise `spawn`, so isolated tools and
  configured callables must be picklable. Child-side output truncation occurs
  before IPC; non-`Exception` child failures are converted to
  `ToolExecutionError`; and `max_memory_bytes=` can enforce
  `resource.RLIMIT_AS` where supported. Platforms that expose but reject
  `RLIMIT_AS` surface `ToolResourceLimitError`. Process isolation is not a
  complete sandbox: child tools still inherit the application's filesystem and
  network credentials unless a stronger external sandbox is provided.
- **Model-driven tool loop**: `agent.run_tool_loop(task)` (or
  `neva.tools.run_tool_loop(agent, task)`) adds a bounded JSON-action loop on
  top of those guardrails. The model must emit exactly one
  `{"action":"tool",...}` or `{"action":"final",...}` object per turn.
  Tool selections always re-enter `AIAgent.call_tool()`, so allowlists,
  approvals, argument schemas, concurrency/resource ceilings, and observer
  accounting remain the enforcement path. Malformed actions, unknown tools,
  schema failures, and permission denials are returned to the model as bounded
  feedback so it can correct itself on a later step. `ToolLoopConfig` bounds
  model turns/tool calls, advertised tools, retained model output, each feedback
  record, and orchestration prompt size. Tool advertisement is budget-aware:
  rich schema metadata is used when it fits, then compact descriptions, then
  name-only entries, without silently dropping registered tool names. On the
  default agent model path, `max_prompt_chars` may not exceed the agent's own
  prompt-validator ceiling; incompatible limits fail before the model call.
  Hitting `max_steps` returns a
  non-success `ToolLoopResult` instead of looping indefinitely. When no
  explicit model callable is supplied, the loop uses the agent's
  `generate_model_output()` hook: this sends the already composed bounded loop
  prompt without re-prepending agent memory/tool summaries. TransformerAgent and
  GPTAgent implement that raw path while retaining their normal
  `respond()` behavior; GPTAgent still applies provider admission, retries,
  spend/circuit controls, cache/telemetry, and provider context checks, but does
  not add prior conversation history to the loop prompt. Tool results are
  explicitly labelled untrusted data, but prompt wording is not a security
  boundary: hostile tool output may still influence a model, while code-level
  tool guards remain authoritative. The generic loop consumes text JSON rather
  than provider-native function-calling events; streaming/native tool APIs are
  separate concerns.
- **Concurrency**: Observer APIs and `LLMCache` are lock-protected. Agent,
  environment, and memory objects are not generally thread-safe. Built-in
  provider calls share account-scoped rate/concurrency admission in-process;
  processes coordinate when they share the configured SQLite coordination DB.
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
