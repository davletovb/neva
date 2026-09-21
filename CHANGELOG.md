# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `SpendBudget` enforces a thread-safe hard ceiling on estimated spend.
  `GPTAgent(spend_budget=...)` refuses models without a pricing entry before
  contacting the provider, consumes each call's estimated cost after token
  accounting (clamping recorded spend to the ceiling), and rejects non-finite
  pricing or costs instead of silently disabling the guard; exceeding the
  ceiling raises `SpendBudgetExceededError`. Share one instance across agents
  for a common budget — amounts are estimates, not live billing, and there is
  no account- or process-wide coordination.
- Durable failure records: `Environment(failure_log=FailureLog(path))` appends
  one JSON line per handled turn failure (both `raise` and `return` policies,
  including scheduler-selection failures), flushed and fsynced by default.
  `FailureLog.load()` tolerates truncated, undecodable, or malformed lines
  (skipped with a warning) and a torn tail is newline-separated so later
  records cannot merge into it, and
  `Environment.replay_failure(record)` re-dispatches a recorded turn through
  the normal turn path. Raw context is stored only when the log opts in with
  `include_context=True`. The log is external storage: it is preserved across
  checkpoint restore rather than serialized into snapshots.
- `RateLimiter.acquire(cancel_event=...)` supports cooperative cancellation of
  token waits using a threading Event; cancellation raises
  `RateLimiterCancelledError` (a `concurrent.futures.CancelledError` subclass).
  Lock waits and provider calls are not interrupted, and callers must pass the
  event explicitly.
- Packaging metadata for distribution: project URLs, keywords, and trove
  classifiers in `pyproject.toml`.
- PEP 561 `py.typed` marker so downstream projects consume Neva's type hints.
- `memory` extra bundling the optional `faiss-cpu` and `numpy` dependencies used
  by the FAISS vector store.
- Repository `.gitignore` covering build, coverage, cache, and runtime
  artifacts.
- Unit tests for `GPTAgent` Chat Completions providers, including Grok/xAI.
- Optional per-agent failure policies: `Environment.register_agent(agent,
  error_policy="raise"|"return", error_value=...)` overrides the
  environment-wide policy for that agent's failed turns (including context and
  transcript-hook failures). Agents without an override inherit the environment
  defaults, and scheduler-selection failures keep using the environment policy.
  Overrides survive version-2 checkpoints and older checkpoints clear them.
- `CircuitBreaker` fails fast after consecutive retryable provider failures
  (each retryable HTTP attempt counts) and allows a single probe after a
  cooldown. A rejected probe releases the in-flight slot so later calls can
  try again. Pass the same instance to share a provider circuit across agents.

### Changed
- OpenAI-compatible providers (OpenAI and Grok/xAI) now call `/v1/chat/completions`
  over `requests` instead of the legacy `openai==0.28.1` SDK.
- Default models follow the selected provider (`gpt-4o-mini`, `grok-4.5`, …).
- The `openai` package is an optional `providers` extra rather than a hard
  dependency.
- `GPTAgent` now sends a recent-turn window as chat messages (flattened for
  Gemini) so live providers see prior dialogue. The window is bounded by
  `max_context_chars` (default 24,000) measured on the serialized provider
  request; a current prompt that cannot fit raises `ConfigurationError`.
  `ConversationState` itself is unchanged.
- README describes input hygiene, per-instance rate limits, and thread-safety
  boundaries instead of calling them "robust safety rails".
- `CostTracker` default prices cover `gpt-4o-mini`, `grok-4.5`,
  `claude-3-5-sonnet-latest`, and `gemini-1.5-flash` (USD per 1k tokens,
  list prices as of 2026-09). Override `pricing_per_1k_tokens` for billing.
- Telemetry omits raw prompts, completions, tool payloads, and reasoning
  text by default (length + SHA-256 fingerprints instead). Pass
  `include_content=True` to opt in to exporting conversation content.

### Fixed
- Corrected a malformed `RUN` instruction in the `Dockerfile` that contained a
  stray line continuation.
- Updated the README "Last Commit" badge to point at the `main` branch.
- Quickstart example adds the repository root to `sys.path` so
  `python examples/quickstart_conversation.py` works from a fresh clone.
- Grok/xAI provider now uses the OpenAI-compatible Chat Completions endpoint
  and sends `max_tokens`.
- FAISS memory module type annotations no longer break `mypy` when NumPy is
  absent.
- Wikipedia missing-dependency test actually runs.
- `batch_communicate(concurrent=True)` uses `asyncio.run` and refuses to nest
  inside an active event loop.
- `remove_from_group` raises `AgentNotFoundError` instead of `ValueError`.
- Prompt validator no longer rejects ordinary uses of the word "shutdown".
- Tavern NPC demo no longer nests the full transcript into every line.
- Anthropic content-block parsing now reads SDK objects as well as dicts.
- OpenTelemetry log SDK import prefers the public `opentelemetry.sdk.logs`
  package and falls back to `_logs`.
- Custom OpenAI/Grok `api_base` values that are not already a Chat Completions
  endpoint now have `/chat/completions` appended.
- `RateLimiter.acquire()` sleeps outside its lock so shared limiters are not
  serialized for the full wait.
- `error_policy="return"` applies to `SchedulingError` from `get_next_agent()`.
- `MathTool` rejects exponentiation outside a bounded range.
- Observer timestamps no longer use deprecated `datetime.utcnow()`.
- Token accounting uses the text actually sent (including history). Gemini
  `usage_metadata` is honoured when the SDK provides it.

## [0.1.0] - 2024-05-01
### Added
- Initial project restructuring introducing the `neva` package with dedicated
  modules for agents, environments, schedulers, tools, memory, and utilities.
- Agent abstractions including a stub-friendly `TransformerAgent` and a
  multi-provider `GPTAgent` (OpenAI, Anthropic, Gemini, Grok).
- Pluggable scheduler registry with round-robin, random, priority,
  least-recently-used, weighted-random, event-driven, conditional, and composite
  strategies.
- Memory integrations spanning short-term, summary, composite, adaptive, and
  budget stores plus an optional FAISS-backed vector store.
- Built-in tools for arithmetic, summarisation, translation, and encyclopedia
  lookups with graceful fallbacks when optional dependencies are absent.
- Observability utilities: structured logging, a metrics-collecting
  `SimulationObserver`, and a vendor-neutral OpenTelemetry instrumentation layer.
- Safety rails (prompt validation/sanitisation, rate limiting, retries),
  snapshot/restore state management, examples, Sphinx documentation, and a CI
  pipeline enforcing formatting, linting, typing, security, and 80% test
  coverage.
