# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Durable automatic recovery: `RecoveryPolicy` adds opt-in bounded retries,
  exponential backoff, retry exception filtering, and final escalation for
  scheduler-selection and selected agent-turn failures. Defaults preserve the
  historical single-attempt behavior. `Environment.recovery_state()` exposes
  recovery counters and the last recovery event; deliberate cancellation is
  never retried.
- Failure records now persist attempt/max-attempt/action/truncation metadata.
  `FailureLog(max_record_bytes=...)` adds an optional hard UTF-8 per-record
  ceiling that preserves structural recovery metadata, drops raw context first,
  and then truncates the diagnostic message with an explicit marker.
  Append/load/rotation are serialized across processes by a sibling advisory
  lock file, closing the previous multi-process rotation race.
- Circuit-breaker tests now cover concurrent half-open contention and release of
  interrupted/rejected recovery probes.
- Checkpoints now natively preserve `VectorStoreMemory` and `AdaptiveConversationMemory` state without custom hooks. Vector records/cached embeddings and adaptive history, summary/short-term views, token counts, and memory-budget embedding usage are restored data-only; incompatible structural configurations fail closed and configured embedder/summarizer callables plus `MemoryBudget` instances are preserved by identity rather than deep-copied. Callables are not serialized or replayed; callers remain responsible for supplying semantically equivalent summarizer/embedder/token-estimator configuration when restoring a checkpoint.
- `ConversationState(max_turn_bytes=N)` adds an opt-in UTF-8 byte ceiling for each stored turn. Oversized stored messages are truncated safely with a `...[truncated]` marker, the setting survives serialization/restore, and the default remains unlimited; agent calls still return the full live response.
- A repository-local checkpoint scaling benchmark (`python -m benchmarks.checkpoint_scaling`) measures deterministic small/medium/large workloads across snapshot creation, streamed save, and load. It reports checkpoint bytes, untraced wall-clock time, and peak Python allocations from a separate `tracemalloc` pass with raw samples and medians; no hardware-dependent pass/fail threshold is imposed. The CLI can target a real checkpoint filesystem with `--workdir` and record a local revision with `--git-sha`.
- Shared provider/account resource coordination for built-in `GPTAgent`
  backends: same-account agents automatically share FIFO request-rate and
  concurrency admission in-process (60 requests/minute and 8 concurrent calls
  by default). A shared SQLite `provider_coordination_path` or
  `NEVA_PROVIDER_COORDINATION_DB` extends the same rate, concurrency, and
  optional `provider_spend_limit` across processes, with expiring leases for
  crash recovery and fail-closed configuration mismatches.
- `SpendBudget` now supports atomic reserve/settle/release semantics. GPT
  calls reserve worst-case estimated input plus configured output-token spend
  before provider admission and settle against provider-reported/estimated
  actual usage, preventing concurrent pre-check oversubscription. Optional
  `billing_reconciler=` hooks can replace completed estimates with an
  authoritative account-spend total while preserving in-flight reservations.
  Unpriced or non-finite model pricing still fails closed when spend
  enforcement is active.
- Tool-call guardrails: `AIAgent(tool_guard=ToolGuard(...))` (forwarded by
  `GPTAgent` and `TransformerAgent`) enforces code-level policy independent of
  prompt content — an allowlist, an approval hook called with each `ToolCall`,
  and execution limits (`ToolLimits(timeout=..., max_output_chars=...)`,
  where `timeout` may not exceed `threading.TIMEOUT_MAX`). The
  call proceeds only when the approval hook returns `True`; `False`, a truthy
  non-bool, an awaitable, or a raised exception denies, and denial reasons
  exclude hook exception details (logged at debug). Denied calls return a
  failed `ToolResponse` without running the tool. Timeouts run the tool on a
  daemon worker thread that is not forcibly stopped (a tool that never
  returns leaks that one thread, and nothing is re-joined at interpreter
  exit) and raise `ToolTimeoutError` (a `ToolExecutionError`); outputs longer
  than `max_output_chars` keep that many characters plus a truncation marker.
  Guardrails apply to `call_tool`; direct `Tool.use` calls bypass them.
- Validated tool argument schemas: tools may declare
  `argument_schema=ArgumentSchema({...})` with per-field `ArgumentSpec` rules
  (type, required, min/max length, min/max value, choices; unknown keys are
  rejected unless `allow_extra=True`). `call_tool` validates mapping
  arguments — a raw-string payload is validated as `{"input": payload}` —
  before executing the tool; violations and schema failures return a failed
  `ToolResponse` carrying a reason and never reach the tool body. Invalid
  schema configuration raises `ToolSchemaConfigurationError` at construction;
  non-finite numbers are rejected whenever value bounds are configured. The
  built-in calculator, Wikipedia, summarizer, and translator tools now declare
  schemas that share the normalizer's alias constant and preserve
  `input`/`task`/`query`/`text`, metadata-bearing calls, and single-string
  mappings. Mapping shapes that previously fell through to JSON serialization
  and reached the tool as JSON text are now rejected before execution.
  Tools without a schema behave as before.
- Durable failure records: `Environment(failure_log=FailureLog(path))` appends
  one JSON line per handled turn failure (both `raise` and `return` policies,
  including scheduler-selection failures), flushed and fsynced by default.
  `FailureLog.load()` tolerates truncated, undecodable, or malformed lines
  (skipped with a warning) and a torn tail is newline-separated so later
  records cannot merge into it. Optional `rotate_bytes` + `backup_count`
  retain a bounded number of rotated JSONL files for long-running jobs.
  Readers must opt into rotation with a sufficient `backup_count` to load
  retained generations oldest-first; a default reader loads only the active
  file. Retention pruning is performed once per configured log instance before
  its first append, while rotations maintain the bound thereafter. Append,
  load, and rotation are coordinated across processes sharing the same path.
  `rotate_bytes` remains a file-rotation threshold; the independent
  `max_record_bytes` option supplies a strict per-record ceiling. In addition,
  `Environment.replay_failure(record)` re-dispatches a recorded turn through
  the normal turn path. Raw context is stored only when the log opts in with
  `include_context=True`. The log is external storage: it is preserved across
  checkpoint restore rather than serialized into snapshots.
- `RateLimiter` now provides FIFO token admission, sleeps outside its mutex,
  and polls normal mutex acquisition for cooperative cancellation. Base async
  agent calls automatically propagate asyncio cancellation through a
  threading Event into GPT provider admission and retry backoff; synchronous
  third-party provider calls themselves remain governed by their request
  timeout and cannot be forcibly killed by Python.
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
- Supported/tested Python versions are now 3.11, 3.12, 3.13, and 3.14; Python
  3.9 and 3.10 were removed from the CI matrix and package classifiers.
- Snapshot saves now serialize JSON incrementally into a sibling temporary file instead of materialising the complete JSON string and UTF-8 byte string in memory. Optional `max_bytes` limits are enforced while encoding; the staged file is flushed/fsynced and installed with atomic `os.replace`, so overflow, serialization errors, staging-write failures, and replacement failures preserve an existing checkpoint. Large saves therefore require temporary disk space on the destination filesystem roughly equal to the new checkpoint.
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
- README describes input hygiene, shared provider/account rate/concurrency/spend
  controls, their SQLite cross-process option, and remaining transport/thread-
  safety boundaries instead of calling them "robust safety rails".
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
