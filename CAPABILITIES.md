# Neva Library Capabilities

This document describes durable capabilities available in Neva today. It is a
reference for users and contributors, not a roadmap, implementation tracker, or
release history. For chronological changes, see [CHANGELOG.md](CHANGELOG.md).

## Agent and environment runtime

- Build simulations from configurable agents, environments, schedulers, tools,
  memory backends, and observers.
- Advance environments one scheduled turn at a time with completed, failed, and
  scheduled turn accounting.
- Use environment-wide or per-agent failure handling with `raise` or
  `return` behavior and optional fallback values.
- Keep transcript updates synchronized with successful turn completion through
  the environment completion hook.
- Use recent conversation history when preparing built-in provider requests
  without requiring the full stored transcript to be sent on every call.

## Provider access

`GPTAgent` supports built-in provider paths for OpenAI-compatible APIs,
Anthropic, Google Gemini, and xAI Grok.

Provider-facing behavior includes:

- provider/model-aware cache identity;
- selective retry classification for retryable transport/provider failures;
- configurable request timeouts and retry behavior;
- provider-reported token usage when available, with local estimates as a
  fallback;
- circuit breaking after consecutive retryable failures, including guarded
  half-open recovery probes;
- configurable maximum response size;
- static built-in cost estimates that callers can override.

Unknown model pricing is not treated as free usage.

## Shared provider coordination and spend control

Built-in `GPTAgent` instances can coordinate access to a provider/account scope
instead of treating every agent as an independent client.

Capabilities include:

- FIFO request-rate admission;
- shared concurrency limits across agents, threads, and event loops in one
  process;
- optional cross-process coordination through a shared SQLite database;
- atomic spend reservation, settlement, and release;
- shared provider/account spend ceilings;
- optional reconciliation against an authoritative completed-spend total;
- cooperative cancellation while waiting for admission or retry backoff.

An already-running synchronous third-party network or SDK call cannot be
forcibly terminated by Python; it remains bounded by the configured provider
request timeout.

## Context and output budgets

Neva provides separate controls for stored history, formatted request size,
model-aware token envelopes, and generated output.

### Conversation retention

`ConversationState` can bound:

- retained turns;
- UTF-8 bytes per stored message;
- aggregate UTF-8 bytes across retained messages.

The limits affect stored history, not the live response returned to the caller.

### Character-based request limits

Built-in provider requests retain the existing formatted-character ceiling so
oversized prepared prompts can fail before a provider request is made.

### Model-aware context budgets

`ModelContextBudget` adds an opt-in provider/model-specific token envelope.
It can:

- count the effective request using a caller-supplied model-aware counter;
- reserve configured output tokens inside the same context ceiling;
- select the largest recent contiguous history suffix that fits;
- reject a current prepared request that cannot fit instead of silently
  truncating it;
- apply the same check to normal, streaming, and raw model/tool requests.

The included OpenAI helper uses `tiktoken` for supported models. Local token
counting is a preflight estimate and is not an authoritative provider context
or billing guarantee.

## Streaming responses

Built-in provider agents support bounded streaming through
`GPTAgent.stream_response()`.

A stream:

- can be consumed synchronously or asynchronously;
- emits text deltas followed by a terminal completion event;
- applies bounded producer/consumer backpressure;
- supports cooperative close/cancellation;
- records first-token and full-completion latency separately;
- commits completed output to history/cache only when the terminal completion is
  accepted by the consumer;
- raises `StreamInterruptedError` with uncommitted partial text when output has
  begun and the stream then fails.

Pre-output retryable failures may use configured retries. Once partial output
has been emitted, Neva does not replay the call automatically.

## Durable failure recovery

`RecoveryPolicy` provides opt-in bounded recovery for scheduler selection and
agent execution.

It supports:

- bounded retries;
- exponential backoff;
- exception filtering;
- configurable final escalation;
- recovery counters exposed by `Environment.recovery_state()`;
- explicit replay of recorded failures.

`FailureLog` can persist handled failures as JSON lines with:

- optional raw context;
- record-size ceilings;
- bounded rotation and backup retention;
- multi-process append/read/rotation coordination when a writable sibling lock
  is available;
- tolerant reads of a torn final record.

Automatic retry stops before completion hooks so a hook failure does not repeat
an already-successful model/backend call. Callers should only retry failures
that are safe to execute more than once or make external side effects
idempotent.

## Checkpointing and persistence

Neva can snapshot and restore supported environment, scheduler, agent, memory,
and conversation state.

Persistence controls include:

- versioned checkpoints;
- atomic streamed saves through a sibling temporary file and `os.replace`;
- exact serialized checkpoint byte ceilings;
- `CheckpointLimits` for nesting depth, node count, individual string/key
  bytes, and aggregate string/key bytes;
- lexical preflight of limited JSON loads before full parsing;
- bounded transcript retention independent of checkpoint size;
- first-party memory adapters that preserve configured callable/budget identity
  where appropriate.

Custom schedulers and memory implementations can participate through explicit
checkpoint hooks.

The checkpoint format remains a single JSON document. The graph and file limits
bound Neva's serialization path; they are not an operating-system RSS limit.

## Memory

Neva supports pluggable memory components, including FAISS-backed vector memory
through the optional `memory` extra.

Supported first-party memory state can participate in checkpointing and
reproducibility workflows. Custom memory implementations that need those
features should expose the corresponding checkpoint/reproducibility hooks.

## Tool schemas, permissions, and execution limits

Tools can enforce policy independently of prompt wording through `ToolGuard`
and `ToolLimits`.

Available controls include:

- argument-schema validation;
- allow/deny policy;
- approval hooks;
- per-tool concurrency limits;
- execution timeouts;
- output-size ceilings;
- optional memory ceilings where supported;
- tool-level guards that also apply to direct `Tool.use()` calls;
- composition of agent-level and tool-level guards, with stricter resource
  ceilings winning.

Thread-based timeout mode is cooperative and cannot forcibly stop arbitrary
synchronous code. For hard termination, tools can opt into process isolation.
Process isolation requires picklable state and is not a complete security
sandbox: child processes still inherit application-level filesystem and network
access unless the application supplies a stronger external sandbox.

## Model-driven tool orchestration

`agent.run_tool_loop(task)` and `neva.tools.run_tool_loop(...)` provide a
bounded model/tool/feedback loop.

The loop:

- expects one structured tool or final action per model turn;
- routes tool execution through the normal guarded `AIAgent.call_tool()`
  path;
- returns bounded feedback for malformed actions, unknown tools, schema errors,
  and permission denials;
- caps model turns, tool calls, prompt size, tool advertisement size, retained
  model output, and feedback size;
- returns a non-success result when the configured step bound is reached.

Tool output is treated as untrusted model input. Prompt instructions are not a
security boundary; code-level tool permissions and resource controls remain the
authoritative enforcement path.

## Reproducible experiments and offline replay

`neva.utils.reproducibility` provides a unified workflow for experiment
manifests, seeding, recording, and replay.

It can capture behavior-affecting configuration such as:

- prompts and environment state;
- scheduler configuration/state;
- provider/model/endpoint and generation settings;
- prompt validation and tool policy;
- supported memory state;
- cache policy fingerprints;
- relevant runtime/dependency metadata;
- seed application.

Replay tapes validate request identity and recorded content before returning a
stored response or recorded failure. They are intended for deterministic
offline replay of captured model-boundary behavior.

This does not make live providers deterministic. Provider revisions, hidden
service state, sampling implementations, and hardware/runtime behavior remain
outside Neva's control.

## Observability and privacy defaults

Neva exposes simulation metrics and optional OpenTelemetry instrumentation for
conversation turns, provider calls, and tool execution.

Telemetry omits raw prompt/completion content by default. Applications may opt
in to content capture explicitly, and should treat telemetry, manifests,
failure logs, checkpoints, and replay artifacts as potentially sensitive when
they contain caller-provided data.

## Scheduling

Neva includes a scheduler registry and built-in strategies such as:

- round robin;
- random;
- priority;
- least recently used;
- weighted random;
- event driven;
- conditional;
- composite scheduling.

Custom schedulers can be registered at runtime.

## Scope and security boundaries

Neva provides library-level controls, not a universal sandbox or account-wide
policy system.

In particular:

- prompt validation is input hygiene, not prompt-injection protection;
- thread-based timeouts cannot forcibly terminate arbitrary synchronous code;
- process-isolated tools still need external sandboxing for strong filesystem or
  network isolation;
- local/context token counters and built-in prices are estimates;
- cross-machine coordination requires an external shared coordination service;
- most agent, environment, and memory objects are not generally thread-safe
  unless documented otherwise.

These boundaries are part of the supported behavior and should be considered
when embedding Neva in a larger application.
