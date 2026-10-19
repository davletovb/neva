# R3: provider reliability, spending boundaries and guarded tools

Issue: [#103](https://github.com/davletovb/hiyal/issues/103).
Status: native guard/HTTP implementation; conditional integrations remain
deferred. This milestone **does not claim exactly-once remote effects**.

## Model boundary

- OpenAI-compatible model requests are opt-in through `live-http`. The
  `OpenAiCompatible` adapter accepts only HTTPS or parsed loopback HTTP,
  rejects URL user-info, and never follows redirects.
- `HttpReliability` bounds HTTP response bytes (1 KiB to 8 MiB; 1 MiB
  default), handles streamed `data:` SSE events with backpressure by
  awaiting each chunk, enforces `[DONE]`, and rejects broken or oversized
  frames. JSON completions are collected incrementally under the same cap.
- Streaming returns a concatenated assistant response and provider-reported
  model identity and usage when present. Requests and durable receipts store
  sampling parameters, provider/model identity and raw token counts.
  **Provider cost is unknown** when usage lacks a verified bill.
- Retry policy defaults to zero. If explicitly configured, only **HTTP 429**
  status may be retried, at most twice after a bounded delay. No transport,
  timeout, 5xx, or partial stream is automatically retried because provider
  side effects and billing may already have occurred. Failure opens the
  in-memory circuit after configurable consecutive failures; a locally
  rejected circuit-open request is `not_charged`.
- Effect-journal timeout drops a pending HTTP request future and records
  the failure receipt with `billing=unknown` after any possible dispatch.
  Dropping a future does not revoke a provider's already submitted work.
  Persisted keyed receipts remain the source for strict offline replay.

## Process-local spend and tool guards

`GovernedBackend` wraps any `Backend`. `BudgetConfig` requires explicit
per-input-token and output-token micro-USD rates, a maximum input size,
maximum output tokens, and total micro-USD reservation. The conservative
input byte count and max output token bound are used to reserve capacity
atomically **before the provider call**, including concurrent calls. There
are no refunds on transport uncertainty. This is a local *admission budget*
based on caller-declared rates, **not provider accounting** and not a
cryptographically guaranteed spend ceiling. Providers may add invisible
wrapper tokens, change prices, or produce delayed charges.

CLI `--live` requires `OPENAI_COMPAT_URL`, `OPENAI_COMPAT_API_KEY`, plus:

- `HIYAL_LIVE_MAX_MICRO_USD` — positive total process reservation
- `HIYAL_LIVE_INPUT_MICRO_USD_PER_TOKEN` — declared input rate
- `HIYAL_LIVE_OUTPUT_MICRO_USD_PER_TOKEN` — declared output rate

All three prices/caps must be positive; missing values fail closed. A new
process creates a new admission budget. No SQLite/distributed shared cap
is claimed; that requires a real multiprocess research workload.

Tools are denied by default. The journal preflights a maximum **32 total
effects and 8 tool calls per logical attempt** before starting any provider
request. Only a backend explicitly implementing `permits_tools` may receive
tool specs. The `GovernedBackend` allowlist enforces tool names, bounded JSON
inputs, and a `ToolExecutor::validate` validator that rejects by default.
For an action with external side effects, `approve_call` binds human approval
to the **full trial, attempt, actor, call index and request hash**. The
approval is consumed at most once within the process. A tool executor is an
explicit Rust implementation, never an arbitrary shell command. Tool output
is durably recorded as an effect receipt before the world is advanced, but
remote side effects remain **billing/commit uncertain** when a process dies
before receipt fsync. Execution must be idempotent or reconciled externally.
Blocking/non-cooperative tool executors are not cancellation safe and should
instead run in supervised isolated workers in a later implementation.

## Conditional R3 decisions

1. **Additional providers:** No Anthropic, Gemini, or other provider API
   is required by the current built-in counter/trial workflows. Their
   capabilities differ; no dishonest “OpenAI-compatible” fallback is added.
   Their unsupported features return explicit errors.
2. **External-process JSON policies:** No cross-language authored policy
   consumer has been selected. Hiyal's API intentionally stays Rust-only,
   and this integration is deferred. A proposed **v1 wire contract** is
   `u32 big-endian length | JSON object`, maximum 256 KiB, with
   `protocol_version`, `trial_id`, `invocation_id`, `agent_id`,
   `logical_step`, `observation`, and response `action` or typed
   `error`. A future implementation must launch fixed executable argv
   (never a shell string), sandbox its privileges, bind responses to exact
   invocation IDs, supervise stdout/stderr, apply timeouts and bounded
   queues, kill the entire process group on cancellation, and test an
   independently built non-Rust policy consumer. Merely specifying this
   contract does not constitute an implemented protocol.
3. **Cross-process SQLite coordination:** Deliberately not added without
   a multiprocess billable workload; `GovernedBackend` limits are local.

## Verification

Unit and captured loopback HTTP tests cover streaming fragmentation,
usage/model capture, permission-denied tools, keyed approval consumption,
budget reservations, and journal failures/replay semantics. Real provider
billing and optional cross-language process operation **are not verified**
by local mocks. No Python benchmarks, legacy CI, or new CI matrix.

- Loopback bearer requests bypass all environment proxy settings with a direct HTTP client, preventing token leakage through `HTTP_PROXY` when `NO_PROXY` is unreliable.
- Permission checks apply only to *pending* tool executions. A strict replay of an already hashed, durable tool receipt can use an inert `ScriptedBackend` without approval or a live executor. A captured keyed-tool replay regression verifies this distinction.

- The admission reservation accounts for **every serialized model request field**, including tool-call IDs, messages, schemas, sampling and wrapper overhead. Only built-in backends that guarantee an `UnsupportedCapability` result occurs **before any dispatch** may return that reservation to the per-process budget; transport and post-dispatch failures never refund.
- Intermediate retryable HTTP 429 statuses do not trip the circuit prematurely. A final/exhausted 429 counts as a failure. The regression tests the boundary with a threshold of one and a single allowed retry.
