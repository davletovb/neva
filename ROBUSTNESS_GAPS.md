# Neva: Robustness and Product Gaps

## Verified baseline and scope

Updated against `main` at `715cd57` (through PR #77 merged).

- Checkpoint file-size limits are merged: opt-in positive UTF-8 byte counts; limited loads read in 64 KiB chunks (total bounded at limit + 1) and reject overflow before decoding or parsing; oversized saves leave existing files untouched. PR #64 additionally streams save serialization into a sibling temporary file instead of materialising the complete JSON string and byte string, fsyncs it, and atomically installs it with `os.replace` so existing checkpoints survive serialization, staging-write, and replacement failures.
- FAISS PR #53 remains open/deferred, but PR #73 intentionally supersedes its coverage-omit removal and missing-dependency-test changes while adding broader FAISS correctness/integration coverage. PR #53 should be rebased or retired after #73 lands.
- These checks do not establish production readiness. Items below include feature gaps, untested risks, and known scope limits—not all are confirmed bugs.

## Already implemented on merged main

### PRs #47–#48: correctness and basic recovery

- Versioned checkpoints for supported agent, scheduler, memory, environment, and observer state; unsupported components fail explicitly.
- Unknown model pricing returns `None`, rather than implying free usage.
- Provider/model-aware cache identity and selective HTTP retry classification.
- Scheduled/completed/failed turn accounting and observer synchronization.
- Shared manager concurrency semaphores within each event loop; repeated synchronous calls avoid stale-loop semaphore reuse.
- Environment-wide `error_policy="return"` and optional fallback value, exposed by `BasicEnvironment`.
- Transcript updates through `on_turn_complete()` before metrics. Bundled examples no longer have the dialogue-length off-by-one.
- Single-agent real local HTTP test for retry classification, headers, model selection, and caching.

Custom transcript environments must use the completion hook. Observer synchronization does not make every agent/environment/memory object thread-safe.

### PR #49: history, limiter, scheduler, arithmetic, and documentation

- Provider requests and cache identity include a recent history window.
- Gemini uses provider-reported usage metadata, with complete-request text for fallback estimates.
- Configured character budgets account for provider-specific formatted text; oversized prepared current prompts fail clearly. The previously reported 24-character/52-character mismatch was fixed in `217bd69` before merge.
- Rate-limiter sleeping happens outside its lock.
- Scheduler selection errors respect the return policy.
- MathTool exponentiation has operand bounds; this is not a general resource sandbox.
- README describes input hygiene and thread-safety limits instead of claiming comprehensive safety rails.

The PR #49 character limit alone is heuristic. Section 10 adds a separate
opt-in model-bound request token envelope and explicit output reservation.
ConversationState is unbounded by default; PR #55 adds an opt-in stored-turn
retention limit independent of the transmitted history window.

### PR #50: circuit breaker and default-model pricing

- `CircuitBreaker` in `neva.utils.safety` counts retryable failures, opens after a threshold, and allows a recovery probe after cooldown.
- `GPTAgent` integrates the breaker. Sharing one instance allows multiple agents to share that circuit; automatic account/process-wide coordination is not provided.
- Non-retryable half-open failures release the probe instead of permanently blocking recovery.
- Default-model price entries were added. These are static estimates, not verified live billing rates or a spend enforcement system.

### PR #51: telemetry privacy defaults

- Raw content is omitted by default from the covered telemetry fields; `include_content=True` explicitly opts into raw content.
- This is not a guarantee that all application logs, custom metadata, or caller-provided exporters are free of sensitive data.

## Merged PR #52: HTTP integration coverage

`tests/integration/test_multi_agent_http.py` adds:

1. Four round-robin turns between two GPTAgents using a loopback HTTP provider. Checks headers, endpoint/model selection, transmitted history, cross-agent observations, and completed-turn/transcript metrics.
2. An actual delayed HTTP response produces a Requests read timeout wrapped as `BackendError`, with one request when retries are disabled.
3. An environment returns its failure sentinel for a timed-out agent, then completes the healthy agent's turn, preserving failed/completed/transcript counts.

The fixture releases and joins server workers during cleanup. It uses no external providers or real credentials. These tests cover read timeouts, not connect/write timeout behavior or live SDK services.

Full-suite verification exposed an existing clock inconsistency in `test_rate_limiter_honours_rate`: construction used real monotonic time before the test switched to a fake zero-based clock. The local test change constructs the limiter after installing the fake clock and asserts the exact half-second wait. No production behavior was changed.

## Remaining gaps

### 1. Shared rate, concurrency, and spend budgets — complete at the Neva coordination layer

PR #69 completes the library-level coordination gap as one integrated provider/account resource system:

- Built-in `GPTAgent` instances derive a non-secret provider/account scope and automatically share FIFO request-rate and global concurrency admission across agents, threads, and event loops in one process. Conflicting limits for the same account scope fail closed rather than silently splitting the budget.
- Cross-process coordination uses the same scope through SQLite when `provider_coordination_path=` or `NEVA_PROVIDER_COORDINATION_DB` points participating processes at the same database. SQLite state covers FIFO tickets, token-bucket state, concurrency leases, spend reservations, and completed spend; expiring leases recover from crashed processes.
- `SpendBudget` now supports atomic reserve/settle/release semantics. Provider calls reserve worst-case estimated input + configured output-token spend before admission, preventing concurrent pre-check oversubscription, then settle against provider-reported or estimated actual usage.
- `provider_spend_limit=` applies the same reservation model at the shared provider/account scope. An optional `billing_reconciler` can replace completed estimates with an authoritative account-spend total while preserving in-flight reservations.
- `RateLimiter` is FIFO, sleeps outside its mutex, and polls mutex acquisition for cancellation. Base async agent execution automatically propagates asyncio cancellation through a threading Event, so provider admission queues and retry backoff cancel cooperatively without leaving FIFO waiters or reservations behind.
- Aggregate-contention, same-account sharing, conflicting-scope configuration, local and SQLite cancellation cleanup, spend reservations, authoritative reconciliation, real spawned-process SQLite concurrency, and cancel-during-in-flight settlement/cache continuity are covered by tests.

Boundary: Python cannot forcibly kill an already-running synchronous third-party HTTP/SDK call. Async callers are cancelled immediately, while that worker remains governed by its provider/request timeout and performs final accounting/release when it returns. Cross-machine/account-wide coordination would require an external shared coordination service; the built-in SQLite coordinator covers processes sharing the configured database.

Correction to the original review remains relevant: receivers do not consume the sender's limiter. Explicit legacy `rate_limiter=` is still supported and replaces only the shared rate component; shared provider concurrency can remain active.

### 2. Durable failure recovery — complete at the Neva environment layer

PR #70 completes the remaining library-level recovery gap:

- `RecoveryPolicy` adds opt-in bounded retries, exponential backoff, exception filtering, and final escalation. `max_retries=0` preserves historical one-attempt behavior. Scheduler-selection failures and selected agent turns use the same policy, while deliberate cancellation is never automatically retried.
- Final escalation may inherit existing environment/per-agent `raise`/`return` handling or force one of those dispositions after retries are exhausted. Intermediate retry failures do not inflate failed-turn metrics; a recovered turn is still recorded as completed, with latency covering the full attempt/backoff window. Explicit `replay_failure()` re-dispatch uses the same configured recovery policy.
- `Environment.recovery_state()` exposes consistent runtime counters for observed failures, retries attempted, successful recoveries, exhausted retry sequences, escalations, durable records written, and the last recovery event. `retries_exhausted` increments only when a configured retry sequence reaches its final allowed attempt; the default `max_retries=0` path does not count as exhaustion.
- Durable records now include attempt number, maximum attempts, recovery action, and truncation status while remaining backward-compatible with older JSONL records.
- `FailureLog(max_record_bytes=...)` adds an optional strict UTF-8 per-record ceiling. Oversized records preserve routing/recovery metadata, drop raw context first, then truncate the diagnostic message with an explicit marker; structurally impossible limits fail closed.
- Appends, reads, and rotation are coordinated across processes by a sibling advisory lock file, closing the multi-process rotation race while retaining fsync, torn-tail separation, tolerant loading, and bounded backup retention. Read-only log storage remains readable: when the sibling lock cannot be opened because the filesystem denies writes, `load()` falls back to an unlocked snapshot read rather than failing.
- Circuit-breaker tests now cover concurrent half-open contention (exactly one recovery probe) and interrupted/rejected probe release so later recovery attempts cannot be permanently stranded.

Boundary: automatic retries apply only to context construction and agent execution before a successful turn completes. Once `agent.step()` succeeds, `on_turn_complete()` is not retried, preventing a hook failure from repeating a completed model/backend call. Agent execution itself is still at-least-once when retries are enabled, so callers should scope `retry_on` to retry-safe failures or make external side effects idempotent. Recovery policy/state counters are runtime-only and intentionally remain configured on the receiving environment across checkpoint restore rather than being serialized.

### 3. Tool schemas, permissions, and execution limits — complete at the Neva tool layer

PR #71 closes the remaining library-level execution gap while preserving the public tool API:

- Direct `Tool.use()` routes through schema validation and optional per-tool guardrails. Approval hooks receive the same mapping-shaped `{"input": ...}` arguments as agent-mediated calls, and guarded execution normalizes direct results to `str`, matching the declared return type.
- Concrete tool implementations are preserved separately from the public wrapper. Inherited/mixin implementations are guarded, while normal subclass `super().use(...)` delegation continues to call the parent implementation without recursively re-entering the guard wrapper.
- `AIAgent.call_tool()` composes its agent guard with any tool-level guard. Permission checks run before execution; the stricter timeout/output/memory ceiling wins; the implementation runs once; observer usage/telemetry still records centrally.
- `ToolLimits(max_concurrency=N)` bounds simultaneous executions per tool object. Slot bookkeeping is weak-reference-backed so dead tool objects are pruned instead of leaking or reusing stale `id(tool)` entries. When a timeout is configured, time spent waiting for a concurrency slot consumes that same deadline and fails with `ToolTimeoutError` instead of hanging behind a stuck timed-out worker.
- The compatibility timeout mode remains thread-based and cannot forcibly stop a worker that ignores cancellation. Such a worker keeps its concurrency slot until it exits, but later callers with a timeout are still bounded by their own deadline. Output truncation remains active on this timeout path.
- `ToolLimits(timeout=..., isolate_process=True)` adds a hard process timeout. Isolation uses `forkserver` where available, otherwise `spawn`, so isolated tools/configured callables must be picklable. Large results are drained concurrently to avoid pipe deadlock, and non-`Exception` failures such as `SystemExit` are converted to `ToolExecutionError` rather than escaping into the parent.
- In isolated mode, `max_output_chars` is applied in the child before IPC. Optional `max_memory_bytes` applies `resource.RLIMIT_AS` where the runtime supports and accepts it. Missing support is rejected at configuration time; a platform that exposes `RLIMIT_AS` but rejects the actual limit operation now fails clearly with `ToolResourceLimitError`. Linux CI covers actual memory enforcement.
- Tests cover direct-call schema/permission enforcement, approval-argument consistency, parent `super().use()` delegation, timeout-path truncation, bounded quota waits, weak slot cleanup, hard process timeout cleanup, large isolated results, `SystemExit` containment, isolated output bounding, Linux memory enforcement, and resource-limit application failures.

Boundary: this is a library-level execution boundary, not a universal OS/container sandbox. Thread mode cannot forcibly terminate arbitrary synchronous code. Process isolation must be explicitly enabled for hard termination, requires picklable tool state, and still inherits the application's filesystem/network credentials. `RLIMIT_AS` availability and enforceability vary by platform; failures are surfaced rather than silently ignored.

Regex prompt validation is input hygiene, not protection against prompt injection or unauthorized execution. Permission enforcement belongs to the tool guard path above.

### 4. Checkpoint and transcript scalability — complete at the Neva persistence layer

PR #72 closes the remaining library-level resource-envelope gaps without
introducing a second persistence format:

- Existing exact serialized-file ceilings remain available through
  `save_snapshot(..., max_bytes=N)` and `load_snapshot(..., max_bytes=N)`,
  with streamed/atomic save staging from PR #64. `CheckpointLimits` now adds
  opt-in in-memory graph ceilings for maximum nesting depth, node count,
  individual serialized string/key UTF-8 bytes, and aggregate string/key UTF-8
  bytes. Defaults remain unlimited for backward compatibility.
- `create_snapshot(..., limits=...)` preflights live environment/conversation
  payloads before the environment-state deepcopy, so callers can cap the graph
  before duplication. `Environment.snapshot(limits=...)` applies the same
  envelope to version-2 runtime state and validates the complete final snapshot.
- Limited loads lexically preflight JSON nesting, node tokens, and raw string
  tokens by their decoded UTF-8 size before full UTF-8 decode or `json.loads`; the serialized byte buffer is
  released after decoding and before object parsing, and the parsed graph is
  checked again before constructing `SimulationSnapshot`. This closes the previously
  unbounded single-scalar/deep-structure path when callers opt into limits.
  The stdlib parser still materialises decoded text, so this is a bounded
  monolithic JSON parser rather than incremental object decoding.
- Runtime capture no longer uses a full `json.dumps` + `json.loads` roundtrip.
  Structural validation plus a JSON-shape-preserving clone keeps the previous
  normalization contract (tuples become lists; numeric/string subclasses become
  plain JSON values; accepted mapping keys become strings) without the encoded
  full-document intermediate. Runtime restore no longer `deepcopy()`s the
  entire saved runtime graph; it treats checkpoint input as read-only and
  stages only attributes, environment extras, scheduler state, and memory
  objects that must become independently owned. Restored memory-record metadata
  is deep-copied independently so mutating live metadata cannot mutate the saved
  snapshot used for rollback.
- First-party memory checkpoint adapters from PR #67 remain data-only and
  preserve configured callable/`MemoryBudget` identities; custom/third-party
  scheduler and memory implementations still require explicit checkpoint hooks.
  Deferred FAISS/PR #53 remains outside this section rather than being silently
  serialized.
- Transcript retention now has three independent opt-in axes:
  `max_turns`, per-message `max_turn_bytes`, and aggregate retained-message
  `max_history_bytes`. The aggregate budget truncates one oversized newest
  turn to the effective byte ceiling, then evicts oldest turns until the retained
  UTF-8 message bytes fit. Because `turns` remains mutable for compatibility,
  direct list/message edits are reconciled on the next `record_turn()` or
  serialization call rather than relying on a stale cached byte total. All
  policies survive serialization/restore; live agent responses are unchanged.
- Limit validation avoids redundant walks where correctness permits:
  `create_snapshot` validates its assembled source graph once before deepcopy,
  and `Environment.restore` validates the complete snapshot once before
  calling runtime restore. Snapshot creation still performs a final combined
  validation after runtime capture so node/string totals are enforced across
  environment, conversations, and runtime state together.
- The checkpoint scaling benchmark now reports
  `peak_python_bytes / checkpoint_bytes` amplification per stage in addition
  to elapsed time, peak Python allocations, and serialized bytes. This makes
  workload-specific externalization decisions measurable without imposing an
  arbitrary hardware-independent threshold.

Boundary: graph/file limits are opt-in and constrain serialized/checkpoint data,
not total process RSS or arbitrary custom-hook allocations. The checkpoint
format intentionally remains one JSON document: the existing streamed save plus
explicit graph/file ceilings bound the library path, while applications with
benchmark evidence that very large blobs dominate their checkpoints should
externalize those blobs in their own checkpoint hooks/storage rather than Neva
inventing a mandatory second format. `tracemalloc` benchmark figures exclude
OS page cache and destination/temp-file space.

### 5. Integration and coverage blind spots — complete for deterministic/local integrations

PR #73 closes the remaining deterministic integration/coverage gaps without
requiring live provider credentials or external model downloads:

- A dedicated optional-integration CI job installs CPU PyTorch, Transformers,
  FAISS/NumPy, and the declared Anthropic SDK on Python 3.11. The normal
  Python 3.11–3.14 matrix remains lightweight; the optional job makes the
  heavyweight paths mandatory rather than silently skipped.
- `TransformerAgent` is exercised with an actual locally generated, saved, and
  reloaded tiny T5 model. The test runs real `generate()` weights through the
  agent path while using a minimal tensor tokenizer, so CI does not depend on
  Hugging Face network availability. `torch` is now an explicit optional
  dependency in the `tools`/`all` extras, matching the documented default
  Transformer runtime.
- The existing `FaissVectorStoreMemory` implementation is no longer globally
  omitted from coverage. Dependency-enabled tests exercise construction,
  semantic search, recent recall, clear/reuse, missing-dependency failures, and
  a dedicated FAISS-module coverage gate of at least 80%. Review of the real
  dependency path also fixed an ordering bug: FAISS already returns nearest
  neighbours in metric order, so Neva no longer reverses L2 distances.
  PR #53 remains separately deferred as a PR, but #73 intentionally overlaps
  two of its test/coverage changes (removing the FAISS coverage omit and making
  the missing-dependency test work when FAISS is installed). #73 targets the
  FAISS implementation already present on main and adds broader correctness and
  integration coverage; #53 should be rebased or retired after #73 lands.
- Composite/Conditional scheduler coverage now includes group fairness
  independent of group size, changing conditional eligibility, pause/resume
  after condition changes, nested Composite schedulers, recursive environment
  propagation, nested termination hooks, and lifecycle removal. This exposed
  and fixed a Composite lifecycle bug: parent pause/resume is now propagated
  to the current child scheduler, including nested composites, so a resumed
  agent cannot remain stuck paused below the parent.
- The real requests transport now has deterministic loopback tests for read
  timeout (PR #52), connect timeout via a saturated Linux accept queue, and
  request-body write timeout against a server that accepts but does not read.
  All are verified through `GPTAgent` and bounded to one attempt when retries
  are disabled.
- Malformed HTTP integration cases cover invalid JSON, non-object JSON, and
  wrong Chat Completions shapes. The provider path now rejects invalid/non-object
  JSON with explicit `BackendError` diagnostics instead of leaking parser or
  attribute errors.
- SDK compatibility is exercised with the actual declared Anthropic package
  against a local loopback `/v1/messages` endpoint, including SDK request
  construction/object parsing and empty-content fail-closed behavior. No live
  Anthropic credentials or network calls are used.

Boundary: these tests establish deterministic compatibility with Neva's local
transport/model/dependency contracts. They do not establish compatibility with
every future third-party SDK release, every hardware backend, or live-provider
service behavior. Live-provider smoke examples/credentials are addressed in
section 9, and provider context/token budgeting in section 10.

### 6. Model-driven tool loop — complete at the Neva orchestration layer

PR #74 adds one bounded model/tool/feedback loop on top of the completed tool
schema and guard layer:

- `run_tool_loop(agent, task, ...)` and `AIAgent.run_tool_loop(...)` accept
  either an explicit synchronous model callable or, by default, the agent's
  `generate_model_output()` raw-generation hook. That hook consumes the
  already composed bounded loop prompt without `prepare_prompt()` adding
  memory/tool summaries a second time. TransformerAgent and GPTAgent preserve
  their normal `respond()` behavior outside the loop; GPTAgent's raw path
  still uses its provider admission/retry/spend/circuit/cache/telemetry stack
  while excluding previous conversation history. The model protocol is
  deliberately small and
  provider-neutral: each turn must be exactly one JSON object selecting either
  `{"action":"tool","name":...,"arguments":...}` or
  `{"action":"final","output":...}`.
- Registered tool metadata is advertised in structured form, including
  descriptions/capabilities and declarative `ArgumentSchema` field
  requirements/bounds where available. Advertisement is prompt-budget-aware:
  it uses full metadata when it fits, then compact name/description entries,
  then name-only entries, preserving every registered tool name rather than
  making the default `max_tools=32` unreachable under the default prompt
  ceiling. Duplicate tool names fail before a model call, and `max_tools`
  bounds the advertised registry.
- Model-selected calls never execute tools directly. Every call is converted to
  `ToolCall` and routed through `AIAgent.call_tool()`, preserving the
  section-3 enforcement path for agent/tool allowlists, synchronous approval,
  argument schemas, per-tool concurrency, timeout/output/memory limits, process
  isolation, observer usage accounting, and normalized tool errors.
- Unknown tools, schema violations, permission denials, execution failures, and
  malformed/oversized model actions become bounded feedback records rather than
  bypasses or unbounded retry loops. The model can correct the action on a later
  step; each model turn consumes the same global step budget.
- `ToolLoopConfig` explicitly bounds model turns/tool calls
  (`max_steps`), advertised tools, retained/parsed model-output characters,
  each feedback record, and orchestration-prompt characters. Old feedback is
  dropped first when needed to keep a later prompt within its envelope. If the
  task/protocol/tool names alone cannot fit, the loop fails before calling the
  model. On the default agent model path, a configured `max_prompt_chars`
  above the agent's own prompt-validator ceiling is rejected up front with
  `ToolLoopLimitError`; an explicit external `model=` callable is not
  constrained by the agent validator.
- An explicit `final` action terminates successfully. A model that never
  finalizes stops at `max_steps` with a non-success `ToolLoopResult`; there
  is no implicit unbounded agentic loop.
- Tests cover successful tool→feedback→final execution, schema correction,
  agent- and tool-level permission denial, unknown tools, malformed/duplicate
  JSON fields/action shapes, synchronous-model enforcement, non-string,
  awaitable, and oversized model outputs, raw Transformer/GPT default model
  paths without context re-wrapping, bounded tool feedback,
  prompt/tool-registry limits, duplicate tool names, schema metadata
  advertisement, direct finalization, and max-step exhaustion.

Boundary: this is a generic text-JSON orchestration layer, not provider-native
function calling and not a security sandbox. A provider/model may still ignore
the requested JSON protocol; those failures are bounded and fed back rather
than trusted. Tool outputs are labelled untrusted data but can still influence
model behavior. Authorization and execution safety therefore continue to live
in code-level tool schemas/guards, not in prompt instructions. The generic
`max_model_output_chars` limit bounds retained/parsing work after a model call
returns; provider-native generation/token limits must still be configured at
the provider layer. The separate provider response stream is described in
section 8; the tool loop itself still consumes a complete model action.

### 7. Reproducible experiments — complete at the Neva experiment boundary

PR #75 adds a single manifest/seeding/offline-replay workflow rather than
treating checkpointed scheduler RNG state as sufficient reproducibility:

- `prepare_reproducible_run(...)`, `seed_everything(...)`, and
  `Environment.seed(...)` seed the process-global Python RNG, optional
  NumPy/PyTorch RNGs, known Neva Python-random schedulers recursively, and
  explicit custom agent/scheduler `set_seed` hooks. Stable path-derived seeds
  separate nested scheduler streams. A custom scheduler's unrelated `_rng`
  object is not replaced. `PYTHONHASHSEED` mutation is opt-in through
  `set_child_hash_seed=True`; the current interpreter's hash seed remains an
  interpreter-startup property.
- Version-1 manifests record caller prompts, behavior-affecting environment
  config/state and per-agent failure policies, recursive scheduler state
  including EventDriven queues, provider/model/endpoint and generation settings,
  full prompt-validator policy, initial conversation state, supported built-in
  and subclassed memory state (including FAISS index digests), agent attributes,
  tool metadata plus ArgumentSchema/ToolGuard policy, cache policy/initial-state
  fingerprints, dependency/runtime versions, seed report, and audit metadata.
  Non-JSON-native metadata is deterministically normalized without collapsing
  distinct keys such as `1` and `"1"`. Unsupported custom memories must
  provide `checkpoint_state()` or `reproducibility_config()`; capture fails
  explicitly instead of silently omitting recall-affecting state.
- `RunManifest.fingerprint()` is replay-compatibility oriented and excludes
  audit-only `created_at`, dependency/runtime versions, seed-report details,
  metadata, and explanatory reproducibility notes. `audit_fingerprint()`
  includes those fields (except `created_at`). Replay tapes persist the
  compatibility payload so mismatches identify differing field paths instead of
  returning only an opaque hash failure.
- `AIAgent.replay_identity_resolver()` defines what replay validates.
  Ordinary/custom backends use the exact prompt. Provider-backed GPT agents use
  the scoped effective request identity, including provider/model/endpoint,
  raw-vs-history behavior, and the conversation history window that the provider
  request would contain. This closes the case where the same bare prompt with
  divergent history previously replayed silently.
- Recording reserves a sequence slot before backend execution and fills that
  slot afterward. Nested calls therefore remain in invocation order, while the
  tape lock is not held across slow/network model calls and unrelated agents are
  not serialized behind one provider call. Recording still preserves GPT's
  cooperative cancellation-aware backend construction.
- Replay records carry a prompt/request-identity SHA-256 plus a whole-record
  SHA-256 covering the response or failure metadata. Load rejects prompt edits,
  response edits, malformed response/error combinations, and malformed shapes.
  Replays also fail on mismatched order, exhausted/extra calls, or unconsumed
  records; recorded failures surface as `RecordedReplayError`.
- End-to-end tests cover seeded RandomScheduler record→persist→fresh replay,
  provider-backed GPT request-history divergence, nested/concurrent recording
  order, cross-machine-compatible replay fingerprints, actionable manifest
  mismatch paths, ShortTerm and real FAISS memory state, non-JSON-native memory
  metadata, tool schema/guard policies, custom RNG preservation, opt-in child
  hash seeding, tape-response integrity, and the earlier environment/cache/
  validator/endpoint/tamper/failure cases.

Boundary: reproducibility is exact at the recorded Neva model-request identity
boundary, not a claim that live providers or arbitrary application code are
deterministic. Provider-side revisions/routing/runtime state remain external;
framework/hardware deterministic settings remain the application's
responsibility. Callable identities in manifests are descriptive and cannot
prove that two arbitrary closures behave identically. Wall-clock timestamps,
UUIDs, and telemetry timing are not replayed. Manifests can contain prompts,
conversation/memory contents, attributes, tool descriptions and metadata; tapes
contain effective request identities and model outputs/failures. Protect both
artifact classes accordingly.

### 8. Streaming — implemented at the built-in provider boundary

- `GPTAgent.stream_response()` returns a single-use synchronous or asynchronous
  `StreamSession` that yields text deltas and a terminal completion event.
  OpenAI-compatible providers use incrementally parsed SSE; Anthropic and
  Gemini use their declared SDK streaming interfaces. Ordinary `respond()` and
  environment turns keep their existing non-streaming semantics.
- The producer uses a bounded queue and blocks when the consumer falls behind.
  Explicit `close()`/`aclose()` signals cooperative cancellation; provider
  admission and retries already observe the signal, while active synchronous
  socket/SDK reads remain subject to `request_timeout`.
- Pre-output transport failures may retry. Once text is emitted, errors raise
  `StreamInterruptedError` with uncommitted partial text, with no replay,
  cache entry, or conversation turn. Provider permits and spend reservations
  are released/settled even on partial failure or cancellation; available
  usage is recorded per stream. The response accumulation and cached reply
  have the same per-call character ceiling. Cache/history commit when the
  consumer accepts the terminal completion event; provider spend can still be
  incurred if the consumer closes after generation finishes.
- Completion events report first-token and full-completion seconds separately.
  LLM telemetry records both and identifies interrupted streams. Local HTTP
  integration tests exercise completion, partial EOF, cache/history, bounded
  cancellation, and async delivery without credentials.

Boundary: no Python thread can forcibly interrupt a blocked synchronous
provider read; configured request timeouts limit its lifetime. Backpressure
bounds pending deltas, while the completed response is accumulated up to the
configured response-character ceiling. A stream that is abandoned before its
terminal event cannot be resumed to recover its generated answer. Streaming is
not provider-native tool
calling; custom backend/wrapper implementations have no generic stream
contract and fail explicitly.

### 9. Real-provider examples — implemented as an opt-in smoke example

- `examples/live_provider_smoke.py` defaults to a labeled, deterministic
  scripted response with no credentials or provider request. Live generation
  requires `--live`, a positive finite `--max-spend-usd`, and `OPENAI_API_KEY`.
- The live path sends one request to the built-in OpenAI provider using
  `gpt-4o-mini`, at most 128 output tokens, a 2,000-character formatted
  context cap, a 15-second request timeout, and zero retries. It prints the
  generated answer, estimated spend, and whether token counts came from the
  provider on success. Provider failures return a short credential/account/
  network hint and recorded estimated spend without exposing the API key or
  printing a traceback. README documents commands, credential setup, expected cost
  behavior, and current-pricing verification.
- Tests prove the default path avoids HTTP even with credentials present,
  invalid or missing live configuration fails before HTTP, a stubbed live
  response takes the provider request path once with configured limits, and
  insufficient estimated budget blocks the request, and stubbed 401/timeout
  failures exit cleanly with recorded spend. No test needs a real key.

Boundary: Neva's prices and spend reservation are estimates, not an
authoritative provider billing limit; a timed-out request might incur charges
even if Neva records zero spend. The character cap is not a model-aware
token-context guarantee. Live service availability and provider billing are
external and are not exercised in CI. Existing multi-agent showcases remain
scripted offline demonstrations.

### 10. Model-aware context budgeting — complete at the opt-in Neva boundary

- `ModelContextBudget` binds a configured provider/model, a positive context
  ceiling, a model-specific request-token counter, and a versioned counter
  identity. The counter sees the actual role/content message sequence for
  OpenAI-compatible and Anthropic requests or the flattened Gemini text, so it
  can include model-specific tokenization and framing. The optional
  `openai_chat_counter()` uses `tiktoken` plus the OpenAI cookbook's message and
  reply-priming estimate for `gpt-4o-mini`; unsupported models fail explicitly.
- `GPTAgent(context_budget=...)` keeps the recent contiguous history suffix
  only if both existing formatted-text character limits and
  `request tokens + max_output_tokens <= max_tokens` hold. Output reservation
  is validated at construction. If the prepared current request cannot fit,
  `ConfigurationError` is raised before cache, admission, or HTTP; the prompt
  is never truncated to fit. Streaming and raw model-tool paths use the same
  history/check. Spend preflight uses model request tokens, including overhead,
  when the token envelope is enabled.
- Cache/replay identity and run manifests include the limit and counter ID;
  tests cover role overhead, suffix eviction, provider payload, Gemini shape,
  invalid/mismatched counters and models, current-prompt rejection, streaming
  preflight, raw mode, and spend reservation. The existing character-only
  behavior remains the default.

Boundary: callers must select the correct provider/model context ceiling and
counter for their endpoint. The OpenAI cookbook message-count function is an
estimate, not an immutable guarantee; other providers require caller-supplied
model-aware counters. Provider-side revisions, hidden framing, tool/image
payloads, and custom gateways can differ from local estimates. This is an
opt-in preflight bound, not a universal provider context guarantee. Stored
conversation retention remains independently configurable.

## Recommended next priorities

Sections 1–10 are implemented at their documented Neva boundaries. Further
work should follow measured provider-token discrepancies or application needs,
not assume a universal context or billing guarantee from local estimates.

## Local delivery status

- [x] Check merged PRs and latest remote main before starting overlapping work.
- [x] Verify existing circuit breaker and context-budget fix are present.
- [x] Add and exercise two-agent HTTP and actual read-timeout coverage.
- [x] Correct inconsistent fake-clock setup in the existing limiter test.
- [x] Pass full local tests and quality checks.
- [x] Commit/push the tests and refreshed gap document; open PR #52.
- [x] Run CI and merge HTTP integration work through PR #52.
- [x] Expand scheduler edge-case tests on an independent branch; no production changes.
- [x] Run CI and merge scheduler coverage after review (PR #54).
- [x] Add conversation retention limits and checkpoint file-size limits (PRs #55, #56).
- [x] Add per-agent failure policies (PR #57), rate-limiter cancellation (PR #58), and spend budgets (PR #59).
- [x] Add durable failure records and replay control (PR #60).
- [x] Tool-call guardrails (PR #61).
- [x] Validated tool argument schemas (PR #62).
- [x] Add dependency-enabled FAISS CI/coverage for the implementation on main; PR #73 supersedes the overlapping coverage/test portions of deferred PR #53.
- [x] Close deterministic optional-integration, scheduler lifecycle/fairness, and transport/SDK coverage gaps (PR #73).
- [x] Complete bounded model-driven tool orchestration with schema/guard feedback and termination limits (PR #74).
- [x] Complete run manifests, unified seeding, and deterministic offline model-boundary replay (PR #75).
- [x] Add bounded provider streaming, cancellation, partial-failure handling, and separate first-token/completion latency accounting.
- [x] Add an opt-in live-provider example, deterministic offline mode, credential/cost guidance, and bounded-call tests (section 9).
- [x] Add model-bound request token accounting, output reservations, preflight failures, and documentation (section 10).

The abandoned circuit-breaker test and previous gap document are preserved in the named git stash `circuit-breaker TDD test + gap doc`; that obsolete test was not applied to the new branch.
