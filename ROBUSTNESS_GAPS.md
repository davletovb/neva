# Neva: Robustness and Product Gaps

## Verified baseline and scope

Updated against `main` at `88815488` (PRs #49–#52, #54–#71 merged).

- Checkpoint file-size limits are merged: opt-in positive UTF-8 byte counts; limited loads read in 64 KiB chunks (total bounded at limit + 1) and reject overflow before decoding or parsing; oversized saves leave existing files untouched. PR #64 additionally streams save serialization into a sibling temporary file instead of materialising the complete JSON string and byte string, fsyncs it, and atomically installs it with `os.replace` so existing checkpoints survive serialization, staging-write, and replacement failures.
- FAISS PR #53 is explicitly deferred for user evaluation; none of its changes are included in this branch.
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

Character limits remain heuristics: they are not model-specific token/context limits or an output-token reservation system. ConversationState is unbounded by default; PR #55 adds an opt-in stored-turn retention limit independent of the transmitted history window.

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
  token bytes before UTF-8 decode or `json.loads`; the parsed graph is checked
  again before constructing `SimulationSnapshot`. This closes the previously
  unbounded single-scalar/deep-structure path when callers opt into limits.
  The stdlib parser still materialises decoded text, so this is a bounded
  monolithic JSON parser rather than incremental object decoding.
- Runtime capture no longer uses a full `json.dumps` + `json.loads` roundtrip.
  Structural validation plus a JSON-shape-preserving clone keeps the previous
  normalization contract (tuples become lists; JSON mapping keys become
  strings) without the encoded full-document intermediate. Runtime restore no
  longer `deepcopy()`s the entire saved runtime graph; it treats checkpoint
  input as read-only and stages only attributes, environment extras, scheduler
  state, and memory objects that must become independently owned.
- First-party memory checkpoint adapters from PR #67 remain data-only and
  preserve configured callable/`MemoryBudget` identities; custom/third-party
  scheduler and memory implementations still require explicit checkpoint hooks.
  Deferred FAISS/PR #53 remains outside this section rather than being silently
  serialized.
- Transcript retention now has three independent opt-in axes:
  `max_turns`, per-message `max_turn_bytes`, and aggregate retained-message
  `max_history_bytes`. The aggregate budget truncates one oversized newest
  turn to the effective byte ceiling, then evicts oldest turns until the retained
  UTF-8 message bytes fit. All policies survive serialization/restore; live
  agent responses are unchanged.
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

### 5. Integration and coverage blind spots — partial

- Tests with actual small transformer-model weights.
- FAISS dependency-enabled CI; its implementation remains excluded from coverage and its local test is skipped.
- Focused Composite/Conditional coverage is merged in PR #54: validation, group migration/removal, child unavailability, environment propagation, scheduler overrides, predicate errors/updates, pause filtering, and termination hooks. Deeper nested lifecycle and fairness testing remains open.
- Connect/write timeout and additional malformed-response/SDK integration cases.

The two-agent loopback HTTP and actual read-timeout gap is covered by merged PR #52. TransformerAgent remains at 54%; the scheduler branch measures CompositeScheduler at 96% and ConditionalScheduler at 100% statement coverage. FAISS remains excluded on main while PR #53 is deferred.

### 6. Model-driven tool loop — not implemented

Structured tool selection, validation, permission checks, execution, result feedback, and termination need a bounded integrated loop. Existing programmatic tool calls are not that loop.

### 7. Reproducible experiments — not implemented as a complete system

- Run manifests with provider/model identifiers, prompts, generation settings, dependency versions, seeds, scheduler configuration, and cache policy.
- Unified seeding and deterministic offline replay.
- Documented limits of live-provider reproducibility.

Checkpointed scheduler RNG state alone does not establish reproducibility.

### 8. Streaming — not implemented

- Streaming interface with cancellation, partial-failure handling, and backpressure.
- First-token versus full-completion latency accounting.

### 9. Real-provider examples — not implemented

Keep deterministic offline examples, but add an opt-in live-provider example with explicit credentials, cost expectations, and limits. Clearly distinguish scripted behavior from generated behavior.

### 10. Model-aware context budgeting — partial

- Token-based budgeting for the selected model and message-format overhead.
- Explicit output-token reservations.
- Documented behavior when the current request cannot fit.

The implemented formatted-text character cap is useful, but it is not a universal provider-context guarantee.

## Recommended next priorities

1. Shared provider/spend budgets and durable failure records for unattended runs.
2. Tool schemas, permissions, and limits before autonomous tool execution.
3. Checkpoint/transcript resource ceilings and optional-integration/scheduler tests.
4. Model-aware context limits and experiment manifests/replay.
5. Bounded tool loops, streaming, and opt-in live-provider examples.

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
- [ ] User decision on deferred FAISS PR #53.

The abandoned circuit-breaker test and previous gap document are preserved in the named git stash `circuit-breaker TDD test + gap doc`; that obsolete test was not applied to the new branch.
