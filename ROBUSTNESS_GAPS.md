# Neva: Robustness and Product Gaps

## Verified baseline and scope

Updated after checking GitHub and local `main` at `dd08cd0` (PRs #49–#52 merged). The earlier document incorrectly treated PR #49 as pending and the circuit breaker as unimplemented; those statuses are superseded below.

- Merged baseline: 187 tests passed, one skipped (FAISS), 83.93% coverage locally.
- Local branch `test/multi-agent-http-timeouts`: 190 passed, one skipped, 83.93% coverage. Black, isort, Flake8, MyPy, and Bandit passed.
- HTTP integration coverage merged in [PR #52](https://github.com/davletovb/neva/pull/52). No live-provider calls were made.
- FAISS follow-up on `test/faiss-integration-ci`: real FAISS 1.15.1 / NumPy 1.26.4 on Python 3.11: 191 passed, zero skipped, inclusive coverage 83.77%. Core-only Python 3.9: 190 passed, one optional skip, inclusive coverage 82.07%. Native Linux CI execution remains pending.
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

Character limits remain heuristics: they are not model-specific token/context limits or an output-token reservation system. ConversationState itself remains unbounded even when transmitted history is windowed.

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

Full-suite verification exposed an existing clock inconsistency in `test_rate_limiter_honours_rate`: construction used real monotonic time before the test switched to a fake zero-based clock. The merged test change constructs the limiter after installing the fake clock and asserts the exact half-second wait. No production behavior was changed.

## Remaining gaps

### 1. Shared rate, concurrency, and spend budgets — partial

- Automatic provider/account-wide coordination across agents and processes.
- Token/spend budgets and hard enforcement; static cost estimates do not enforce spending.
- Global cross-thread/cross-loop concurrency limits, if required.
- Shared-limiter fairness, cancellation, and aggregate contention tests.

Correction to the original review: receivers use their own backend limiters, not the sender's limiter. Explicitly sharing a limiter already shares that instance's request-rate budget.

### 2. Durable failure recovery — partial

- Per-agent recovery policies, beyond the current environment-wide policy.
- Durable failure records/dead-letter storage and replay controls.
- Escalation policies and richer recovery-state observability.
- Broader concurrent and interruption-path testing of the existing circuit breaker.

The circuit breaker itself is no longer an unimplemented gap. Nor should the old unconditional claim that every turn burns its full retry budget be retained.

### 3. Tool schemas, permissions, and execution limits — open

- General validated argument schemas.
- Permission/approval checks independent of model instructions.
- Appropriate time, output, and resource limits for side-effecting tools.

Regex prompt validation is input hygiene, not protection against prompt injection or unauthorized execution. Narrow math bounds and corrected README wording do not close this gap.

### 4. Checkpoint and transcript scalability — partial

- Checkpoint size/resource ceilings and large-state performance benchmarks.
- Incremental/externalized persistence where justified by measured scale.
- Support for components currently requiring custom checkpoint hooks.
- Retention limits for stored conversation state, distinct from request-history trimming.

Deep-copy isolation may be expensive at scale; no benchmark establishes its limits yet.

### 5. Integration and coverage blind spots — partial

- Tests with actual small transformer-model weights.
- FAISS dependency-enabled CI is configured on `test/faiss-integration-ci`, pending merge/remote verification. The new job requires imports and zero skipped FAISS tests, runs the full suite, and uploads inclusive coverage. The global FAISS coverage exclusion is removed. Deeper FAISS edge-case coverage remains open (78% locally with native dependencies).
- Focused Composite/Conditional scheduler edge-case coverage.
- Connect/write timeout and additional malformed-response/SDK integration cases.

The two-agent loopback HTTP and actual read-timeout gap is covered by merged PR #52. Review-time module coverage remains approximately TransformerAgent 54%, CompositeScheduler 61%, ConditionalScheduler 75%.

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
- [x] Add FAISS dependency-enabled CI configuration and remove its coverage exclusion.
- [x] Run the real-FAISS targeted tests (zero skips) and both full-suite configurations locally.
- [ ] Verify the new FAISS job on GitHub and merge the follow-up.

The abandoned circuit-breaker test and previous gap document are preserved in the named git stash `circuit-breaker TDD test + gap doc`; that obsolete test was not applied to the new branch.
