# Production-Readiness Acceptance Plan

This document defines how a Neva release candidate becomes a defensible
production-readiness claim: the checks to run, the evidence to record, and the
exit criteria for each area, against a declared scope.

It is an acceptance checklist for a release candidate — not a roadmap and not a
capability list. Capabilities and boundary statements live in
[CAPABILITIES.md](CAPABILITIES.md); chronological changes live in
[CHANGELOG.md](CHANGELOG.md).

## Declared scope

- Neva is a **library-level toolkit**. Its controls are bounded and fail-closed,
  but it is not a universal sandbox or an account-wide policy engine; the
  boundary statements under "Security model" in CAPABILITIES.md are the warranty.
- Supported Python: 3.11–3.14. Heavy integrations are optional extras, and CI's
  optional-integrations job keeps them mandatory (never silently skipped) at the
  versions pinned in `requirements-optional.txt`.
- CI is intentionally loopback-only: no test requires provider credentials or
  network egress. Live-provider acceptance (Section B) therefore happens
  out-of-band, on a declared provider/model matrix.

## How to use this plan

For a release candidate `X.Y.Z` at commit `<sha>`:

1. Freeze the commit and note it in every evidence row.
2. Run Sections A–D; every exit criterion must be **pass** or **explicitly
   deferred** with a recorded rationale and an issue link.
3. Attach the completed evidence table (Section E) to the release notes.

Precondition: no open P0/P1 defects at the freeze commit. Tracked follow-ups
(dependency holds, upstream interactions) must each be assessed either as "does
not affect the declared scope" or as scheduled work before the release.

## Section A — Distributable artifact

Goal: a consumer installs a pinnable version without a git SHA, and the artifact
reproduces the tested behavior.

| # | Check | Procedure | Exit criterion |
|---|-------|-----------|----------------|
| A1 | Release policy recorded | Decide and document: semver semantics for a 0.x library (breaking changes may ship in minor releases until 1.0), the deprecation window, and the requirement of migration notes for breaking changes. | Policy section exists and is linked from the README. |
| A2 | Lock consistency | `poetry check --lock` on the frozen commit. | Exit 0. |
| A3 | Artifact builds | `python -m build` (the same command as CI's `package` job). | sdist and wheel produced. |
| A4 | Wheel installs clean | Fresh venv → `pip install dist/*.whl` → `python scripts/wheel_smoke.py`. Repeat per supported Python. | Smoke script passes on 3.11–3.14. |
| A5 | Version importable | `python -c "import neva; print(neva.__version__)"` in the wheel venv. | Prints `X.Y.Z`. |
| A6 | Release published and tagged | Tag `vX.Y.Z` and publish to the package index — or record the explicit "consume by git SHA" limitation. | Tag exists; index install verified. |

## Section B — Live-provider acceptance

Goal: the declared provider surface works against the real APIs, not only the
loopback fakes.

Matrix: OpenAI-compatible (OpenAI, `/v1/chat/completions`), Anthropic, Google
Gemini (`google-genai`), xAI Grok.

Preconditions: credentials supplied via environment variables; a spend ceiling
per run; a throwaway key where the provider supports it.

| # | Check | Procedure | Exit criterion |
|---|-------|-----------|----------------|
| B1 | Bounded first call per provider | `python examples/live_provider_smoke.py --live --max-spend-usd <cap>` for OpenAI; replicate its pattern per other provider (the shipped example targets OpenAI). | One response per provider; no secrets in logs or transcripts. |
| B2 | Streaming edge cases | `stream_response()` per provider: normal stream, empty final delta, consumer abandonment mid-stream (`StreamInterruptedError`), provider error mid-stream. | Observed behavior matches CAPABILITIES.md; no hung producer threads. |
| B3 | Rate limits and retries | Drive 429s (parallel calls) and transient 5xx where the provider offers a fault path; observe backoff, circuit breaker, and half-open recovery. | Retries respect the configured policy; breaker opens and recovers. |
| B4 | Token accounting | Compare client-recorded usage against the provider's billed usage for a fixed prompt sample. | Records match within noise, or the mismatch is filed as a defect or documented drift. |
| B5 | Timeouts and failure modes | Force transport timeouts (unroutable host) and invalid credentials per provider. | Fail closed with `BackendError`/`ConfigurationError`; no retry storms. |
| B6 | Deprecation watch | Record the provider API and SDK versions exercised and their deprecation status. | Each is supported at freeze date or has a tracked migration. |

Note: transport-level timeouts raised by provider SDKs are not reclassified as
retryable by Neva's retry policy (pre-existing behavior for SDK-based
providers). B5 records the observed behavior per release so this boundary stays
deliberate rather than accidental.

## Section C — Operational envelope

Goal: a measured, documented envelope instead of best-guess statements.

| # | Check | Procedure | Exit criterion |
|---|-------|-----------|----------------|
| C1 | Checkpoint scaling | Re-run the checkpoint scaling benchmark (agents × transcript sizes); record save/load time and peak RSS. | Numbers recorded; no super-linear blowup on the tested range. |
| C2 | Soak | ≥8 h (target 24 h) run with periodic checkpoints and memory-store writes; sample RSS. | Memory flat within ±10%; no file-descriptor or thread leaks; every checkpoint loads. |
| C3 | Concurrency | Exercise the thread-safety matrix: declared-safe combinations under threads; probe the "not generally thread-safe" objects to confirm the boundary statement. | Documented safe/unsafe matrix; no silent corruption in safe combinations. |
| C4 | Memory store under load | Long-run FAISS add/search cycle (the C2 run can cover it). | Recall consistent; index growth bounded by configured limits. |
| C5 | Import-time budget | Measure cold and warm `import neva` per supported Python on the release runner. | Values recorded; any guardrail whose budget is close to the measured time is widened or re-designed. Known risk: the heaviest optional environment has measured ≈3.3 s — the same latent failure mode as any isolated-worker budget that assumes a 2 s startup cost. |

## Section D — Deployment threat model

Goal: turn the honest scope ("library-level controls, not a universal sandbox")
into deployment guidance.

| # | Check | Procedure | Exit criterion |
|---|-------|-----------|----------------|
| D1 | Trust boundaries | Document: application code is trusted; model output, tool endpoints, and retrieved content are untrusted. | Section published; consistent with CAPABILITIES.md. |
| D2 | Tool sandboxing | State the requirement that high-risk tools run in a subprocess or container with least-privilege filesystem and network egress (the library provides allowlists, approvals, and limits — not isolation). | Guidance published. |
| D3 | Prompt injection via tool output | Run a canned malicious-tool-output scenario (retrieved text tries to trigger a high-risk tool); assert approval gating and allowlists hold. | The tool call requires explicit approval; nothing executes automatically. |
| D4 | Secrets | Verify manifests capture no environment dump (explicit fields only), telemetry redaction settings, and the documented "secrets never in prompts or telemetry" rule. | Evidence attached (source references + one recorded run). |
| D5 | Telemetry and PII | Document the OTel toggles, default-off exporters, and the fields Neva emits. | Section published. |
| D6 | Process isolation | State requirements for deployments that execute generated code (isolated worker processes, resource limits). | Section published. |
| D7 | Approval-flow audit | Walk one full tool-approval flow (manual approve, timeout, denial) against the documentation. | Behavior matches documentation. |

## Section E — Evidence record and go/no-go

Record per executed row: `id | commit | procedure | observed result | pass/fail/deferred | evidence link | date`.

Go/no-go: release only when (a) A1–A6 pass, (b) B1–B6 pass or carry recorded,
scoped deferrals, (c) C1–C5 are recorded with no unexplained regressions against
the previous release, and (d) D1–D7 pass with the guidance published. Anything
else is a dated exception attached to the release.

## What this plan does not establish

- It is not a substitute for a security review for regulated or adversarial
  deployments.
- It does not turn the library into a sandbox: Sections D2 and D6 state the
  external controls each deployment must supply.
- It does not guarantee provider behavior; Sections B and D record the versions
  and dates actually exercised.
