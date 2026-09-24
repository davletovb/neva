# Security Policy

## Supported versions

Neva is pre-1.0. Security fixes land on `main` and ship in the next patch
release; older commits, forks, and unpinned environments are not maintained.

| Version | Supported |
| ------- | --------- |
| `main` (0.1.x) | ✅ |
| Anything else | ❌ |

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue:

1. Use GitHub's private vulnerability reporting on this repository
   (**Security → Report a vulnerability**).
2. If private reporting is unavailable, contact the maintainer through the
   GitHub account that owns the repository.

Include the affected version or commit, a minimal reproduction, and the impact
you believe it has. You can expect an acknowledgement within a week, and either
a fix with a regression test or a written explanation of why the report is out
of scope.

## Scope notes

- Provider credentials are never logged, serialized, or replayed by Neva.
  Reports that a key leaks into checkpoints, run manifests, logs, telemetry, or
  exception messages are in scope and treated as high severity.
- `ToolGuard`, prompt validators, checkpoint limits, spend budgets, and rate
  limiters are guardrails for accidental resource use in a trusted
  application, not security boundaries between untrusted parties; the tracker
  documents each boundary explicitly.
- The library performs network I/O only through provider calls, the optional
  tools (Wikipedia, translation), and the MLflow/OpenTelemetry exporters you
  configure. Reports that an unconfigured path performs I/O are in scope.
- Denial-of-service reports that require unbounded *trusted* input (for example
  a caller passing an enormous prompt with no configured limits) are expected
  behavior of the documented opt-in limits, not vulnerabilities.
