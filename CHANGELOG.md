# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Packaging metadata for distribution: project URLs, keywords, and trove
  classifiers in `pyproject.toml`.
- PEP 561 `py.typed` marker so downstream projects consume Neva's type hints.
- `memory` extra bundling the optional `faiss-cpu` and `numpy` dependencies used
  by the FAISS vector store.
- Repository `.gitignore` covering build, coverage, cache, and runtime
  artifacts.

### Fixed
- Corrected a malformed `RUN` instruction in the `Dockerfile` that contained a
  stray line continuation.
- Updated the README "Last Commit" badge to point at the `main` branch.

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
