# R0 prior art and scope check

Checked against public project descriptions on 2026-10-19. This is a
*capability-positioning assessment*, not a comparative benchmark or an
assertion that another project's implementation has been fully audited.

| Framework | Strength relevant to our decision | Why Hiyal still has a distinct research objective |
| --- | --- | --- |
| [Mesa](https://github.com/projectmesa/mesa) | Established Python general-purpose agent-based modeling | Hiyal prioritizes Rust-only native distribution and explicit effect-call replay; Mesa remains a good reference rather than a feature-parity target |
| [Agents.jl](https://github.com/JuliaDynamics/Agents.jl) | Rich Julia-native ABM with discrete-time/event queues and research integration | Hiyal deliberately excludes an embedded Julia/Python runtime; JSONL permits analysis with these ecosystems |
| [krABMaga](https://github.com/krABMaga/krABMaga) | Rust ABM state/agent traits, scheduler and visualization | **Strongest Rust comparator**; before broadening Hiyal, evaluate whether its primitives could host a trial-owned effect ledger and deterministic replay more cheaply |
| [Concordia](https://github.com/google-deepmind/concordia) | Generative social simulation with components, environments and game-master mediation | Rich LLM agent behavior, but Hiyal targets reproducibility contracts across scripted/replayed/live effects and native simulation rather than a similar game-master framework |
| [OASIS](https://github.com/camel-ai/oasis) | Social media interaction simulation at large claimed scales | A domain-specific social-platform focus; Hiyal should be domain-neutral with user-defined actions/transitions |
| [AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety) | Large-scale LLM-native urban/social simulations; AgentSociety 2 adds distributed execution | Useful research workflow comparator; Hiyal's defining constraints are auditable effects, exact offline replay and open event logs, not claims to outscale distributed frameworks |

**Go/no-go test:** independently written research world must be expressible as a
consumer crate, reproduce its scripted path, resume/fork correctly, and expose
a traceable effect-cost ledger. If krABMaga plus modest adapters can satisfy
these contracts with less code, revisit building a full new engine.

Do not claim unique functionality, universal determinism or superiority
without checking implementation tests and measured like-for-like workloads.
