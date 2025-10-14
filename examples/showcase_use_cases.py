"""Showcase multiple Neva agent simulations across diverse scenarios.

This script expands on the quickstart example by demonstrating how to:

* orchestrate tool-augmented research with Wikipedia lookups,
* coordinate hierarchical leader-follower teams,
* simulate emergent roleplay between game NPCs,
* facilitate structured debates and negotiation exercises,
* explore social dynamics in cooperative neighbourhood planning,
* script adaptive game-AI party coordination scenes, and
* run productivity swarms or customer support triage routines.

All simulations rely on lightweight, fully offline language-model backends so
they can be executed without external dependencies. Optional integrations (such
as the Wikipedia API) are stubbed to keep the examples reproducible in any
environment.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import neva.tools as tool_module
from neva.agents import AgentManager
from neva.environments import BasicEnvironment
from neva.schedulers import RoundRobinScheduler
from neva.tools import MathTool, WikipediaTool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def install_wikipedia_stub() -> None:
    """Ensure the Wikipedia tool works without the optional dependency.

    The :class:`~neva.tools.WikipediaTool` relies on the third-party ``wikipedia``
    package. This helper installs a tiny offline stub when the dependency is
    missing so the example can run in hermetic test environments.
    """

    if getattr(tool_module, "wikipedia", None) is not None:
        return

    class _OfflineWikipedia:
        @staticmethod
        def summary(query: str, sentences: int = 2) -> str:
            return (
                f"Offline summary for '{query}' with {sentences} sentences. "
                "Replace this stub with the real wikipedia package for live data."
            )

    tool_module.wikipedia = _OfflineWikipedia()  # type: ignore[attr-defined]


class TranscriptEnvironment(BasicEnvironment):
    """Environment that tracks a rolling transcript of agent outputs."""

    def __init__(
        self,
        name: str,
        description: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__(name, description, scheduler)
        self.transcript: List[str] = []

    def recent_dialogue(self, limit: int = 3) -> str:
        if not self.transcript:
            return "No dialogue yet."
        return " | ".join(self.transcript[-limit:])

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.transcript.append(message)
        return message


def make_persona_backend(
    name: str,
    persona: str,
    formatter: Callable[[str], str],
) -> Callable[[str], str]:
    """Return a stub LLM backend that formats responses consistently."""

    def backend(prompt: str) -> str:
        return f"{name} ({persona}) {formatter(prompt)}"

    return backend


def run_simulation(
    title: str,
    environment: TranscriptEnvironment,
    steps: int,
) -> None:
    """Execute ``steps`` environment turns and print a compact transcript."""

    print(f"\n=== {title} ===")
    messages = environment.run(steps)
    for index, message in enumerate(messages, start=1):
        if message is None:
            continue
        print(f"Turn {index}: {message}")

    observer = environment.scheduler.simulation_observer
    snapshot = observer.latest_snapshot()
    interesting_metrics = {
        key: snapshot[key]
        for key in (
            "turn_count",
            "per_agent_participation",
            "tool_usage_counts",
        )
        if key in snapshot
    }
    print("Metrics:", interesting_metrics)


# ---------------------------------------------------------------------------
# Tool-augmented research demo
# ---------------------------------------------------------------------------


class ResearchLabEnvironment(TranscriptEnvironment):
    """Environment encouraging tool usage for factual grounding."""

    def __init__(
        self,
        topic: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Research Lab", "Collaborative fact-finding sprint.", scheduler)
        self.topic = topic
        self.state["research_notes"] = []

    def context(self) -> str:  # type: ignore[override]
        latest_note = self.state["research_notes"][-1] if self.state["research_notes"] else "None"
        return (
            f"Topic: {self.topic}. Use the wikipedia tool to ground claims. "
            f"Latest verified note: {latest_note}. "
            f"Recent discussion: {self.recent_dialogue()}"
        )


def make_researcher_backend(
    name: str,
    persona: str,
    *,
    tool: WikipediaTool,
    environment: ResearchLabEnvironment,
) -> Callable[[str], str]:
    """Create a backend that consults Wikipedia exactly once."""

    used_tool = False

    def formatter(prompt: str) -> str:
        nonlocal used_tool
        topic = environment.topic
        if not used_tool:
            used_tool = True
            summary = tool.use(topic)
            environment.state["research_notes"].append(summary)
            return f"consults wikipedia about {topic}: {summary}"
        context = environment.recent_dialogue()
        return f"synthesises findings about {topic} using prior notes ({context})."

    return make_persona_backend(name, persona, formatter)


def make_analyst_backend(
    name: str,
    persona: str,
    *,
    environment: ResearchLabEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        latest = environment.state["research_notes"][-1] if environment.state["research_notes"] else "hypothesis pending"
        return f"reviews the shared notes and proposes experiments building on {latest}."

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Hierarchical leader-follower demo
# ---------------------------------------------------------------------------


class MissionControlEnvironment(TranscriptEnvironment):
    """Environment coordinating hierarchical directives."""

    def __init__(
        self,
        directives: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Mission Control", "Leader assigns directives to specialists.", scheduler)
        self._directives: Iterator[str] = iter(directives)
        self.state["directive"] = "Awaiting assignment"
        self.state["progress"]: Dict[str, str] = {}

    def next_directive(self) -> str:
        try:
            directive = next(self._directives)
        except StopIteration:
            directive = "Wrap up and summarise outcomes"
        self.state["directive"] = directive
        return directive

    def context(self) -> str:  # type: ignore[override]
        progress = ", ".join(
            f"{agent}: {status}" for agent, status in sorted(self.state["progress"].items())
        ) or "No updates yet"
        return (
            f"Active directive: {self.state['directive']}. "
            f"Progress so far: {progress}. "
            f"Recent chatter: {self.recent_dialogue()}"
        )


def make_leader_backend(
    name: str,
    persona: str,
    *,
    environment: MissionControlEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        directive = environment.next_directive()
        return f"issues directive -> {directive}"

    return make_persona_backend(name, persona, formatter)


def make_specialist_backend(
    name: str,
    persona: str,
    *,
    speciality: str,
    environment: MissionControlEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        directive = environment.state["directive"]
        update = f"{speciality} executing '{directive}'"
        environment.state["progress"][name] = update
        return f"acknowledges directive '{directive}' and reports {update}."

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Game NPC emergent behaviour demo
# ---------------------------------------------------------------------------


class TavernEnvironment(TranscriptEnvironment):
    """Lightweight environment simulating NPC banter."""

    def __init__(
        self,
        events: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Tavern", "A cosy tavern filled with dynamic NPCs.", scheduler)
        self.events: List[str] = list(events)
        self.state["event_index"] = 0

    def _current_event(self) -> str:
        index = int(self.state["event_index"]) % len(self.events)
        return self.events[index]

    def context(self) -> str:  # type: ignore[override]
        return (
            f"Current tavern event: {self._current_event()}. "
            f"Play to your archetype and riff on the rumours. "
            f"Recent rumours: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.state["event_index"] = (int(self.state["event_index"]) + 1) % len(self.events)
        return message


def make_npc_backend(
    name: str,
    persona: str,
    *,
    environment: TavernEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        event = environment._current_event()
        rumours = environment.recent_dialogue()
        return f"reacts to '{event}' while weaving it into tavern gossip ({rumours})."

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Productivity swarm demo
# ---------------------------------------------------------------------------


class ProductivityEnvironment(TranscriptEnvironment):
    """Environment coordinating a self-organising productivity swarm."""

    def __init__(
        self,
        backlog: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Productivity Swarm", "Async collective planning.", scheduler)
        self.backlog: Deque[str] = deque(backlog)
        self.state["active_task"]: Optional[str] = None
        self.state["swarm_progress"]: Dict[str, str] = {}
        self.state["completed_tasks"]: List[str] = []

    def _ensure_active_task(self) -> None:
        if self.state["active_task"] is None and self.backlog:
            self.state["active_task"] = self.backlog.popleft()

    def context(self) -> str:  # type: ignore[override]
        active = self.state["active_task"] or "None"
        backlog_preview = ", ".join(list(self.backlog)[:2]) or "Empty"
        progress = ", ".join(
            f"{agent}: {status}" for agent, status in sorted(self.state["swarm_progress"].items())
        ) or "No updates yet"
        completed = ", ".join(self.state["completed_tasks"][-2:]) or "None"
        return (
            f"Active task: {active}. Backlog preview: {backlog_preview}. "
            f"Progress: {progress}. Completed recently: {completed}. "
            f"Recent sync: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        self._ensure_active_task()
        message = super().step()
        return message


def make_orchestrator_backend(
    name: str,
    persona: str,
    *,
    environment: ProductivityEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        environment._ensure_active_task()
        task = environment.state["active_task"]
        if task is None:
            return "confirms all backlog items are complete and thanks the swarm."
        environment.state["swarm_progress"].clear()
        return f"announces focus on '{task}' and requests progress updates."

    return make_persona_backend(name, persona, formatter)


def make_worker_backend(
    name: str,
    persona: str,
    *,
    specialty: str,
    environment: ProductivityEnvironment,
    tool: Optional[MathTool] = None,
) -> Callable[[str], str]:
    iteration = 0

    def formatter(prompt: str) -> str:
        nonlocal iteration
        iteration += 1
        task = environment.state["active_task"]
        if task is None:
            return "reviews documentation while awaiting the next assignment."

        progress = environment.state["swarm_progress"]
        if iteration == 1 and tool is not None:
            estimation = tool.use("2 * 3 + 1")
            progress[name] = f"planning estimates ({estimation} hours)"
            return (
                f"takes ownership of '{task}', runs quick maths with the calculator tool, "
                f"and shares the estimate of {estimation} hours for the team."
            )

        if iteration == 2:
            progress[name] = f"executing {specialty} tasks on '{task}'"
            return f"dives into {specialty} work for '{task}' and narrates intermediate findings."

        progress[name] = f"completed {specialty} contribution"
        environment.state["completed_tasks"].append(task)
        environment.state["active_task"] = None
        return f"wraps up {specialty} duties for '{task}' and signals readiness for handoff."

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Multi-agent debate / negotiation demo
# ---------------------------------------------------------------------------


class DebateEnvironment(TranscriptEnvironment):
    """Environment guiding agents through a structured debate."""

    def __init__(
        self,
        motion: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Debate Chamber", "Negotiation toward a shared statement.", scheduler)
        self.motion = motion
        self.state["stances"]: Dict[str, str] = {}
        self.state["offers"]: List[str] = []
        self.state["consensus"] = "Undecided"

    def _summarise_consensus(self) -> str:
        stances = self.state["stances"]
        if not stances:
            return "No positions recorded yet"

        counts: Dict[str, int] = {}
        for stance in stances.values():
            counts[stance] = counts.get(stance, 0) + 1

        majority = max(counts.values())
        leaders = [stance for stance, value in counts.items() if value == majority]
        if len(leaders) == 1:
            leader = leaders[0]
            if majority == len(stances):
                return f"Full alignment on {leader}"
            if majority >= len(stances) - 1:
                return f"Emerging consensus leaning {leader}"
        breakdown = ", ".join(f"{stance}: {count}" for stance, count in sorted(counts.items()))
        return f"Split viewpoints ({breakdown})"

    def record_position(self, agent: str, stance: str, proposal: str) -> None:
        self.state["stances"][agent] = stance
        self.state["offers"].append(f"{agent}: {proposal}")
        self.state["offers"] = self.state["offers"][-6:]
        self.state["consensus"] = self._summarise_consensus()

    def context(self) -> str:  # type: ignore[override]
        offers = "; ".join(self.state["offers"][-3:]) or "None"
        return (
            f"Debate motion: {self.motion}. "
            f"Consensus trend: {self.state['consensus']}. "
            f"Recent offers: {offers}. "
            f"Latest remarks: {self.recent_dialogue()}"
        )


def make_debater_backend(
    name: str,
    persona: str,
    *,
    stance: str,
    environment: DebateEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        if stance == "support":
            proposal = (
                "amending the charter to rapidly pilot the motion with transparent "
                "evaluation checkpoints"
            )
        elif stance == "oppose":
            proposal = (
                "slowing adoption until risk studies address funding, training, and "
                "long-term governance"
            )
        else:
            proposal = (
                "a compromise timeline combining a small trial with joint oversight "
                "from both camps"
            )

        environment.record_position(name, stance, proposal)
        return (
            f"argues from a {stance} stance on '{environment.motion}', builds on recent "
            f"offers, and proposes {proposal}."
        )

    return make_persona_backend(name, persona, formatter)


def make_moderator_backend(
    name: str,
    persona: str,
    *,
    environment: DebateEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        consensus = environment._summarise_consensus()
        environment.state["consensus"] = consensus
        return (
            f"synthesises the debate motion '{environment.motion}', highlights {consensus}, "
            "and invites actionable next steps."
        )

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Social simulation demo
# ---------------------------------------------------------------------------


class CommunityScenarioEnvironment(TranscriptEnvironment):
    """Environment modelling social coordination dynamics."""

    def __init__(
        self,
        scenario: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Community Hub", "Neighbours co-create solutions.", scheduler)
        self.scenario = scenario
        self.state["shared_resources"]: List[str] = []
        self.state["mood"]: Dict[str, str] = {}

    def record_contribution(self, agent: str, contribution: str, mood: str) -> None:
        self.state["shared_resources"].append(f"{agent}: {contribution}")
        self.state["shared_resources"] = self.state["shared_resources"][-6:]
        self.state["mood"][agent] = mood

    def _mood_summary(self) -> str:
        if not self.state["mood"]:
            return "Balanced"
        counts: Dict[str, int] = {}
        for mood in self.state["mood"].values():
            counts[mood] = counts.get(mood, 0) + 1
        dominant = max(counts, key=counts.get)
        return f"Dominant mood: {dominant}"

    def context(self) -> str:  # type: ignore[override]
        resources = "; ".join(self.state["shared_resources"][-3:]) or "None"
        return (
            f"Scenario: {self.scenario}. "
            f"Mood snapshot: {self._mood_summary()}. "
            f"Shared resources: {resources}. "
            f"Recent coordination: {self.recent_dialogue()}"
        )


def make_social_actor_backend(
    name: str,
    persona: str,
    *,
    focus: str,
    environment: CommunityScenarioEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        if focus == "logistics":
            contribution = "secures venue layout and timeline checkpoints"
            mood = "organised"
        elif focus == "inclusion":
            contribution = "maps accessibility needs and rotating volunteer pairs"
            mood = "caring"
        else:
            contribution = "launches story-sharing circle with musical interludes"
            mood = "energised"

        environment.record_contribution(name, contribution, mood)
        return (
            f"leans on {focus} expertise for '{environment.scenario}', contributes {contribution}, "
            f"and keeps the group feeling {mood}."
        )

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Tactical game AI party demo
# ---------------------------------------------------------------------------


class QuestEncounterEnvironment(TranscriptEnvironment):
    """Environment capturing a tactical party responding to encounters."""

    def __init__(
        self,
        encounters: Iterable[tuple[str, str]],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Quest Encounter", "Cooperative party strategy.", scheduler)
        self.encounters: List[tuple[str, str]] = list(encounters)
        self.state["stage"] = 0
        self.state["team_status"]: Dict[str, str] = {}

    def _current_encounter(self) -> tuple[str, str]:
        index = int(self.state["stage"]) % max(len(self.encounters), 1)
        return self.encounters[index]

    def context(self) -> str:  # type: ignore[override]
        encounter, threat = self._current_encounter()
        status = ", ".join(
            f"{agent}: {state}" for agent, state in sorted(self.state["team_status"].items())
        ) or "Awaiting orders"
        return (
            f"Active encounter: {encounter} (threat {threat}). "
            f"Team status: {status}. "
            f"Recent tactical chatter: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.state["stage"] = (int(self.state["stage"]) + 1) % max(len(self.encounters), 1)
        return message


def make_party_member_backend(
    name: str,
    persona: str,
    *,
    role: str,
    environment: QuestEncounterEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        encounter, threat = environment._current_encounter()
        if role == "tank":
            action = "raises a barrier and calls focus fire"
        elif role == "healer":
            action = "deploys protective wards and triage rotations"
        else:
            action = "channels elemental combos targeting enemy weaknesses"

        environment.state["team_status"][name] = f"{role} covering {encounter}"
        return (
            f"evaluates the {encounter} (threat {threat}), {action}, and synchronises "
            "cooldown timings with the squad."
        )

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Customer service triage demo
# ---------------------------------------------------------------------------


class SupportDeskEnvironment(TranscriptEnvironment):
    """Environment simulating a customer service exchange."""

    def __init__(
        self,
        issue: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Support Desk", "Resolve a high-priority ticket.", scheduler)
        self.issue = issue
        self.state["ticket_status"] = "New"
        self.state["resolution_steps"]: List[str] = []

    def log_step(self, actor: str, update: str, *, status: Optional[str] = None) -> None:
        self.state["resolution_steps"].append(f"{actor}: {update}")
        self.state["resolution_steps"] = self.state["resolution_steps"][-6:]
        if status is not None:
            self.state["ticket_status"] = status

    def context(self) -> str:  # type: ignore[override]
        steps = "; ".join(self.state["resolution_steps"][-3:]) or "None"
        return (
            f"Issue: {self.issue}. "
            f"Ticket status: {self.state['ticket_status']}. "
            f"Key actions: {steps}. "
            f"Recent dialogue: {self.recent_dialogue()}"
        )


def make_customer_backend(
    name: str,
    persona: str,
    *,
    environment: SupportDeskEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        status = environment.state["ticket_status"]
        if status == "New":
            update = "shares diagnostic logs and clarifies business impact"
            environment.log_step(name, update, status="Investigating")
        elif status == "Investigating":
            update = "asks for an estimated resolution timeline and offers remote session access"
            environment.log_step(name, update)
        else:
            update = "confirms the fix resolved the outage and requests preventive guidance"
            environment.log_step(name, update, status="Closed")
        return (
            f"describes the '{environment.issue}' incident, {update}, and keeps the support "
            "team informed."
        )

    return make_persona_backend(name, persona, formatter)


def make_support_agent_backend(
    name: str,
    persona: str,
    *,
    environment: SupportDeskEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        status = environment.state["ticket_status"]
        if status == "New":
            update = "collects error codes and initiates diagnostics"
            environment.log_step(name, update, status="Investigating")
        elif status == "Investigating":
            update = "reproduces the issue in staging and drafts a mitigation plan"
            environment.log_step(name, update, status="Mitigating")
        else:
            update = "confirms patches deployed and prepares customer-ready summary"
            environment.log_step(name, update, status="Monitoring")
        return (
            f"triages the '{environment.issue}' ticket, {update}, and coordinates next actions."
        )

    return make_persona_backend(name, persona, formatter)


def make_support_manager_backend(
    name: str,
    persona: str,
    *,
    environment: SupportDeskEnvironment,
) -> Callable[[str], str]:
    def formatter(_: str) -> str:
        status = environment.state["ticket_status"]
        if status in {"New", "Investigating"}:
            update = "allocates additional engineers and ensures status-page updates"
            environment.log_step(name, update, status="Mitigating")
        elif status == "Mitigating":
            update = "signs off on the mitigation plan and schedules post-mortem review"
            environment.log_step(name, update, status="Monitoring")
        else:
            update = "closes the ticket after verifying stability metrics"
            environment.log_step(name, update, status="Closed")
        return (
            f"oversees the '{environment.issue}' response, {update}, and documents follow-up "
            "actions."
        )

    return make_persona_backend(name, persona, formatter)


# ---------------------------------------------------------------------------
# Scenario assembly
# ---------------------------------------------------------------------------


def run_tool_demo() -> None:
    install_wikipedia_stub()
    scheduler = RoundRobinScheduler()
    environment = ResearchLabEnvironment(topic="Mars habitat design", scheduler=scheduler)
    manager = AgentManager()

    researcher_tool = WikipediaTool(summary_sentences=1)
    researcher = manager.create_agent(
        "transformer",
        name="Atlas",
        llm_backend=make_researcher_backend(
            "Atlas",
            "curious field researcher",
            tool=researcher_tool,
            environment=environment,
        ),
    )
    researcher.set_attribute("role", "Research Lead")
    researcher.register_tool(researcher_tool)

    analyst = manager.create_agent(
        "transformer",
        name="Nova",
        llm_backend=make_analyst_backend(
            "Nova",
            "systems analyst",
            environment=environment,
        ),
    )
    analyst.set_attribute("role", "Data Analyst")

    environment.register_agent(researcher)
    environment.register_agent(analyst)

    run_simulation("Tool-Augmented Research", environment, steps=4)


def run_debate_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = DebateEnvironment(
        motion="Adopt autonomous rovers for frontier habitat logistics",
        scheduler=scheduler,
    )
    manager = AgentManager()

    moderator = manager.create_agent(
        "transformer",
        name="Moderator",
        llm_backend=make_moderator_backend(
            "Moderator",
            "neutral facilitator",
            environment=environment,
        ),
    )
    moderator.set_attribute("role", "Moderator")

    advocate = manager.create_agent(
        "transformer",
        name="Advocate",
        llm_backend=make_debater_backend(
            "Advocate",
            "optimistic strategist",
            stance="support",
            environment=environment,
        ),
    )
    advocate.set_attribute("role", "Debater")

    skeptic = manager.create_agent(
        "transformer",
        name="Skeptic",
        llm_backend=make_debater_backend(
            "Skeptic",
            "risk analyst",
            stance="oppose",
            environment=environment,
        ),
    )
    skeptic.set_attribute("role", "Debater")

    mediator = manager.create_agent(
        "transformer",
        name="Mediator",
        llm_backend=make_debater_backend(
            "Mediator",
            "consensus architect",
            stance="compromise",
            environment=environment,
        ),
    )
    mediator.set_attribute("role", "Debater")

    environment.register_agent(moderator)
    environment.register_agent(advocate)
    environment.register_agent(skeptic)
    environment.register_agent(mediator)

    run_simulation("Multi-Agent Debate", environment, steps=6)


def run_hierarchical_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = MissionControlEnvironment(
        directives=(
            "Scout resource locations",
            "Synthesize habitat risk assessment",
            "Draft daily routine for colonists",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    leader = manager.create_agent(
        "transformer",
        name="Command",
        llm_backend=make_leader_backend(
            "Command",
            "mission lead",
            environment=environment,
        ),
    )
    leader.set_attribute("role", "Leader")

    scout = manager.create_agent(
        "transformer",
        name="Scout",
        llm_backend=make_specialist_backend(
            "Scout",
            "terrain specialist",
            speciality="terrain analysis",
            environment=environment,
        ),
    )
    scout.set_attribute("role", "Follower")

    medic = manager.create_agent(
        "transformer",
        name="Medic",
        llm_backend=make_specialist_backend(
            "Medic",
            "biomedical expert",
            speciality="health planning",
            environment=environment,
        ),
    )
    medic.set_attribute("role", "Follower")

    environment.register_agent(leader)
    environment.register_agent(scout)
    environment.register_agent(medic)

    run_simulation("Hierarchical Mission Planning", environment, steps=5)


def run_social_sim_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = CommunityScenarioEnvironment(
        scenario="Coordinate an inclusive neighbourhood resilience festival",
        scheduler=scheduler,
    )
    manager = AgentManager()

    logistics = manager.create_agent(
        "transformer",
        name="Logistics",
        llm_backend=make_social_actor_backend(
            "Logistics",
            "operations volunteer",
            focus="logistics",
            environment=environment,
        ),
    )
    logistics.set_attribute("role", "Planner")

    inclusion = manager.create_agent(
        "transformer",
        name="Inclusion",
        llm_backend=make_social_actor_backend(
            "Inclusion",
            "community advocate",
            focus="inclusion",
            environment=environment,
        ),
    )
    inclusion.set_attribute("role", "Wellbeing Lead")

    culture = manager.create_agent(
        "transformer",
        name="Culture",
        llm_backend=make_social_actor_backend(
            "Culture",
            "arts facilitator",
            focus="culture",
            environment=environment,
        ),
    )
    culture.set_attribute("role", "Experience Designer")

    environment.register_agent(logistics)
    environment.register_agent(inclusion)
    environment.register_agent(culture)

    run_simulation("Social Dynamics Simulation", environment, steps=6)


def run_npc_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = TavernEnvironment(
        events=(
            "A mysterious traveller shares news of a dragon",
            "A bard starts an upbeat melody",
            "A sudden blackout reveals a hidden map",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    barkeep = manager.create_agent(
        "transformer",
        name="Barkeep",
        llm_backend=make_npc_backend(
            "Barkeep",
            "gruff host",
            environment=environment,
        ),
    )
    bard = manager.create_agent(
        "transformer",
        name="Bard",
        llm_backend=make_npc_backend(
            "Bard",
            "optimistic musician",
            environment=environment,
        ),
    )
    rogue = manager.create_agent(
        "transformer",
        name="Rogue",
        llm_backend=make_npc_backend(
            "Rogue",
            "whispering informant",
            environment=environment,
        ),
    )

    environment.register_agent(barkeep)
    environment.register_agent(bard)
    environment.register_agent(rogue)

    run_simulation("Game NPC Roleplay", environment, steps=6)


def run_game_ai_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = QuestEncounterEnvironment(
        encounters=(
            ("ambush in the crystalline canyon", "high"),
            ("arcane puzzle locking the vault", "medium"),
            ("hydra boss awakening", "critical"),
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    tank = manager.create_agent(
        "transformer",
        name="Vanguard",
        llm_backend=make_party_member_backend(
            "Vanguard",
            "shield-bearing tactician",
            role="tank",
            environment=environment,
        ),
    )
    tank.set_attribute("role", "Frontline")

    healer = manager.create_agent(
        "transformer",
        name="Lumina",
        llm_backend=make_party_member_backend(
            "Lumina",
            "radiant healer",
            role="healer",
            environment=environment,
        ),
    )
    healer.set_attribute("role", "Support")

    mage = manager.create_agent(
        "transformer",
        name="Aeris",
        llm_backend=make_party_member_backend(
            "Aeris",
            "elemental tactician",
            role="caster",
            environment=environment,
        ),
    )
    mage.set_attribute("role", "Damage")

    environment.register_agent(tank)
    environment.register_agent(healer)
    environment.register_agent(mage)

    run_simulation("Tactical Party Coordination", environment, steps=6)


def run_productivity_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = ProductivityEnvironment(
        backlog=(
            "Design solar panel layout",
            "Draft emergency response checklist",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    orchestrator = manager.create_agent(
        "transformer",
        name="Pilot",
        llm_backend=make_orchestrator_backend(
            "Pilot",
            "swarm orchestrator",
            environment=environment,
        ),
    )
    orchestrator.set_attribute("role", "Coordinator")

    math_tool = MathTool()
    engineer = manager.create_agent(
        "transformer",
        name="Engineer",
        llm_backend=make_worker_backend(
            "Engineer",
            "systems engineer",
            specialty="systems integration",
            environment=environment,
            tool=math_tool,
        ),
    )
    engineer.set_attribute("role", "Maker")
    engineer.register_tool(math_tool)

    designer = manager.create_agent(
        "transformer",
        name="Designer",
        llm_backend=make_worker_backend(
            "Designer",
            "ux specialist",
            specialty="human-centred design",
            environment=environment,
        ),
    )
    designer.set_attribute("role", "Maker")

    environment.register_agent(orchestrator)
    environment.register_agent(engineer)
    environment.register_agent(designer)

    run_simulation("Productivity Swarm", environment, steps=6)


def run_support_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = SupportDeskEnvironment(
        issue="Payment gateway outage affecting checkouts",
        scheduler=scheduler,
    )
    manager = AgentManager()

    customer = manager.create_agent(
        "transformer",
        name="Customer",
        llm_backend=make_customer_backend(
            "Customer",
            "startup founder",
            environment=environment,
        ),
    )
    customer.set_attribute("role", "Reporter")

    agent = manager.create_agent(
        "transformer",
        name="Agent",
        llm_backend=make_support_agent_backend(
            "Agent",
            "incident responder",
            environment=environment,
        ),
    )
    agent.set_attribute("role", "Responder")

    manager_agent = manager.create_agent(
        "transformer",
        name="DutyManager",
        llm_backend=make_support_manager_backend(
            "DutyManager",
            "support lead",
            environment=environment,
        ),
    )
    manager_agent.set_attribute("role", "Supervisor")

    environment.register_agent(customer)
    environment.register_agent(agent)
    environment.register_agent(manager_agent)

    run_simulation("Customer Support Triage", environment, steps=6)


def main() -> None:
    run_tool_demo()
    run_hierarchical_demo()
    run_debate_demo()
    run_social_sim_demo()
    run_npc_demo()
    run_game_ai_demo()
    run_productivity_demo()
    run_support_demo()


if __name__ == "__main__":
    main()
