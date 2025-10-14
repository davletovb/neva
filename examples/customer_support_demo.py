"""Customer support triage and escalation example."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation


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
    def formatter(prompt: str) -> str:
        del prompt
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
    def formatter(prompt: str) -> str:
        del prompt
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
    def formatter(prompt: str) -> str:
        del prompt
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


def run_demo() -> None:
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
    run_demo()


if __name__ == "__main__":
    main()
