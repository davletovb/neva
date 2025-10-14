"""Run all advanced showcase examples sequentially."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.customer_support_demo import run_demo as run_support_demo
from examples.debate_demo import run_demo as run_debate_demo
from examples.hierarchical_mission_demo import run_demo as run_hierarchical_demo
from examples.productivity_swarm_demo import run_demo as run_productivity_demo
from examples.quest_party_demo import run_demo as run_game_ai_demo
from examples.social_scenario_demo import run_demo as run_social_sim_demo
from examples.tavern_npc_demo import run_demo as run_npc_demo
from examples.tool_research_demo import run_demo as run_tool_demo


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
