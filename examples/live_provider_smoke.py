"""One-turn OpenAI example: scripted by default, explicitly opt in to a live call.

Run ``python examples/live_provider_smoke.py`` without credentials or network.
For a generated response, set OPENAI_API_KEY and pass --live --max-spend-usd.
The spend ceiling uses Neva's estimated token costs, not provider billing.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neva.agents import GPTAgent
from neva.utils.exceptions import SpendBudgetExceededError
from neva.utils.metrics import CostTracker, SpendBudget, TokenUsageTracker

MODEL = "gpt-4o-mini"
PROMPT = "Suggest one practical way two agents could collaborate on a research task."
MAX_OUTPUT_TOKENS = 128
MAX_CONTEXT_CHARS = 2000
REQUEST_TIMEOUT_SECONDS = 15.0


def positive_usd(value: str) -> float:
    """Parse a strictly positive, finite USD limit for argparse."""

    try:
        amount = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite USD amount") from exc
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite USD amount")
    return amount


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="make one billed OpenAI API request")
    parser.add_argument(
        "--max-spend-usd",
        type=positive_usd,
        help="required with --live; ceiling on Neva's estimated spend (USD)",
    )
    args = parser.parse_args(argv)

    if not args.live:
        if args.max_spend_usd is not None:
            parser.error("--max-spend-usd requires --live")
        agent = GPTAgent(
            name="Research partner",
            llm_backend=lambda prompt: "Split the task into source finding and source checking.",
        )
        print("[SCRIPTED] Offline stub response; no provider call or charge:")
        print(agent.receive(PROMPT))
        return 0

    if args.max_spend_usd is None:
        parser.error("--live requires --max-spend-usd")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("--live requires a nonempty OPENAI_API_KEY environment variable")

    cost = CostTracker()
    usage = TokenUsageTracker()
    budget = SpendBudget(args.max_spend_usd)
    agent = GPTAgent(
        name="Research partner",
        provider="openai",
        model=MODEL,
        api_key=api_key,
        cost_tracker=cost,
        token_tracker=usage,
        spend_budget=budget,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_context_chars=MAX_CONTEXT_CHARS,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    prices = cost.pricing_per_1k_tokens[MODEL]
    print(
        f"[GENERATED] One live {MODEL} request; up to {MAX_OUTPUT_TOKENS} output tokens, "
        f"{REQUEST_TIMEOUT_SECONDS:g}s timeout, no retries."
    )
    print(
        f"Neva's static estimate: ${prices['input']:.6f}/1k input tokens, "
        f"${prices['output']:.6f}/1k output tokens; "
        f"estimated spend ceiling ${args.max_spend_usd:.6f}."
    )
    print("Provider billing can differ from these estimates; check current provider pricing.")
    try:
        response = agent.receive(PROMPT)
    except SpendBudgetExceededError:
        parser.error("estimated prompt + maximum output cost exceeds --max-spend-usd")
    print(response)
    print(
        f"Estimated spend for this run: ${budget.spent:.6f} "
        f"({'provider usage' if usage.estimated_calls == 0 else 'estimated token counts'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
