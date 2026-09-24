"""Tool whose module import is deliberately slow (isolated-worker test helper).

Process-isolated workers re-import the tool's module in the child, so this
module simulates a heavyweight environment in which child boot (for example an
optional stack such as torch) takes seconds. The import delay must stay larger
than the execution timeout used by the regression test that consumes it.
"""

from __future__ import annotations

import time

from neva.agents.base import Tool

IMPORT_DELAY_SECONDS = 0.75

time.sleep(IMPORT_DELAY_SECONDS)


class SlowStartupTool(Tool):
    """Trivial tool that can only run after the slow module import."""

    def __init__(self) -> None:
        super().__init__("slow-startup", "Module import is intentionally slow")

    def use(self, task: str) -> str:
        return f"ok:{task}"
