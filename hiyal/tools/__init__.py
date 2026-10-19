"""Tool implementations used by Hiyal agents."""

from __future__ import annotations

from hiyal.agents.base import Tool

from .guard import ToolGuard, ToolLimits
from .loop import ToolLoopConfig, ToolLoopResult, ToolLoopStep, run_tool_loop
from .math import MathTool
from .schemas import ArgumentSchema, ArgumentSpec
from .summarizer import SummarizerTool
from .translator import TranslatorTool
from .wikipedia import WikipediaTool

__all__ = [
    "ArgumentSchema",
    "ArgumentSpec",
    "MathTool",
    "SummarizerTool",
    "Tool",
    "ToolGuard",
    "ToolLimits",
    "ToolLoopConfig",
    "ToolLoopResult",
    "ToolLoopStep",
    "TranslatorTool",
    "run_tool_loop",
    "WikipediaTool",
]
