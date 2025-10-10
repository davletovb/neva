"""Agent convenience exports.

The project historically defined agent implementations in both ``agents.py`` and
``models.py`` which led to divergent behaviours.  The canonical implementations
now live in :mod:`models`; this module simply re-exports them to maintain the
public API.
"""

from models import AIAgent, GPTAgent, TransformerAgent

__all__ = ["AIAgent", "GPTAgent", "TransformerAgent"]
