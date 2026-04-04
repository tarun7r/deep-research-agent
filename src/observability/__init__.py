"""Observability integrations (optional).

Currently supported:
- Langfuse via its LangChain callback handler

All integrations are optional and should never break default local runs.
"""

from __future__ import annotations

from typing import Any, Optional

from .langfuse import get_langfuse_callbacks


def runnable_config(*, tags: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Build a LangChain Runnable config dict with optional callbacks.

    Returns None when no config is needed.
    """

    callbacks = get_langfuse_callbacks()

    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks
    if tags:
        config["tags"] = tags
    if metadata:
        config["metadata"] = metadata

    return config or None
