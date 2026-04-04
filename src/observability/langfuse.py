"""Langfuse integration for LangChain/LangGraph.

This module is deliberately defensive:
- If Langfuse isn't installed, it becomes a no-op.
- If Langfuse isn't enabled via config, it becomes a no-op.

Langfuse's CallbackHandler reads configuration primarily from env vars.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from src.config import config

logger = logging.getLogger(__name__)

_LANGFUSE_HANDLER: Any | None = None


def _ensure_env() -> None:
    if config.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.langfuse_public_key)
    if config.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", config.langfuse_secret_key)
    if config.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", config.langfuse_host)


def get_langfuse_callbacks() -> list[Any]:
    """Return Langfuse callback handler(s) for LangChain, if enabled."""

    global _LANGFUSE_HANDLER

    if not getattr(config, "langfuse_enabled", False):
        return []

    try:
        from langfuse.langchain import CallbackHandler
    except Exception as e:
        logger.warning(f"Langfuse enabled but could not import langfuse.langchain.CallbackHandler: {e}")
        return []

    if _LANGFUSE_HANDLER is None:
        _ensure_env()
        try:
            # CallbackHandler reads env vars; we still pass public_key for clarity.
            _LANGFUSE_HANDLER = CallbackHandler(public_key=config.langfuse_public_key or None)
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse CallbackHandler: {e}")
            _LANGFUSE_HANDLER = None
            return []

    return [_LANGFUSE_HANDLER]
