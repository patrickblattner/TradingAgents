"""Codex-primary, API-key-fallback chat model.

Wraps a CodexChatOpenAI (ChatGPT subscription) and a regular NormalizedChatOpenAI
(API key) so the user's ChatGPT credits get consumed first; on 429/quota-style
errors we transparently fall back to the API path.

Why a custom wrapper instead of LangChain's ``with_fallbacks``: TradingAgents
calls ``with_structured_output(schema)`` and ``bind_tools(tools)`` on the model
to get structured-output and tool-using variants. Those return
``Runnable``-typed objects, and ``Runnable.with_fallbacks`` doesn't preserve the
``BaseChatModel`` surface that callers further upstream rely on. The wrapper
applies the same transformation to both legs and re-wraps so the fallback
follows along.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from langchain_core.runnables import Runnable

from ..openai_client import NormalizedChatOpenAI
from . import auth as codex_auth
from .client import CodexChatOpenAI, build_codex_chat_model

logger = logging.getLogger(__name__)

# Errors that trigger fallback to API key (quota / rate limit). Other OpenAI
# errors (auth, malformed request) propagate so the user sees the real cause.
_QUOTA_EXCEPTION_TYPES: tuple = ()


def _resolve_quota_exception_types() -> tuple:
    """Lazy-import OpenAI SDK exception classes so import-time failures don't break
    callers that never use this client."""
    global _QUOTA_EXCEPTION_TYPES
    if _QUOTA_EXCEPTION_TYPES:
        return _QUOTA_EXCEPTION_TYPES
    classes: list[type] = []
    try:
        from openai import APIStatusError, RateLimitError  # type: ignore

        classes.append(RateLimitError)
        # 402 (quota purchased run out) shows up as APIStatusError; we filter on
        # status_code below.
        classes.append(APIStatusError)
    except ImportError:
        pass
    _QUOTA_EXCEPTION_TYPES = tuple(classes)
    return _QUOTA_EXCEPTION_TYPES


def _is_quota_error(exc: BaseException) -> bool:
    """Detect quota-exhaustion vs other API errors.

    True for: 429 (rate limit), 402 (insufficient quota / payment required).
    False for: 401 (auth), 403 (forbidden — usually missing originator header,
    not a quota issue), 4xx/5xx with other shapes.
    """
    types = _resolve_quota_exception_types()
    if not types or not isinstance(exc, types):
        return False
    status = getattr(exc, "status_code", None)
    if status in (402, 429):
        return True
    # Some quota errors come back as 400 with a quota-style code in the body.
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code") or body.get("code")
        if isinstance(code, str) and code in {
            "insufficient_quota", "quota_exceeded", "rate_limit_exceeded",
        }:
            return True
    return False


class CodexWithFallbackChatModel:
    """Chat-model-like wrapper: try Codex first, fall back to API on quota errors.

    Implements only the surface TradingAgents agents actually call:
    ``invoke``, ``with_structured_output``, ``bind_tools``. Both legs are
    transformed in lockstep so each downstream variant inherits the fallback.

    Not a true ``BaseChatModel`` subclass — LangChain's chat-model class
    enforces a Pydantic schema we don't need here, and the surface required by
    TradingAgents is small enough that subclassing would buy nothing but
    boilerplate.
    """

    def __init__(self, primary: Any, fallback: Any):
        self._primary = primary
        self._fallback = fallback

    def invoke(self, input_, config=None, **kwargs):
        try:
            return self._primary.invoke(input_, config, **kwargs)
        except Exception as exc:  # noqa: BLE001 — we re-raise non-matching
            if not _is_quota_error(exc):
                raise
            logger.warning(
                "Codex subscription quota exhausted (%s). Falling back to OpenAI API key.",
                type(exc).__name__,
            )
            return self._fallback.invoke(input_, config, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return CodexWithFallbackChatModel(
            primary=self._primary.with_structured_output(schema, **kwargs),
            fallback=self._fallback.with_structured_output(schema, **kwargs),
        )

    def bind_tools(self, tools, **kwargs):
        return CodexWithFallbackChatModel(
            primary=self._primary.bind_tools(tools, **kwargs),
            fallback=self._fallback.bind_tools(tools, **kwargs),
        )

    # Pass-through helpers for occasional callers that introspect the model.
    def __getattr__(self, name):
        # Only consulted when normal attribute lookup fails. Delegate to primary
        # for everything we haven't overridden — most callers care about
        # ``model_name``, ``temperature``, etc., which are identical between
        # the two legs.
        return getattr(self._primary, name)


def build_codex_with_fallback(
    model: str,
    *,
    api_key: Optional[str] = None,
    fallback_model: Optional[str] = None,
    **extra_kwargs: Any,
) -> Any:
    """Construct the Codex+API fallback model.

    When an OpenAI API key is available (``OPENAI_API_KEY`` env or ``api_key``
    arg), returns a CodexWithFallbackChatModel wrapping both legs. When no
    API key is available, returns the bare CodexChatOpenAI and logs a warning
    that quota errors will propagate instead of falling back. This keeps users
    who only have a ChatGPT subscription unblocked, at the cost of resilience.

    Raises CodexAuthRequired if the user has not run ``tradingagents codex-login``.
    """
    primary = build_codex_chat_model(model, **extra_kwargs)

    api_key_resolved = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key_resolved:
        logger.warning(
            "openai_codex: no OPENAI_API_KEY in environment — running Codex-only "
            "without fallback. Quota exhaustion will propagate as an error. "
            "Set OPENAI_API_KEY to enable transparent API fallback."
        )
        return primary

    fallback = NormalizedChatOpenAI(
        model=fallback_model or model,
        api_key=api_key_resolved,
        use_responses_api=True,
        **extra_kwargs,
    )
    return CodexWithFallbackChatModel(primary=primary, fallback=fallback)
