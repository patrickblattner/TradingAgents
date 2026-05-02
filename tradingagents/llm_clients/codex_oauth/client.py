"""LangChain ChatOpenAI subclass that talks to chatgpt.com/backend-api/codex.

What's different from a regular ChatOpenAI:

1. **Endpoint**: ``https://chatgpt.com/backend-api/codex`` instead of api.openai.com.
   Combined with ``use_responses_api=True``, the SDK posts to ``/codex/responses``.

2. **Auth**: bearer token from the Codex OAuth flow (`auth.get_access_token`),
   not a static API key. Tokens are refreshed automatically when within
   ``REFRESH_SKEW_SECONDS`` of expiry.

3. **Cloudflare-required headers**: ``originator: codex_cli_rs``, a
   codex_cli_rs-shaped User-Agent, and ``ChatGPT-Account-ID`` extracted from
   the JWT. Without these, the endpoint returns 403 regardless of auth.

4. **Request shape**: the Codex backend rejects ``max_output_tokens`` and
   requires ``store: false``. We strip the former and force the latter in
   ``_get_request_payload``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..openai_client import NormalizedChatOpenAI
from . import auth as codex_auth

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Cloudflare's allow-list whitelists this exact value. Don't change it.
_ORIGINATOR = "codex_cli_rs"
_USER_AGENT = "codex_cli_rs/0.0.0 (TradingAgents)"

# The Codex backend rejects requests without `instructions` (HTTP 400
# "Instructions are required") — the public Responses API treats it as
# optional, but chatgpt.com/backend-api/codex does not. When LangChain doesn't
# emit a system message, we synthesise a generic one so the request goes
# through. Real TradingAgents agents always supply their own system prompt,
# which then takes precedence.
_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


def _extract_text_from_content(content: Any) -> str:
    """Pull plain text out of a Responses API content field (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # Responses API uses {"type": "input_text"|"output_text", "text": "..."}
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _hoist_system_messages(input_field: Any) -> tuple[str, Any]:
    """Return (hoisted_text, input_without_system_items).

    Walks the Responses API ``input`` field, extracts any items with role
    ``system`` or ``developer``, joins their text into a single string, and
    returns the input list with those items removed. Returns the input
    unchanged when there's nothing to hoist.
    """
    if not isinstance(input_field, list):
        return "", input_field
    hoisted: list[str] = []
    kept: list[Any] = []
    for item in input_field:
        if isinstance(item, dict) and item.get("role") in ("system", "developer"):
            text = _extract_text_from_content(item.get("content"))
            if text.strip():
                hoisted.append(text.strip())
            continue
        kept.append(item)
    return ("\n\n".join(hoisted), kept)


def codex_default_headers(access_token: str) -> dict:
    """Build the header set required by chatgpt.com/backend-api/codex."""
    headers = {
        "User-Agent": _USER_AGENT,
        "originator": _ORIGINATOR,
    }
    account_id = codex_auth.extract_account_id(access_token)
    if account_id:
        # PascalCase is canonical (matches codex-rs auth.rs); other casings
        # are not honored by the backend.
        headers["ChatGPT-Account-ID"] = account_id
    return headers


class CodexChatOpenAI(NormalizedChatOpenAI):
    """ChatOpenAI bound to the ChatGPT subscription Codex backend.

    Construction loads (and refreshes if needed) the OAuth access token, then
    configures the underlying SDK with the bearer token and Cloudflare-friendly
    headers. Token expiry within a single client's lifetime is not yet handled
    here — the fallback wrapper retries on 401 by rebuilding the client.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Codex backend rejects max_output_tokens; strip it if upstream LangChain
        # populated it from max_tokens config.
        payload.pop("max_output_tokens", None)
        # store=true is rejected by the ChatGPT backend (returns 400). Force false.
        payload["store"] = False
        # Codex rejects system/developer messages inside `input` ("System messages
        # are not allowed"); the system prompt must move to the `instructions`
        # field. langchain-openai's Responses adapter doesn't hoist this for us
        # in every shape, so we do it defensively.
        existing_instructions = str(payload.get("instructions") or "").strip()
        hoisted, filtered_input = _hoist_system_messages(payload.get("input"))
        if hoisted:
            existing_instructions = (
                hoisted if not existing_instructions
                else f"{existing_instructions}\n\n{hoisted}"
            )
            payload["input"] = filtered_input
        if not existing_instructions:
            existing_instructions = _DEFAULT_INSTRUCTIONS
        payload["instructions"] = existing_instructions
        return payload


def build_codex_chat_model(model: str, **extra_kwargs: Any) -> CodexChatOpenAI:
    """Construct a CodexChatOpenAI for the given model name.

    Raises CodexAuthRequired when no usable tokens are available so the caller
    can prompt for `tradingagents codex-login`.
    """
    access_token = codex_auth.get_access_token()
    init_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": access_token,
        "base_url": CODEX_BASE_URL,
        "default_headers": codex_default_headers(access_token),
        "use_responses_api": True,
        # Codex backend rejects non-streaming requests ("Stream must be set to true").
        # LangChain's _generate falls through to _stream when this is True and
        # collects chunks into a normal ChatResult, so callers see the same
        # AIMessage interface.
        "streaming": True,
    }
    init_kwargs.update(extra_kwargs)
    return CodexChatOpenAI(**init_kwargs)
