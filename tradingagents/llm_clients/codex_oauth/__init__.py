"""Codex OAuth subscription auth for TradingAgents.

Adds an OpenAI provider variant that authenticates via the user's ChatGPT
Plus/Pro/Business subscription (Codex CLI OAuth) and falls back transparently
to a regular OpenAI API key when the subscription quota is exhausted.

Public surface:
- ``auth.login`` / ``auth.get_access_token`` — OAuth flow + token retrieval
- ``client.CodexChatOpenAI`` — LangChain ChatModel pointed at chatgpt.com/backend-api/codex
- ``fallback.CodexWithFallbackClient`` — wraps Codex + API key with auto-fallback
"""

from .auth import (
    CodexAuthError,
    CodexAuthRequired,
    extract_account_id,
    get_access_token,
    load_tokens,
    login,
    refresh_tokens,
    save_tokens,
)

__all__ = [
    "CodexAuthError",
    "CodexAuthRequired",
    "extract_account_id",
    "get_access_token",
    "load_tokens",
    "login",
    "refresh_tokens",
    "save_tokens",
]
