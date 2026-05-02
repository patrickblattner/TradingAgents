"""BaseLLMClient adapter that registers the Codex+fallback model with the factory."""

from __future__ import annotations

from typing import Any, Optional

from ..base_client import BaseLLMClient
from ..validators import validate_model
from .fallback import build_codex_with_fallback


class CodexOAuthClient(BaseLLMClient):
    """Provider entry for ``llm_provider: "openai_codex"``.

    Routes to ChatGPT Plus/Pro subscription (Codex OAuth) for inference and
    falls back to the OpenAI API key on quota exhaustion. Models are validated
    against the OpenAI catalog since the Codex backend's allow-list is a
    documented subset of the public API model list.
    """

    provider = "openai_codex"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # base_url is accepted for interface parity but ignored — the Codex
        # endpoint URL is fixed (chatgpt.com/backend-api/codex), and a
        # user-supplied base_url here would route through a non-Codex backend
        # while we send Codex-specific headers, which 403s.
        super().__init__(model, base_url=None, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        api_key = self.kwargs.get("api_key")
        fallback_model = self.kwargs.get("fallback_model")

        # Forward only kwargs that ChatOpenAI accepts on both legs. Drop
        # provider-internal config keys.
        forwarded = {
            k: v for k, v in self.kwargs.items()
            if k in {"timeout", "max_retries", "reasoning_effort", "callbacks"}
        }
        return build_codex_with_fallback(
            self.model,
            api_key=api_key,
            fallback_model=fallback_model,
            **forwarded,
        )

    def validate_model(self) -> bool:
        # Codex backend accepts the same model namespace as native OpenAI.
        return validate_model("openai", self.model)
