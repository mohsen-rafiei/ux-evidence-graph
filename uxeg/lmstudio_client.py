"""Thin wrapper around the OpenAI client pointed at a local LM Studio server.

LM Studio exposes an OpenAI-compatible API. We use the official ``openai``
Python client and simply change the ``base_url`` and ``api_key``. No cloud
calls are made; everything stays on the user's machine.
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .config import Config
from .utils import error, info, success


class LMStudioError(RuntimeError):
    """Raised when we cannot talk to LM Studio in a useful way."""


class LMStudioClient:
    """Wrapper for chat completions and embeddings against LM Studio."""

    def __init__(self, config: Config):
        self.config = config
        self.lm = config.lmstudio
        # The OpenAI client just needs a base_url + key; LM Studio ignores the
        # key value but the client requires a non-empty string.
        self.client = OpenAI(base_url=self.lm.base_url, api_key=self.lm.api_key or "lm-studio")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health_check(self, verbose: bool = True) -> bool:
        """Try to reach LM Studio. Returns True if reachable.

        First tries to list models; if that is not supported, falls back to a
        tiny chat completion.
        """

        try:
            models = self.client.models.list()
            names = [m.id for m in getattr(models, "data", [])]
            if verbose:
                success(f"LM Studio reachable at {self.lm.base_url}")
                if names:
                    info(f"Loaded models: {', '.join(names)}")
                else:
                    info("LM Studio responded but reported no loaded models.")
            return True
        except Exception as exc:  # noqa: BLE001 - we want a friendly message
            if verbose:
                self._print_connection_help(exc)
            return False

    def _print_connection_help(self, exc: Exception) -> None:
        error(f"Could not reach LM Studio at {self.lm.base_url}")
        info("Checklist:")
        info("  1. Open LM Studio and load a chat model.")
        info("  2. Go to the 'Local Server' (Developer) tab and click 'Start Server'.")
        info("  3. Confirm the server URL matches your config (default http://localhost:1234/v1).")
        info("  4. Set LMSTUDIO_CHAT_MODEL in your .env to a loaded model id.")
        info(f"Underlying error: {exc}")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run a single chat completion and return the text content."""

        if not self.lm.chat_model or self.lm.chat_model.startswith("change-this"):
            raise LMStudioError(
                "No chat model configured. Set LMSTUDIO_CHAT_MODEL in .env "
                "(or chat_model in config.yaml) to a model loaded in LM Studio."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.lm.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LMStudioError(
                f"Chat completion failed. Is LM Studio running with model "
                f"'{self.lm.chat_model}' loaded? Underlying error: {exc}"
            ) from exc

        content = response.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from the LM Studio embeddings endpoint."""

        if not self.lm.embedding_model or self.lm.embedding_model.startswith("change-this"):
            raise LMStudioError(
                "No embedding model configured for LM Studio. Set "
                "LMSTUDIO_EMBEDDING_MODEL or switch EMBEDDING_PROVIDER to "
                "sentence_transformers."
            )

        try:
            response = self.client.embeddings.create(model=self.lm.embedding_model, input=texts)
        except Exception as exc:  # noqa: BLE001
            raise LMStudioError(
                f"Embedding request failed for model '{self.lm.embedding_model}'. "
                f"Underlying error: {exc}"
            ) from exc

        return [item.embedding for item in response.data]
