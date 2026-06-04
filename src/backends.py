"""
Thin abstraction over generation backends.

Supports:
  - vllm_local: offline vLLM LLM instance (for both generation and judging)
  - openai_api: OpenAI-compatible API endpoint (for adding GPT-4, etc. to the committee)
"""

from __future__ import annotations

import os
import re
from typing import Any

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# vLLM local backend
# ---------------------------------------------------------------------------

class VLLMLocalBackend:
    """Wraps a vLLM LLM instance for both raw generation and chat."""

    _instances: dict[str, LLM] = {}

    def __init__(self, model: str, tensor_parallel_size: int = 1, **llm_kwargs: Any):
        self.model = model
        if model not in VLLMLocalBackend._instances:
            VLLMLocalBackend._instances[model] = LLM(
                model=model, tensor_parallel_size=tensor_parallel_size, **llm_kwargs
            )
        self.llm = VLLMLocalBackend._instances[model]
        self.tokenizer = self.llm.get_tokenizer()

    @property
    def supports_raw_generation(self) -> bool:
        return True

    def generate_raw(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[str]:
        """Raw text completion (used for instruction generation).
        vLLM handles batching internally — pass all prompts at once."""
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def chat(
        self,
        conversations: list[list[dict[str, str]]],
        sampling_params: SamplingParams,
    ) -> list[list[str]]:
        """Chat completion over a batch of conversations.
        vLLM handles batching internally — pass all conversations at once.

        Returns list of lists: outer = per conversation, inner = n completions.
        When sampling_params.n == 1, inner list has one element.
        """
        outputs = self.llm.chat(conversations, sampling_params)
        return [[o.text for o in out.outputs] for out in outputs]


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend
# ---------------------------------------------------------------------------

class OpenAIAPIBackend:
    """Wraps an OpenAI-compatible API for chat completion.

    Note: this backend has no tokenizer and does not support generate_raw().
    It cannot be used as the primary backend for instruction generation or
    multi-turn follow-up generation (which require raw text completion).
    """

    tokenizer = None  # #9: explicit attribute so hasattr checks work

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        )

    @property
    def supports_raw_generation(self) -> bool:
        return False

    def generate_raw(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        raise NotImplementedError(
            f"OpenAI API backend ({self.model}) does not support raw text generation. "
            f"Use a vllm_local backend for instruction/follow-up generation."
        )

    def chat(
        self,
        conversations: list[list[dict[str, str]]],
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Returns list of lists to match VLLMLocalBackend interface."""
        results: list[list[str]] = []
        extra: dict[str, Any] = {}
        n = 1
        if sampling_params is not None:
            extra["temperature"] = sampling_params.temperature
            extra["top_p"] = sampling_params.top_p
            if sampling_params.max_tokens and sampling_params.max_tokens < 1_000_000:
                extra["max_tokens"] = sampling_params.max_tokens
            n = sampling_params.n if sampling_params.n else 1
        extra.update(kwargs)

        for conv in conversations:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=conv,
                n=n,
                **extra,
            )
            results.append([c.message.content or "" for c in resp.choices])
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(cfg: dict[str, Any], tensor_parallel_size: int = 1, **llm_kwargs: Any) -> VLLMLocalBackend | OpenAIAPIBackend:
    backend_type = cfg["type"]
    model = cfg["model"]
    if backend_type == "vllm_local":
        return VLLMLocalBackend(model, tensor_parallel_size=tensor_parallel_size, **llm_kwargs)
    elif backend_type == "openai_api":
        return OpenAIAPIBackend(
            model,
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def parse_judge_score(text: str) -> tuple[float | None, str | None]:
    """Extract the numeric score from a judge response ending with 'Score: N'.

    Returns (score, error_reason). On success error_reason is None.
    On failure, score is None and error_reason explains what went wrong.
    """
    if not text or not text.strip():
        return None, "empty judge response"

    match = re.search(r"Score:\s*(\d(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 5.0:
            return score, None
        return None, f"score {score} out of range [1,5]"

    # Try fallback: look for a bare digit 1-5 at the very end
    end_match = re.search(r"(\d)\s*$", text.strip())
    if end_match:
        score = float(end_match.group(1))
        if 1.0 <= score <= 5.0:
            return score, None

    tail = text.strip()[-200:] if len(text.strip()) > 200 else text.strip()
    return None, f"no 'Score: N' found. Tail: {repr(tail)}"
