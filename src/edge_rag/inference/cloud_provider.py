from __future__ import annotations

import time

from openai import OpenAI

from edge_rag.config import Settings
from edge_rag.inference.base import InferenceClient
from edge_rag.types import GenerationResult


class CloudProvider(InferenceClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required when INFERENCE_PROVIDER=nim")
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nim_base_url,
        )

    @property
    def provider_name(self) -> str:
        return "nim"

    @property
    def model_name(self) -> str:
        return self._settings.nim_model

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        start = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        usage_output_tokens = 0

        stream = self._client.chat.completions.create(
            model=self._settings.nim_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if getattr(chunk, "usage", None):
                output_tokens = getattr(chunk.usage, "output_tokens", None)
                completion_tokens = getattr(chunk.usage, "completion_tokens", None)
                token_value = output_tokens if output_tokens is not None else completion_tokens
                if token_value is not None:
                    usage_output_tokens = int(token_value)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if delta:
                output_parts.append(delta)
                if first_token_at is None:
                    first_token_at = time.perf_counter()

        end = time.perf_counter()
        output_text = "".join(output_parts).strip()
        ttft = (first_token_at - start) if first_token_at is not None else 0.0
        generation_seconds = end - start

        return GenerationResult(
            text=output_text,
            ttft_seconds=max(ttft, 0.0),
            generation_seconds=max(generation_seconds, 0.0),
            output_tokens=usage_output_tokens if usage_output_tokens > 0 else len(output_text.split()),
            raw_response={"base_url": self._settings.nim_base_url, "model": self._settings.nim_model},
        )
