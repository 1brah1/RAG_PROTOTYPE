from __future__ import annotations

import json
import time

import requests

from edge_rag.config import Settings
from edge_rag.inference.base import InferenceClient
from edge_rag.types import GenerationResult


class LocalProvider(InferenceClient):
    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        self._settings = settings
        self._model_name = model_name or settings.local_model_q4

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        endpoint = f"{self._settings.local_ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self._model_name,
            "system": system_prompt,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        eval_count = 0

        with requests.post(endpoint, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                token_text = data.get("response", "")
                if token_text:
                    output_parts.append(token_text)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                if data.get("done"):
                    eval_count = int(data.get("eval_count") or 0)

        end = time.perf_counter()
        output_text = "".join(output_parts).strip()
        ttft = (first_token_at - start) if first_token_at is not None else 0.0
        generation_seconds = end - start

        return GenerationResult(
            text=output_text,
            ttft_seconds=max(ttft, 0.0),
            generation_seconds=max(generation_seconds, 0.0),
            output_tokens=eval_count if eval_count > 0 else len(output_text.split()),
            raw_response={"endpoint": endpoint, "model": self._model_name},
        )
