from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    page: int
    score: float | None = None


@dataclass
class GenerationResult:
    text: str
    ttft_seconds: float
    generation_seconds: float
    output_tokens: int
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUSnapshot:
    timestamp_utc: str
    memory_used_mb: float | None
    utilization_gpu_percent: float | None
    available: bool
    note: str = ""


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    provider: str
    model: str
    retrieval_seconds: float
    generation_seconds: float
    ttft_seconds: float
    output_tokens: int
    tokens_per_second: float
    telemetry_pre: GPUSnapshot | None
    telemetry_post: GPUSnapshot | None
