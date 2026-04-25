from __future__ import annotations

import time

from edge_rag.config import Settings
from edge_rag.indexing import ChromaIndexer
from edge_rag.inference.base import InferenceClient
from edge_rag.telemetry import capture_generation_telemetry
from edge_rag.types import RAGResult, RetrievedChunk

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. "
    "Use only the provided context to answer. "
    "If the context is insufficient, reply with exactly: I do not know."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{idx}] filename={chunk.filename} page={chunk.page} chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "filename": chunk.filename,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
    ]


def _format_sources_section(sources: list[dict[str, object]]) -> str:
    if not sources:
        return "Sources:\n- none"
    lines = ["Sources:"]
    for source in sources:
        lines.append(
            f"- filename={source['filename']}, page={source['page']}, chunk_id={source['chunk_id']}"
        )
    return "\n".join(lines)


class RAGOrchestrator:
    def __init__(
        self,
        settings: Settings,
        indexer: ChromaIndexer,
        inference_client: InferenceClient,
    ) -> None:
        self.settings = settings
        self.indexer = indexer
        self.inference_client = inference_client

    def ask(self, query: str) -> RAGResult:
        retrieval_start = time.perf_counter()
        chunks = self.indexer.similarity_search(query, top_k=self.settings.top_k)
        retrieval_seconds = time.perf_counter() - retrieval_start

        sources = _build_sources(chunks)
        if not chunks:
            fallback = "I do not know.\n\nSources:\n- none"
            return RAGResult(
                query=query,
                answer=fallback,
                sources=[],
                provider=self.inference_client.provider_name,
                model=self.inference_client.model_name,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=0.0,
                ttft_seconds=0.0,
                output_tokens=0,
                tokens_per_second=0.0,
                telemetry_pre=None,
                telemetry_post=None,
            )

        user_prompt = (
            "Question:\n"
            f"{query}\n\n"
            "Context:\n"
            f"{_build_context(chunks)}\n\n"
            "Answer using only the context."
        )

        with capture_generation_telemetry(self.settings.telemetry_enabled) as telemetry:
            generation = self.inference_client.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

        answer_body = generation.text.strip() or "I do not know."
        answer = f"{answer_body}\n\n{_format_sources_section(sources)}"
        tps = (
            generation.output_tokens / generation.generation_seconds
            if generation.generation_seconds > 0
            else 0.0
        )

        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            provider=self.inference_client.provider_name,
            model=self.inference_client.model_name,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation.generation_seconds,
            ttft_seconds=generation.ttft_seconds,
            output_tokens=generation.output_tokens,
            tokens_per_second=tps,
            telemetry_pre=telemetry.pre,
            telemetry_post=telemetry.post,
        )
