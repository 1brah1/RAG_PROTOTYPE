from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    inference_provider: str = "local"
    telemetry_enabled: bool = True
    benchmark_runs: int = 10

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "data/chroma"
    collection_name: str = "edge_rag_documents"

    local_ollama_url: str = "http://localhost:11434"
    local_model_q4: str = "llama3.1:8b-instruct-q4_K_M"
    local_model_q8: str = "llama3.1:8b-instruct-q8_0"

    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_api_key: str | None = None

    max_tokens: int = 512
    temperature: float = 0.1

    pdf_dir: str = "data/pdfs"

    @staticmethod
    def _parse_bool(value: str | None, default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "3")),
            inference_provider=os.getenv("INFERENCE_PROVIDER", "local").strip().lower(),
            telemetry_enabled=cls._parse_bool(os.getenv("TELEMETRY_ENABLED"), True),
            benchmark_runs=int(os.getenv("BENCHMARK_RUNS", "10")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chroma_path=os.getenv("CHROMA_PATH", "data/chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION", "edge_rag_documents"),
            local_ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            local_model_q4=os.getenv("LOCAL_MODEL_Q4", "llama3.1:8b-instruct-q4_K_M"),
            local_model_q8=os.getenv("LOCAL_MODEL_Q8", "llama3.1:8b-instruct-q8_0"),
            nim_base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nim_model=os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            pdf_dir=os.getenv("PDF_DIR", "data/pdfs"),
        )

    def validate(self) -> None:
        if self.inference_provider not in {"local", "nim"}:
            raise ValueError("INFERENCE_PROVIDER must be 'local' or 'nim'.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be a positive integer.")
        if self.benchmark_runs <= 0:
            raise ValueError("BENCHMARK_RUNS must be a positive integer.")
