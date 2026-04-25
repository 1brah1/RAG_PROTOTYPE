from __future__ import annotations

from collections.abc import Sequence as CSequence
from typing import Sequence

from sentence_transformers import SentenceTransformer


class CPUEmbeddingFunction:
    """Chroma-compatible embedding function pinned to CPU to preserve GPU VRAM."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        if not input:
            return []
        embeddings = self._model.encode(
            list(input),
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cpu",
        )
        return embeddings.tolist()

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_query(self, input: str | Sequence[str]) -> list[list[float]]:
        if isinstance(input, str):
            return self.__call__([input])
        if isinstance(input, CSequence):
            return self.__call__(list(input))
        return self.__call__([str(input)])

    def name(self) -> str:
        return f"cpu_sentence_transformer:{self.model_name}"
