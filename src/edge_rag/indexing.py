from __future__ import annotations

from pathlib import Path

import chromadb

from edge_rag.config import Settings
from edge_rag.ingestion import parse_pdf_into_chunks
from edge_rag.types import RetrievedChunk


class ChromaIndexer:
    def __init__(self, settings: Settings, embedding_function: object) -> None:
        self.settings = settings
        Path(self.settings.chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdf(self, pdf_path: str) -> int:
        chunks = parse_pdf_into_chunks(
            pdf_path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return 0

        ids = [item["id"] for item in chunks]
        documents = [item["document"] for item in chunks]
        metadatas = [item["metadata"] for item in chunks]

        # Deterministic ids + upsert make repeated indexing idempotent.
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def index_directory(self, pdf_dir: str) -> dict[str, int]:
        path = Path(pdf_dir)
        if not path.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        indexed: dict[str, int] = {}
        for pdf_path in sorted(path.glob("*.pdf")):
            indexed[pdf_path.name] = self.index_pdf(str(pdf_path))
        return indexed

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    text=str(doc),
                    filename=str(meta.get("filename", "unknown")),
                    page=int(meta.get("page", 0)),
                    score=float(distance) if distance is not None else None,
                )
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()
