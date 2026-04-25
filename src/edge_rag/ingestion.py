from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(stripped):
        chunk = stripped[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _stable_chunk_id(filename: str, page: int, chunk_index: int, chunk_text: str) -> str:
    seed = f"{filename}|{page}|{chunk_index}|{chunk_text}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()[:16]
    return f"{Path(filename).stem}-p{page}-c{chunk_index}-{digest}"


def parse_pdf_into_chunks(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    all_chunks: list[dict[str, object]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _normalize_text(raw_text)
        page_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = _stable_chunk_id(path.name, page_idx, chunk_idx, chunk_text)
            all_chunks.append(
                {
                    "id": chunk_id,
                    "document": chunk_text,
                    "metadata": {
                        "filename": path.name,
                        "page": page_idx,
                        "chunk_id": chunk_id,
                    },
                }
            )

    return all_chunks
