from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import requests
from dotenv import load_dotenv

from edge_rag.benchmark import run_benchmark
from edge_rag.config import Settings
from edge_rag.embeddings import CPUEmbeddingFunction
from edge_rag.indexing import ChromaIndexer
from edge_rag.inference.factory import create_inference_client
from edge_rag.preflight import run_preflight
from edge_rag.rag import RAGOrchestrator


def _build_rag(settings: Settings, model_override: str | None = None) -> RAGOrchestrator:
    embedding_function = CPUEmbeddingFunction(settings.embedding_model)
    indexer = ChromaIndexer(settings=settings, embedding_function=embedding_function)
    inference_client = create_inference_client(settings=settings, model_override=model_override)
    return RAGOrchestrator(settings=settings, indexer=indexer, inference_client=inference_client)


def _load_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.query or [])
    if args.queries_file:
        file_queries = [
            line.strip()
            for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queries.extend(file_queries)
    if not queries:
        raise ValueError("Provide --query and/or --queries-file")
    return queries


def _max_peak_vram_mb(report: dict) -> float | None:
    peaks: list[float] = []
    for query_payload in report.get("queries", {}).values():
        peak = query_payload.get("summary", {}).get("peak_vram_mb", {}).get("max")
        if isinstance(peak, (int, float)):
            peaks.append(float(peak))
    return max(peaks) if peaks else None


def _ollama_model_available(settings: Settings, model_name: str) -> bool:
    endpoint = f"{settings.local_ollama_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        payload = response.json()
        for model in payload.get("models", []):
            if model.get("name") == model_name:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _command_index(args: argparse.Namespace, settings: Settings) -> None:
    embedding_function = CPUEmbeddingFunction(settings.embedding_model)
    indexer = ChromaIndexer(settings=settings, embedding_function=embedding_function)

    if args.pdf_path:
        count = indexer.index_pdf(args.pdf_path)
        print(json.dumps({"file": args.pdf_path, "chunks_indexed": count}, indent=2))
        return

    indexed = indexer.index_directory(args.pdf_dir or settings.pdf_dir)
    print(json.dumps({"indexed": indexed, "collection_count": indexer.count()}, indent=2))


def _command_ask(args: argparse.Namespace, settings: Settings) -> None:
    rag = _build_rag(settings)
    result = rag.ask(args.query)
    print(json.dumps(asdict(result), indent=2))


def _command_benchmark(args: argparse.Namespace, settings: Settings) -> None:
    queries = _load_queries(args)
    runs = args.runs or settings.benchmark_runs
    rag = _build_rag(settings)

    base_report = run_benchmark(
        rag=rag,
        queries=queries,
        runs_per_query=runs,
        output_path=args.output,
    )
    print(json.dumps(base_report, indent=2))

    if not args.compare_q8:
        return

    if settings.inference_provider != "local":
        print("Skipped q8 comparison: INFERENCE_PROVIDER must be local.")
        return

    peak_vram = _max_peak_vram_mb(base_report)
    if peak_vram is not None and peak_vram > 7500:
        print(
            f"Skipped q8 comparison: q4 peak VRAM {peak_vram:.2f} MB exceeds 7500 MB threshold."
        )
        return

    q8_model = settings.local_model_q8
    if not _ollama_model_available(settings, q8_model):
        print(f"Skipped q8 comparison: model not available in Ollama -> {q8_model}")
        return

    q8_output = args.q8_output
    q8_rag = _build_rag(settings=settings, model_override=q8_model)
    q8_report = run_benchmark(
        rag=q8_rag,
        queries=queries,
        runs_per_query=runs,
        output_path=q8_output,
    )
    print(json.dumps({"q8_comparison": q8_report, "q8_output": q8_output}, indent=2))


def _command_parity(args: argparse.Namespace, settings: Settings) -> None:
    local_settings = replace(settings, inference_provider="local")
    local_result = _build_rag(local_settings).ask(args.query)

    nim_settings = replace(settings, inference_provider="nim")
    nim_result = _build_rag(nim_settings).ask(args.query)

    local_keys = set(asdict(local_result).keys())
    nim_keys = set(asdict(nim_result).keys())
    if local_keys != nim_keys:
        raise RuntimeError("Provider schema mismatch detected between local and nim responses.")

    print(
        json.dumps(
            {
                "schema_parity": True,
                "keys": sorted(local_keys),
                "local": asdict(local_result),
                "nim": asdict(nim_result),
            },
            indent=2,
        )
    )


def _command_preflight(_: argparse.Namespace, settings: Settings) -> None:
    report = run_preflight(settings=settings, repo_root=".")
    print(json.dumps(report, indent=2))


def _command_pull_model(args: argparse.Namespace, settings: Settings) -> None:
    model_name = args.model or settings.local_model_q4
    endpoint = f"{settings.local_ollama_url.rstrip('/')}/api/pull"
    response = requests.post(endpoint, json={"name": model_name, "stream": False}, timeout=600)
    response.raise_for_status()

    validation = _ollama_model_available(settings, model_name)
    print(
        json.dumps(
            {
                "pulled_model": model_name,
                "validated_in_tags": validation,
            },
            indent=2,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NIM-ready edge RAG prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_index = subparsers.add_parser("index", help="Index one PDF or all PDFs in a directory")
    parser_index.add_argument("--pdf-path", type=str, help="Path to a single PDF")
    parser_index.add_argument("--pdf-dir", type=str, help="Directory containing PDFs")

    parser_ask = subparsers.add_parser("ask", help="Ask a grounded RAG question")
    parser_ask.add_argument("--query", type=str, required=True, help="User query")

    parser_benchmark = subparsers.add_parser("benchmark", help="Run repeated benchmark queries")
    parser_benchmark.add_argument("--query", action="append", help="Benchmark query (repeatable)")
    parser_benchmark.add_argument("--queries-file", type=str, help="Path to newline-separated queries")
    parser_benchmark.add_argument("--runs", type=int, help="Runs per query override")
    parser_benchmark.add_argument(
        "--output",
        type=str,
        default="benchmarks/benchmark_report_q4.json",
        help="Output JSON report path for baseline q4",
    )
    parser_benchmark.add_argument(
        "--compare-q8",
        action="store_true",
        help="Optionally run q8 benchmark if under VRAM threshold and model exists",
    )
    parser_benchmark.add_argument(
        "--q8-output",
        type=str,
        default="benchmarks/benchmark_report_q8.json",
        help="Output JSON report path for q8 results",
    )

    parser_parity = subparsers.add_parser("parity", help="Validate local vs nim schema parity")
    parser_parity.add_argument("--query", type=str, required=True, help="Single parity check query")

    subparsers.add_parser(
        "preflight",
        help="Run prerequisite, model-readiness, and secret-scan checks",
    )

    parser_pull = subparsers.add_parser(
        "pull-model",
        help="Pull local Ollama model and validate availability",
    )
    parser_pull.add_argument(
        "--model",
        type=str,
        help="Model to pull (defaults to LOCAL_MODEL_Q4)",
    )

    return parser


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.validate()

    if args.command == "index":
        _command_index(args, settings)
    elif args.command == "ask":
        _command_ask(args, settings)
    elif args.command == "benchmark":
        _command_benchmark(args, settings)
    elif args.command == "parity":
        _command_parity(args, settings)
    elif args.command == "preflight":
        _command_preflight(args, settings)
    elif args.command == "pull-model":
        _command_pull_model(args, settings)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
