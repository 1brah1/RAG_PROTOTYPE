from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from edge_rag.rag import RAGOrchestrator
from edge_rag.types import RAGResult


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _safe_min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _variability_note(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient data"
    mean_value = _safe_mean(values)
    if mean_value <= 0:
        return "no variability signal"
    stdev = statistics.pstdev(values)
    cv = stdev / mean_value
    if cv < 0.10:
        return "stable"
    if cv < 0.20:
        return "moderate variability"
    return "high variability"


def _metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": _safe_mean(values),
        "median": _safe_median(values),
        "min": _safe_min(values),
        "max": _safe_max(values),
        "variability": _variability_note(values),
    }


def _peak_vram(result: RAGResult) -> float | None:
    candidates: list[float] = []
    if result.telemetry_pre and result.telemetry_pre.memory_used_mb is not None:
        candidates.append(result.telemetry_pre.memory_used_mb)
    if result.telemetry_post and result.telemetry_post.memory_used_mb is not None:
        candidates.append(result.telemetry_post.memory_used_mb)
    return max(candidates) if candidates else None


def run_benchmark(
    rag: RAGOrchestrator,
    queries: list[str],
    runs_per_query: int,
    output_path: str | None = None,
) -> dict[str, Any]:
    if runs_per_query <= 0:
        raise ValueError("runs_per_query must be positive")

    report: dict[str, Any] = {
        "provider": rag.inference_client.provider_name,
        "model": rag.inference_client.model_name,
        "runs_per_query": runs_per_query,
        "queries": {},
    }

    for query in queries:
        run_results: list[RAGResult] = []
        for _ in range(runs_per_query):
            run_results.append(rag.ask(query))

        ttft_values = [result.ttft_seconds for result in run_results]
        generation_values = [result.generation_seconds for result in run_results]
        tps_values = [result.tokens_per_second for result in run_results]
        peak_vram_values = [
            value for value in (_peak_vram(result) for result in run_results) if value is not None
        ]

        report["queries"][query] = {
            "summary": {
                "ttft_seconds": _metric_summary(ttft_values),
                "generation_seconds": _metric_summary(generation_values),
                "tokens_per_second": _metric_summary(tps_values),
                "peak_vram_mb": _metric_summary(peak_vram_values)
                if peak_vram_values
                else {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "variability": "GPU metrics unavailable",
                },
            },
            "runs": [asdict(result) for result in run_results],
        }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
