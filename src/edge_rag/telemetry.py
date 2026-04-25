from __future__ import annotations

import datetime as dt
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from edge_rag.types import GPUSnapshot


@dataclass
class GenerationTelemetry:
    pre: GPUSnapshot | None = None
    post: GPUSnapshot | None = None


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_nvidia_smi_snapshot() -> GPUSnapshot:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_row = completed.stdout.strip().splitlines()[0]
        memory_used, utilization = [value.strip() for value in first_row.split(",", maxsplit=1)]
        return GPUSnapshot(
            timestamp_utc=_utc_now(),
            memory_used_mb=float(memory_used),
            utilization_gpu_percent=float(utilization),
            available=True,
            note="",
        )
    except Exception as exc:  # noqa: BLE001
        return GPUSnapshot(
            timestamp_utc=_utc_now(),
            memory_used_mb=None,
            utilization_gpu_percent=None,
            available=False,
            note=f"GPU metrics unavailable: {exc}",
        )


@contextmanager
def capture_generation_telemetry(enabled: bool) -> Iterator[GenerationTelemetry]:
    telemetry = GenerationTelemetry()
    if enabled:
        telemetry.pre = read_nvidia_smi_snapshot()

    yield telemetry

    if enabled:
        telemetry.post = read_nvidia_smi_snapshot()
