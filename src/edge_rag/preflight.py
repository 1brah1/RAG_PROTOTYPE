from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

from edge_rag.config import Settings
from edge_rag.security import scan_for_nvapi_tokens


def _check_python_version() -> dict[str, Any]:
    ok = sys.version_info >= (3, 10)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return {
        "name": "python_version",
        "ok": ok,
        "details": f"Detected Python {version} (requires >= 3.10)",
    }


def _check_command_exists(command_name: str) -> dict[str, Any]:
    path = shutil.which(command_name)
    if path is None and command_name == "ollama":
        candidate = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            path = str(candidate)
    return {
        "name": f"command_{command_name}",
        "ok": path is not None,
        "details": path or f"{command_name} not found in PATH",
    }


def _check_nvidia_smi() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
        details = completed.stdout.strip() or "nvidia-smi is available"
        return {"name": "nvidia_smi", "ok": True, "details": details}
    except Exception as exc:  # noqa: BLE001
        return {"name": "nvidia_smi", "ok": False, "details": str(exc)}


def _check_disk_space(path: str, min_free_gb: float = 15.0) -> dict[str, Any]:
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    return {
        "name": "disk_space",
        "ok": free_gb >= min_free_gb,
        "details": f"Free disk: {free_gb:.2f} GB (recommended >= {min_free_gb:.1f} GB)",
    }


def _check_ollama_model(settings: Settings) -> dict[str, Any]:
    endpoint = f"{settings.local_ollama_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = [model.get("name") for model in payload.get("models", [])]
        ok = settings.local_model_q4 in models
        details = (
            f"Found required model: {settings.local_model_q4}"
            if ok
            else f"Missing required model: {settings.local_model_q4}"
        )
        return {"name": "ollama_model_q4", "ok": ok, "details": details}
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "ollama_model_q4",
            "ok": False,
            "details": f"Could not query Ollama tags endpoint: {exc}",
        }


def _check_nvidia_key_placeholder(settings: Settings) -> dict[str, Any]:
    value = settings.nvidia_api_key
    if not value:
        return {
            "name": "nvidia_api_key",
            "ok": True,
            "details": "No key loaded (safe default for local mode).",
        }

    if value.startswith("nvapi-") and "REPLACE_ME" not in value:
        return {
            "name": "nvidia_api_key",
            "ok": True,
            "details": "NVIDIA key loaded for cloud mode. Keep it local-only and never commit .env.",
        }

    return {
        "name": "nvidia_api_key",
        "ok": True,
        "details": "Placeholder key detected.",
    }


def _check_repo_secrets(repo_root: str) -> dict[str, Any]:
    findings = scan_for_nvapi_tokens(repo_root)
    return {
        "name": "repo_secret_scan",
        "ok": len(findings) == 0,
        "details": "No nvapi- tokens found" if not findings else f"Possible secret exposure in: {findings}",
    }


def run_preflight(settings: Settings, repo_root: str) -> dict[str, Any]:
    checks = [
        _check_python_version(),
        _check_command_exists("ollama"),
        _check_nvidia_smi(),
        _check_disk_space(repo_root),
        _check_ollama_model(settings),
        _check_nvidia_key_placeholder(settings),
        _check_repo_secrets(repo_root),
    ]

    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
        "repo_root": str(Path(repo_root).resolve()),
    }
