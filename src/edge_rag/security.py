from __future__ import annotations

import re
from pathlib import Path


NVAPI_KEY_PATTERN = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")


def _is_ignored_path(path: Path) -> bool:
    ignored_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", ".egg-info"}
    if any(part in ignored_parts for part in path.parts):
        return True

    if path.name.startswith(".") and path.name not in {".env.example", ".gitignore"}:
        return True

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".pyc"}:
        return True

    return False


def _has_real_nvapi_token(text: str) -> bool:
    for match in NVAPI_KEY_PATTERN.findall(text):
        if "REPLACE_ME" in match:
            continue
        return True
    return False


def scan_for_nvapi_tokens(repo_root: str) -> list[str]:
    root = Path(repo_root)
    findings: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if _is_ignored_path(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue

        if _has_real_nvapi_token(text):
            findings.append(str(path))

    return findings
