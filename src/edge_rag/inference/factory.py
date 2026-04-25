from __future__ import annotations

from edge_rag.config import Settings
from edge_rag.inference.base import InferenceClient
from edge_rag.inference.cloud_provider import CloudProvider
from edge_rag.inference.local_provider import LocalProvider


def create_inference_client(settings: Settings, model_override: str | None = None) -> InferenceClient:
    if settings.inference_provider == "local":
        return LocalProvider(settings=settings, model_name=model_override)
    if settings.inference_provider == "nim":
        return CloudProvider(settings=settings)
    raise ValueError("Unsupported provider. Use INFERENCE_PROVIDER=local or INFERENCE_PROVIDER=nim")
