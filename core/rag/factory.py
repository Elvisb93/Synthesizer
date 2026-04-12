from core.models import RagBackend

from .backends import LlamaIndexRagService, RagBackendProtocol
from .service import RagService


def create_rag_backend(*, backend: RagBackend | str, **kwargs) -> RagBackendProtocol:
    backend_value = backend.value if isinstance(backend, RagBackend) else str(backend or "")
    if backend_value.strip().lower() == RagBackend.LLAMA_INDEX.value.lower():
        return LlamaIndexRagService(**kwargs)
    native_kwargs = dict(kwargs)
    for extra_key in (
        "llm_model_name",
        "llm_base_url",
        "llm_api_key",
        "llm_temperature",
        "llm_context_window",
        "llm_num_output",
        "llm_enabled",
    ):
        native_kwargs.pop(extra_key, None)
    return RagService(**native_kwargs)
