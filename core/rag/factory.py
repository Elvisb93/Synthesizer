from core.models import RagBackend

from .backends import LlamaIndexRagService, RagBackendProtocol
from .service import RagService


def create_rag_backend(*, backend: RagBackend | str, **kwargs) -> RagBackendProtocol:
    backend_value = backend.value if isinstance(backend, RagBackend) else str(backend or "")
    if backend_value.strip().lower() == RagBackend.LLAMA_INDEX.value.lower():
        return LlamaIndexRagService(**kwargs)
    return RagService(**kwargs)
