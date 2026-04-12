from unittest.mock import patch

from core.controller import GeneratorController
from core.models import GeneratorConfig, RagBackend, RagConfig


def test_controller_initialize_rag_uses_selected_backend():
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local-model",
        rag=RagConfig(backend=RagBackend.LLAMA_INDEX, collection_name="llama_docs"),
    )

    sentinel = object()
    with patch("core.controller.create_rag_backend", return_value=sentinel) as mock_factory:
        controller.initialize_rag()

    assert controller.rag_service is sentinel
    assert mock_factory.call_args.kwargs["backend"] == RagBackend.LLAMA_INDEX
