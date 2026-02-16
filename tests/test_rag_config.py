from core.models import GeneratorConfig, RagConfig


def test_generator_config_serializes_rag_block():
    cfg = GeneratorConfig(
        model_id="local-model",
        rag=RagConfig(enabled=True, collection_name="test_collection", top_k=3),
    )

    data = cfg.model_dump()
    assert data["rag"]["enabled"] is True
    assert data["rag"]["collection_name"] == "test_collection"
    assert data["rag"]["top_k"] == 3
