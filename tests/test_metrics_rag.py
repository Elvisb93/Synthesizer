from core.metrics import calculate_metrics
from core.models import GeneratorConfig


class _FakeLLMClient:
    def __init__(self):
        self.latency_stats = {"total_time": 0.4, "count": 2}

    def get_token_usage(self):
        return {"prompt_tokens": 1000, "completion_tokens": 500}

    def get_rag_stats(self):
        return {
            "queries": 10,
            "queries_with_hits": 8,
            "total_retrieval_time": 0.25,
            "total_context_chars": 4000,
            "last_hits": 3,
        }


def test_metrics_includes_rag_stats():
    config = GeneratorConfig(model_id="local-model")
    llm = _FakeLLMClient()
    result = calculate_metrics(
        config=config,
        generated_rows=[],
        llm_client=llm,
        run_metrics={},
    )

    rag = result["stats"]["rag"]
    assert rag["queries"] == 10
    assert rag["last_hits"] == 3
    assert rag["hit_rate"] == 80.0
    assert rag["avg_retrieval_ms"] > 0
    assert rag["avg_context_chars"] == 400.0
