import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.controller import GeneratorController
from core.models import GeneratorConfig, ColumnDefinition, ColumnType

# Mock LLM Client
class MockLLM:
    def get_token_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 50}

def test_metrics():
    config = GeneratorConfig(
        model_id="test-model",
        num_rows=10,
        input_price_per_1m=1.0,  # Easy number for math
        output_price_per_1m=2.0
    )
    
    # Setup Logic
    from queue import Queue
    controller = GeneratorController()
    controller.initialize(config, [])
    controller.llm_client = MockLLM()
    
    # Simulate some "metrics" state normally accumulated during generation
    controller.generated_rows = [1] * 5  # 5 rows
    controller.metrics["llm_cols"] = 2
    controller.metrics["faker_cols"] = 3
    
    # Simulate time and attempts for new metrics
    import time
    controller.metrics["start_time"] = time.time() - 2.5 # 2.5 seconds ago
    controller.metrics["total_attempts"] = 10
    controller.metrics["failed_attempts"] = 2

    # Get Metrics
    metrics = controller.get_metrics()
    
    # Verify Structure
    assert "total" in metrics
    assert "avg_row" in metrics
    assert "stats" in metrics
    
    # Verify Calculations
    t = metrics["total"]
    # Cost = (100/1M * 1.0) + (50/1M * 2.0) = 0.0001 + 0.0001 = 0.0002
    assert abs(t["cost"] - 0.0002) < 1e-9
    
    # Verify Stats
    s = metrics["stats"]
    # Throughput: 5 rows / 2.5s = 2.0 rows/sec
    assert abs(s["throughput"] - 2.0) < 0.1
    # Retry Rate: 2 failed / 10 total = 20%
    assert abs(s["retry_rate"] - 20.0) < 1e-9

    print("Metrics Verified:")
    print(metrics)
    print("\n✅ Metrics verification PASSED")

if __name__ == "__main__":
    test_metrics()
