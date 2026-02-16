"""
Token usage tracking, cost calculation, and generation performance metrics.

Provides real-time metrics including:
- Token counts (prompt/completion) and cost estimates
- Savings from deterministic (Faker) columns vs LLM columns
- Throughput, retry rates, latency, and estimated time remaining
"""
import time
from typing import Dict, Any, Optional, List

from .models import GeneratorConfig, RowData


def calculate_metrics(
    config: GeneratorConfig,
    generated_rows: List[RowData],
    llm_client: Any,
    run_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate real-time metrics including token usage, cost, and estimated savings.

    Args:
        config: Current generator configuration (for pricing).
        generated_rows: List of generated row data.
        llm_client: The LLM client instance (for token usage / latency stats).
        run_metrics: Dict with keys like 'faker_cols', 'llm_cols', 'start_time', etc.

    Returns:
        Dict with 'total', 'avg_row', and 'stats' sub-dicts.
    """
    if not llm_client:
        return {}

    token_usage = llm_client.get_token_usage()
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)

    # Cost calculation
    def calc_cost(prompt: int, completion: int) -> float:
        return ((prompt / 1_000_000) * config.input_price_per_1m) + \
               ((completion / 1_000_000) * config.output_price_per_1m)

    total_cost = calc_cost(prompt_tokens, completion_tokens)
    total_tokens = prompt_tokens + completion_tokens

    # Averages per ROW
    num_rows = len(generated_rows)
    if num_rows > 0:
        avg_tokens_in_row = prompt_tokens / num_rows
        avg_tokens_out_row = completion_tokens / num_rows
        avg_tokens_total_row = total_tokens / num_rows
        avg_cost_row = total_cost / num_rows
    else:
        avg_tokens_in_row = 0
        avg_tokens_out_row = 0
        avg_tokens_total_row = 0
        avg_cost_row = 0

    # Savings estimation (Faker cols vs LLM cols)
    llm_cols_count = run_metrics.get("llm_cols", 0)
    faker_cols_count = run_metrics.get("faker_cols", 0)

    avg_tokens_per_llm_col = 0
    avg_cost_per_llm_col = 0

    if llm_cols_count > 0:
        avg_tokens_per_llm_col = total_tokens / llm_cols_count
        avg_cost_per_llm_col = total_cost / llm_cols_count

    saved_tokens = int(faker_cols_count * avg_tokens_per_llm_col)
    saved_cost = faker_cols_count * avg_cost_per_llm_col

    # Savings per ROW
    avg_saved_tokens_row = 0
    avg_saved_cost_row = 0
    if num_rows > 0:
        avg_saved_tokens_row = saved_tokens / num_rows
        avg_saved_cost_row = saved_cost / num_rows

    # Throughput & timing stats
    rows_per_sec = 0
    elapsed = 0

    start_t = run_metrics.get("start_time")
    end_t = run_metrics.get("end_time")
    last_row_t = run_metrics.get("last_row_time")

    if start_t and num_rows > 0:
        current_elapsed_ref = end_t if end_t else time.time()
        elapsed = current_elapsed_ref - start_t

        calc_ref = end_t if end_t else (last_row_t or time.time())
        duration = max(0.1, calc_ref - start_t)
        rows_per_sec = num_rows / duration

    retry_rate = 0
    total_attempts = run_metrics.get("total_attempts", 0)
    if total_attempts > 0:
        retry_rate = (run_metrics.get("failed_attempts", 0) / total_attempts) * 100

    # Latency stats
    avg_latency = 0.0
    if llm_client and llm_client.latency_stats["count"] > 0:
        avg_latency = llm_client.latency_stats["total_time"] / llm_client.latency_stats["count"]

    # ETR (Estimated Time Remaining)
    etr = 0.0
    target_rows = len(config.existing_data) if config.existing_data else config.num_rows
    remaining = max(0, target_rows - num_rows)
    if rows_per_sec > 0 and remaining > 0:
        etr = remaining / rows_per_sec

    return {
        "total": {
            "in": prompt_tokens,
            "out": completion_tokens,
            "used": total_tokens,
            "cost": total_cost,
            "saved_tokens": saved_tokens,
            "saved_cost": saved_cost
        },
        "avg_row": {
            "in": avg_tokens_in_row,
            "out": avg_tokens_out_row,
            "used": avg_tokens_total_row,
            "cost": avg_cost_row,
            "saved_tokens": avg_saved_tokens_row,
            "saved_cost": avg_saved_cost_row
        },
        "stats": {
            "throughput": rows_per_sec,
            "retry_rate": retry_rate,
            "elapsed": elapsed,
            "generated": num_rows,
            "target": target_rows,
            "latency": avg_latency,
            "etr": etr
        }
    }
