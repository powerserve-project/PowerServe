from common import *


def get_score(stat: dict) -> float:
    draft_model_latency_ms = 8.421
    target_model_latency_ms = 62.867

    n_iterations = stat["n_iterations"]
    n_draft_times = stat["n_draft_times"]
    n_generated_tokens = stat["n_generated_tokens"]

    latency_ms = n_iterations * target_model_latency_ms + (n_iterations + n_draft_times) * draft_model_latency_ms
    tokens_per_second = n_generated_tokens * 1000 / latency_ms

    return tokens_per_second


leaderboard = sorted((get_score(stat), params, stat) for params, stat in database.items() if stat is not None)

for score, params, stat in leaderboard:
    print(f"{score:.3f} '{format_params(params)}' {stat}")
