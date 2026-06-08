"""Benchmark sequential vs parallel evaluation to find optimal settings."""

import time
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from core.strategy import random_strategy, big_money_strategy
from ga.fitness import evaluate_vs_opponent, make_seed_list


def _eval_one(strategy, seed_list, kingdom, opponent):
    """Evaluate a single strategy (top-level function for pickling)."""
    return evaluate_vs_opponent(strategy, seed_list, kingdom, opponent=opponent)


def benchmark(pop_size=60, games_per_eval=50, kingdom=None):
    rng = random.Random(42)
    population = [random_strategy(rng) for _ in range(pop_size)]
    seed_list = make_seed_list(games_per_eval, rng)
    opponent = big_money_strategy()

    max_workers = os.cpu_count() or 4
    configs = [
        ("sequential", 0),
        ("2 workers", 2),
        ("4 workers", 4),
    ]
    if max_workers >= 8:
        configs.append(("8 workers", 8))
    if max_workers >= 12:
        configs.append(("12 workers", 12))

    print(f"CPU cores: {max_workers}")
    print(f"Population: {pop_size}, Games/eval: {games_per_eval} "
          f"({games_per_eval * 2} games per strategy)")
    print()

    for label, workers in configs:
        start = time.time()
        if workers == 0:
            results = [evaluate_vs_opponent(s, seed_list, kingdom, opponent=opponent)
                       for s in population]
        else:
            fn = partial(_eval_one, seed_list=seed_list, kingdom=kingdom,
                         opponent=opponent)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(fn, population))
        elapsed = time.time() - start
        print(f"  {label:>15s}: {elapsed:5.2f}s  "
              f"({elapsed / pop_size * 1000:.0f}ms/individual)")

    print()
    print(f"Recommendation: use parallel evaluation with "
          f"{min(max_workers, pop_size)} workers")


if __name__ == "__main__":
    benchmark(
        kingdom=["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel"],
    )
