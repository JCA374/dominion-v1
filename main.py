"""Phase-Aware Genetic Algorithm for Simplified Dominion — entry point.

Usage:
    python main.py              # fresh training run
    python main.py --continue   # continue from best model, numbering continues
"""

import argparse
import re
import time

from copy import deepcopy
from ga import run_ga, mutate
from strategy import describe, summarize, save_best_model, load_strategy, random_strategy
from plotting import save_all_plots
from fitness import evaluate_vs_opponent, evaluate_vs_hall, make_seed_list
import os
import random

# === Config ===
POP_SIZE        = 60 #60
GENERATIONS     = 10000 #100
GAMES_PER_EVAL  = 80 #50
TOURNAMENT_SIZE = 3
ELITE_COUNT     = 2
MUTATION_RATE   = 0.18
SEED            = 42
# All available kingdom cards (12 total — a standard game uses 10)
ALL_KINGDOM     = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel",
                   "Throne Room", "Council Room", "Moneylender", "Gardens",
                   "Mine", "Merchant"]
# Select 10 for training — change this list to train on different kingdoms
KINGDOM         = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel",
                   "Throne Room", "Council Room", "Mine", "Moneylender"]
HALL_MAX_SIZE   = 6    # Maximum hall of fame opponents
HALL_ADD_THRESHOLD = 0.55  # Add to hall when win rate exceeds this
WORKERS         = 8    # Parallel workers for evaluation (1 = sequential)


def _find_last_gen(model_dir: str = "best_model") -> int:
    """Scan best_model/gen_NNN dirs and return the highest generation number, or 0."""
    last = 0
    if os.path.isdir(model_dir):
        for entry in os.listdir(model_dir):
            m = re.match(r"gen_(\d+)", entry)
            if m:
                last = max(last, int(m.group(1)))
    return last


def _build_continue_population(best: 'Strategy', pop_size: int,
                               rng: random.Random,
                               kingdom: list[str] | None = None) -> list['Strategy']:
    """Build a population seeded from the best model.

    Keeps the best strategy as elite, fills the rest with mutated variants
    to provide genetic diversity.
    """
    population = [deepcopy(best)]
    for _ in range(pop_size - 1):
        # Higher mutation rate for initial diversity
        child = mutate(deepcopy(best), rate=0.3, rng=rng, kingdom=kingdom)
        population.append(child)
    return population


def main():
    parser = argparse.ArgumentParser(description="Dominion GA trainer")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue training from best_model/, generation numbering continues")
    args = parser.parse_args()

    continuing = args.continue_training
    start_gen = 0
    initial_population = None

    if continuing:
        if not os.path.exists("best_model/strategy.json"):
            print("ERROR: No best_model/strategy.json found. Run a fresh training first.")
            return
        best = load_strategy("best_model/strategy.json")
        start_gen = _find_last_gen()
        rng = random.Random(SEED + start_gen)
        initial_population = _build_continue_population(best, POP_SIZE, rng, KINGDOM)
        print(f"=== Continuing from generation {start_gen} ===")
    else:
        print(f"=== Phase-Aware GA for Simplified Dominion ===")

    print(f"Seed: {SEED}")
    print(f"Population: {POP_SIZE}, Generations: {start_gen + 1}-{start_gen + GENERATIONS}, "
          f"Games/eval: {GAMES_PER_EVAL}")
    print(f"Hall of Fame: max_size={HALL_MAX_SIZE}, add_threshold={HALL_ADD_THRESHOLD}")
    print(f"Workers: {WORKERS}")
    print()

    config = {
        "pop_size": POP_SIZE,
        "generations": GENERATIONS,
        "games_per_eval": GAMES_PER_EVAL,
        "tournament_size": TOURNAMENT_SIZE,
        "elite_count": ELITE_COUNT,
        "mutation_rate": MUTATION_RATE,
        "kingdom": KINGDOM,
        "seed": SEED + start_gen,  # different seed so we don't replay same games
        "hall_max_size": HALL_MAX_SIZE,
        "hall_add_threshold": HALL_ADD_THRESHOLD,
        "workers": WORKERS,
        "csv_path": "evolution_log.csv",
        "start_gen": start_gen,
        "initial_population": initial_population,
        "csv_append": continuing,
    }

    start = time.time()
    result = run_ga(config)
    elapsed = time.time() - start

    print(f"\nEvolution complete in {elapsed:.1f}s\n")

    # Evaluate best strategy with more games against the final hall
    hall = result["hall"]
    rng = random.Random(SEED + 1000)
    eval_seeds = make_seed_list(200, rng)
    vs = evaluate_vs_hall(result["best_strategy"], eval_seeds, hall, KINGDOM)
    vs["num_games"] = 200
    vs["opponent"] = f"hall({len(hall)})"

    # Print full summary
    print(summarize(result["best_strategy"], vs))

    # Save best model (also saved during evolution, but final eval has more games)
    save_best_model(result["best_strategy"], vs)

    # Generate plots
    print()
    save_all_plots(result["log"], result["best_strategy"])

    print("\nDone. Check evolution_log.csv, best_model/, and *.png files.")


if __name__ == "__main__":
    main()
