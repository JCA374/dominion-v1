"""Phase-Aware Genetic Algorithm for Simplified Dominion — entry point."""

import time

from ga import run_ga
from strategy import describe, summarize, save_best_model, load_strategy
from plotting import save_all_plots
from fitness import evaluate_vs_opponent, make_seed_list
import os
import random

# === Config ===
POP_SIZE        = 60 #60
GENERATIONS     = 500 #100
GAMES_PER_EVAL  = 50 #50
TOURNAMENT_SIZE = 4
ELITE_COUNT     = 2
MUTATION_RATE   = 0.1
SEED            = 42
KINGDOM         = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel"]
OPPONENT_PATH   = "auto"  # "auto" = use best_model/strategy.json if it exists, None = Big Money
SWITCH_AT       = 0.7  # Auto-switch opponent when win rate exceeds this
WORKERS         = 8    # Parallel workers for evaluation (1 = sequential)


def main():
    # Load opponent: "auto" checks for saved model, None forces Big Money
    opponent = None
    opponent_label = "Big Money"
    opponent_path = OPPONENT_PATH
    if opponent_path == "auto":
        opponent_path = "best_model/strategy.json"
    if opponent_path and os.path.exists(opponent_path):
        opponent = load_strategy(opponent_path)
        opponent_label = opponent_path

    print(f"=== Phase-Aware GA for Simplified Dominion ===")
    print(f"Seed: {SEED}")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}, "
          f"Games/eval: {GAMES_PER_EVAL}")
    print(f"Opponent: {opponent_label}")
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
        "seed": SEED,
        "opponent": opponent,
        "opponent_label": opponent_label,
        "switch_threshold": SWITCH_AT,
        "workers": WORKERS,
        "csv_path": "evolution_log.csv",
    }

    start = time.time()
    result = run_ga(config)
    elapsed = time.time() - start

    print(f"\nEvolution complete in {elapsed:.1f}s\n")

    # Evaluate best strategy with more games against the final opponent
    final_opponent = result["opponent"]
    final_label = result["opponent_label"]
    rng = random.Random(SEED + 1000)
    eval_seeds = make_seed_list(200, rng)
    vs = evaluate_vs_opponent(result["best_strategy"], eval_seeds, KINGDOM,
                              opponent=final_opponent)
    vs["num_games"] = 200
    vs["opponent"] = final_label

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
