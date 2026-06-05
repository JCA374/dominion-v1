"""Genetic algorithm: population management, selection, crossover, mutation."""

from __future__ import annotations

import csv
import os
import random
from copy import deepcopy
from dataclasses import asdict

from cards import BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS
from fitness import evaluate, evaluate_vs_opponent, evaluate_population, make_seed_list
from strategy import (
    Strategy, Transitions, random_strategy, big_money_strategy,
    engine_strategy, gardens_strategy, describe, get_current_phase,
    save_best_model,
)


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def init_population(pop_size: int, rng: random.Random) -> list[Strategy]:
    """Create initial population: random strategies + seeded archetypes."""
    population = [big_money_strategy(), engine_strategy(), gardens_strategy()]
    for _ in range(pop_size - len(population)):
        population.append(random_strategy(rng))
    return population


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_select(population: list[Strategy], fitnesses: list[float],
                      k: int, rng: random.Random) -> Strategy:
    """Pick k random individuals, return the one with highest fitness."""
    indices = rng.sample(range(len(population)), k)
    best = max(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best])


# ---------------------------------------------------------------------------
# Order Crossover (OX)
# ---------------------------------------------------------------------------

def order_crossover(parent1: list[str], parent2: list[str],
                    rng: random.Random) -> list[str]:
    """OX crossover on a single priority list. Returns one child."""
    # Unify element sets (handle PASS/STOP presence differences)
    s1, s2 = set(parent1), set(parent2)
    if s1 != s2:
        # Add missing elements at end of each parent (temporary)
        missing_in_1 = list(s2 - s1)
        missing_in_2 = list(s1 - s2)
        p1 = list(parent1) + missing_in_1
        p2 = list(parent2) + missing_in_2
    else:
        p1, p2 = list(parent1), list(parent2)

    n = len(p1)
    if n <= 1:
        return list(p1)

    # Pick two crossover points
    i, j = sorted(rng.sample(range(n), 2))

    child = [None] * n
    # Copy segment from parent1
    child[i:j + 1] = p1[i:j + 1]
    placed = set(child[i:j + 1])

    # Fill from parent2 starting at j+1, wrapping around
    fill = [x for x in p2[j + 1:] + p2[:j + 1] if x not in placed]
    pos = (j + 1) % n
    for item in fill:
        while child[pos] is not None:
            pos = (pos + 1) % n
        child[pos] = item
        pos = (pos + 1) % n

    # Normalize: remove duplicate PASS (keep first)
    seen_pass = False
    result = []
    for item in child:
        if item == "PASS":
            if seen_pass:
                continue
            seen_pass = True
        result.append(item)

    return result


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def _crossover_targets(t1: dict[str, int], t2: dict[str, int],
                       rng: random.Random) -> dict[str, int]:
    """Per-card crossover: pick each target from either parent."""
    all_keys = set(t1) | set(t2)
    result = {}
    for key in all_keys:
        if key in t1 and key in t2:
            result[key] = rng.choice([t1[key], t2[key]])
        elif key in t1:
            result[key] = t1[key]
        else:
            result[key] = t2[key]
    return result


def crossover(p1: Strategy, p2: Strategy, rng: random.Random) -> Strategy:
    """Crossover two strategies into a child."""
    return Strategy(
        early_buy_priority=order_crossover(p1.early_buy_priority, p2.early_buy_priority, rng),
        mid_buy_priority=order_crossover(p1.mid_buy_priority, p2.mid_buy_priority, rng),
        late_buy_priority=order_crossover(p1.late_buy_priority, p2.late_buy_priority, rng),
        action_priority=order_crossover(p1.action_priority, p2.action_priority, rng),
        chapel_trash_priority=order_crossover(
            p1.chapel_trash_priority, p2.chapel_trash_priority, rng
        ),
        throne_room_priority=order_crossover(
            p1.throne_room_priority, p2.throne_room_priority, rng
        ),
        mine_trash_priority=order_crossover(
            p1.mine_trash_priority, p2.mine_trash_priority, rng
        ),
        chapel_max_trash=rng.choice([p1.chapel_max_trash, p2.chapel_max_trash]),
        transitions=Transitions(
            early_to_mid_turn=rng.choice([
                p1.transitions.early_to_mid_turn,
                p2.transitions.early_to_mid_turn,
            ]),
            mid_to_late_provinces=rng.choice([
                p1.transitions.mid_to_late_provinces,
                p2.transitions.mid_to_late_provinces,
            ]),
        ),
        buy_targets=_crossover_targets(p1.buy_targets, p2.buy_targets, rng),
    )


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _mutate_list(lst: list[str], rate: float, rng: random.Random,
                 allow_pass: bool = True) -> list[str]:
    """Swap mutation on a priority list. Optionally insert/remove PASS."""
    lst = list(lst)
    # Swap two positions
    if len(lst) >= 2 and rng.random() < rate:
        i, j = rng.sample(range(len(lst)), 2)
        lst[i], lst[j] = lst[j], lst[i]

    # PASS insert/remove
    if allow_pass and rng.random() < rate / 5:
        if "PASS" in lst:
            lst.remove("PASS")
        else:
            lst.insert(rng.randint(0, len(lst)), "PASS")

    return lst


def mutate(strategy: Strategy, rate: float, rng: random.Random) -> Strategy:
    """Apply mutation to a strategy."""
    s = deepcopy(strategy)
    s.early_buy_priority = _mutate_list(s.early_buy_priority, rate, rng)
    s.mid_buy_priority = _mutate_list(s.mid_buy_priority, rate, rng)
    s.late_buy_priority = _mutate_list(s.late_buy_priority, rate, rng)
    s.action_priority = _mutate_list(s.action_priority, rate, rng, allow_pass=False)
    s.throne_room_priority = _mutate_list(s.throne_room_priority, rate, rng, allow_pass=False)
    s.mine_trash_priority = _mutate_list(s.mine_trash_priority, rate, rng, allow_pass=False)
    s.chapel_trash_priority = _mutate_list(s.chapel_trash_priority, rate, rng, allow_pass=False)

    # Jitter chapel_max_trash
    if rng.random() < rate:
        s.chapel_max_trash += rng.choice([-1, 0, 1])
        s.chapel_max_trash = max(0, min(4, s.chapel_max_trash))

    # Jitter transitions
    if rng.random() < rate:
        s.transitions.early_to_mid_turn += rng.choice([-1, 0, 1])
        s.transitions.early_to_mid_turn = max(2, min(15, s.transitions.early_to_mid_turn))

    if rng.random() < rate:
        s.transitions.mid_to_late_provinces += rng.choice([-1, 0, 1])
        s.transitions.mid_to_late_provinces = max(0, min(8, s.transitions.mid_to_late_provinces))

    # Jitter buy targets
    for card in list(s.buy_targets):
        if rng.random() < rate:
            s.buy_targets[card] += rng.choice([-1, 0, 1])
            s.buy_targets[card] = max(0, min(10, s.buy_targets[card]))

    # Occasionally add/remove a target for a kingdom card
    if rng.random() < rate / 3:
        card = rng.choice(KINGDOM_CARDS)
        if card in s.buy_targets:
            del s.buy_targets[card]  # remove limit
        else:
            s.buy_targets[card] = rng.randint(1, 4)  # add limit

    return s


# ---------------------------------------------------------------------------
# GA run loop
# ---------------------------------------------------------------------------

def run_ga(config: dict) -> dict:
    """Main GA loop. Returns {"best_strategy": Strategy, "log": list[dict]}."""
    pop_size = config["pop_size"]
    generations = config["generations"]
    games_per_eval = config["games_per_eval"]
    tournament_size = config["tournament_size"]
    elite_count = config["elite_count"]
    mutation_rate = config["mutation_rate"]
    kingdom = config.get("kingdom")
    seed = config["seed"]
    opponent = config.get("opponent")
    opponent_label = config.get("opponent_label", "Big Money")
    switch_threshold = config.get("switch_threshold", 0.7)
    best_model_dir = config.get("best_model_dir", "best_model")
    workers = config.get("workers", 1)

    start_gen = config.get("start_gen", 0)
    initial_population = config.get("initial_population")

    master_rng = random.Random(seed)
    ga_rng = random.Random(master_rng.randint(0, 2**31))

    if initial_population is not None:
        population = initial_population
    else:
        population = init_population(pop_size, ga_rng)
    log = []
    overall_best_fitness = -1.0
    overall_best_strategy = None
    opponent_num = 1

    csv_path = config.get("csv_path", "evolution_log.csv")
    csv_append = config.get("csv_append", False)
    csv_file = open(csv_path, "a" if csv_append else "w", newline="")
    writer = csv.writer(csv_file)
    if not csv_append:
        writer.writerow([
            "generation", "best_win_rate", "mean_win_rate", "worst_win_rate",
            "early_to_mid_turn", "mid_to_late_provinces",
            "best_top3_early", "best_top3_mid", "best_top3_late",
            "mean_turns",
        ])

    try:
        for gen in range(start_gen + 1, start_gen + generations + 1):
            # Generate seeds for this generation
            seed_list = make_seed_list(games_per_eval, master_rng)

            # Evaluate all individuals via 2-player games against opponent
            eval_results = evaluate_population(population, seed_list, kingdom,
                                               opponent=opponent,
                                               workers=workers)
            fitnesses = [r["win_rate"] for r in eval_results]

            # Stats
            best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
            best_fitness = fitnesses[best_idx]
            mean_fitness = sum(fitnesses) / pop_size
            worst_fitness = min(fitnesses)
            best_strat = population[best_idx]
            best_turns = eval_results[best_idx]["mean_turns"]

            # Log entry
            entry = {
                "gen": gen,
                "best_win_rate": best_fitness,
                "mean_win_rate": mean_fitness,
                "worst_win_rate": worst_fitness,
                "early_to_mid_turn": best_strat.transitions.early_to_mid_turn,
                "mid_to_late_provinces": best_strat.transitions.mid_to_late_provinces,
                "best_top3_early": best_strat.early_buy_priority[:3],
                "best_top3_mid": best_strat.mid_buy_priority[:3],
                "best_top3_late": best_strat.late_buy_priority[:3],
                "mean_turns": best_turns,
            }
            log.append(entry)

            # CSV
            writer.writerow([
                gen, f"{best_fitness:.3f}", f"{mean_fitness:.3f}", f"{worst_fitness:.3f}",
                best_strat.transitions.early_to_mid_turn,
                best_strat.transitions.mid_to_late_provinces,
                " > ".join(best_strat.early_buy_priority[:3]),
                " > ".join(best_strat.mid_buy_priority[:3]),
                " > ".join(best_strat.late_buy_priority[:3]),
                f"{best_turns:.1f}",
            ])
            csv_file.flush()

            # Console output
            new_best = best_fitness > overall_best_fitness
            if new_best:
                overall_best_fitness = best_fitness
                overall_best_strategy = deepcopy(best_strat)

            line = (f"Gen {gen:3d} | best={best_fitness:5.0%}  mean={mean_fitness:5.0%}"
                    f"  worst={worst_fitness:5.0%} | turns={best_turns:4.1f}"
                    f" | early→mid t{best_strat.transitions.early_to_mid_turn}"
                    f"  mid→late p{best_strat.transitions.mid_to_late_provinces}")
            if new_best:
                line += f"  *** new best vs {opponent_label} ***"
            print(line)

            # Save best model to disk whenever we find a new best
            if new_best:
                vs_stats = {"win_rate": best_fitness,
                            "tie_rate": eval_results[best_idx]["tie_rate"],
                            "loss_rate": eval_results[best_idx]["loss_rate"],
                            "mean_turns": best_turns,
                            "num_games": games_per_eval * 2,
                            "opponent": opponent_label,
                            "avg_final_deck": eval_results[best_idx].get("avg_final_deck")}
                gen_dir = os.path.join(best_model_dir, f"gen_{gen:03d}")
                save_best_model(best_strat, vs_stats, output_dir=gen_dir)
                # Also save as "latest" for easy access
                save_best_model(best_strat, vs_stats, output_dir=best_model_dir)

            # Switch opponent when win rate is high enough
            if (new_best and best_fitness >= switch_threshold
                    and gen < start_gen + generations):
                opponent = deepcopy(overall_best_strategy)
                opponent_num += 1
                opponent_label = f"best_model_v{opponent_num}"
                overall_best_fitness = -1.0
                print(f"  >>> Opponent switched to {opponent_label} "
                      f"(won {best_fitness:.0%} vs previous) <<<")

            # Don't evolve after the last generation
            if gen == start_gen + generations:
                break

            # --- Selection + reproduction ---
            # Sort by fitness for elitism
            ranked = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
            elites = [deepcopy(population[ranked[i]]) for i in range(elite_count)]

            new_pop = list(elites)
            while len(new_pop) < pop_size:
                p1 = tournament_select(population, fitnesses, tournament_size, ga_rng)
                p2 = tournament_select(population, fitnesses, tournament_size, ga_rng)
                child = crossover(p1, p2, ga_rng)
                child = mutate(child, mutation_rate, ga_rng)
                new_pop.append(child)

            population = new_pop

    finally:
        csv_file.close()

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    return {
        "best_strategy": population[best_idx],
        "best_fitness": fitnesses[best_idx],
        "log": log,
        "population": population,
        "fitnesses": fitnesses,
        "opponent": opponent,
        "opponent_label": opponent_label,
    }
