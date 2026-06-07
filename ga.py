"""Genetic algorithm: population management, selection, crossover, mutation."""

from __future__ import annotations

import csv
import os
import random
from copy import deepcopy
from dataclasses import asdict

from cards import BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS
from fitness import (evaluate_population_vs_hall, make_seed_list)
from strategy import (
    Strategy, Transitions, random_strategy, big_money_strategy,
    engine_strategy, gardens_strategy, describe, get_current_phase,
    save_best_model,
)


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def init_population(pop_size: int, rng: random.Random,
                    kingdom: list[str] | None = None) -> list[Strategy]:
    """Create initial population: random strategies + seeded archetypes."""
    population = [big_money_strategy(kingdom), engine_strategy(kingdom),
                  gardens_strategy(kingdom)]
    for _ in range(pop_size - len(population)):
        population.append(random_strategy(rng, kingdom))
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
        early_nonterminal_priority=order_crossover(p1.early_nonterminal_priority, p2.early_nonterminal_priority, rng),
        early_terminal_priority=order_crossover(p1.early_terminal_priority, p2.early_terminal_priority, rng),
        mid_nonterminal_priority=order_crossover(p1.mid_nonterminal_priority, p2.mid_nonterminal_priority, rng),
        mid_terminal_priority=order_crossover(p1.mid_terminal_priority, p2.mid_terminal_priority, rng),
        late_nonterminal_priority=order_crossover(p1.late_nonterminal_priority, p2.late_nonterminal_priority, rng),
        late_terminal_priority=order_crossover(p1.late_terminal_priority, p2.late_terminal_priority, rng),
        early_chapel_trash=["Estate", "Copper", "STOP"],
        mid_chapel_trash=["Estate", "Copper", "STOP"],
        late_chapel_trash=["STOP"],
        throne_room_priority=order_crossover(
            p1.throne_room_priority, p2.throne_room_priority, rng
        ),
        mine_trash_priority=order_crossover(
            p1.mine_trash_priority, p2.mine_trash_priority, rng
        ),
        chapel_max_trash=4,
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
        province_max_coins=rng.choice([p1.province_max_coins, p2.province_max_coins]),
        duchy_max_coins=rng.choice([p1.duchy_max_coins, p2.duchy_max_coins]),
    )


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _mutate_list(lst: list[str], rate: float, rng: random.Random,
                 allow_pass: bool = True) -> list[str]:
    """Mutate a priority list via swap, insertion move, or PASS toggle."""
    lst = list(lst)
    if len(lst) >= 2 and rng.random() < rate:
        if rng.random() < 0.3:
            # Insertion move: pull a card out and reinsert at a random position
            # (can move a card across the entire list in one step)
            i = rng.randrange(len(lst))
            card = lst.pop(i)
            j = rng.randrange(len(lst) + 1)
            lst.insert(j, card)
        else:
            # Swap two positions
            i, j = rng.sample(range(len(lst)), 2)
            lst[i], lst[j] = lst[j], lst[i]

    # PASS insert/remove
    if allow_pass and rng.random() < rate / 5:
        if "PASS" in lst:
            lst.remove("PASS")
        else:
            lst.insert(rng.randint(0, len(lst)), "PASS")

    return lst


def mutate(strategy: Strategy, rate: float, rng: random.Random,
           kingdom: list[str] | None = None) -> Strategy:
    """Apply mutation to a strategy."""
    s = deepcopy(strategy)
    s.early_buy_priority = _mutate_list(s.early_buy_priority, rate, rng)
    s.mid_buy_priority = _mutate_list(s.mid_buy_priority, rate, rng)
    s.late_buy_priority = _mutate_list(s.late_buy_priority, rate, rng)
    s.early_nonterminal_priority = _mutate_list(s.early_nonterminal_priority, rate, rng, allow_pass=False)
    s.early_terminal_priority = _mutate_list(s.early_terminal_priority, rate, rng, allow_pass=False)
    s.mid_nonterminal_priority = _mutate_list(s.mid_nonterminal_priority, rate, rng, allow_pass=False)
    s.mid_terminal_priority = _mutate_list(s.mid_terminal_priority, rate, rng, allow_pass=False)
    s.late_nonterminal_priority = _mutate_list(s.late_nonterminal_priority, rate, rng, allow_pass=False)
    s.late_terminal_priority = _mutate_list(s.late_terminal_priority, rate, rng, allow_pass=False)
    # Chapel trash is hardcoded — no mutation
    s.early_chapel_trash = ["Estate", "Copper", "STOP"]
    s.mid_chapel_trash = ["Estate", "Copper", "STOP"]
    s.late_chapel_trash = ["STOP"]
    s.chapel_max_trash = 4
    s.throne_room_priority = _mutate_list(s.throne_room_priority, rate, rng, allow_pass=False)
    s.mine_trash_priority = _mutate_list(s.mine_trash_priority, rate, rng, allow_pass=False)

    # Jitter transitions (occasional large jumps to escape sticky boundaries)
    if rng.random() < rate:
        delta = rng.choice([-3, -2, -1, -1, 0, 1, 1, 2, 3])
        s.transitions.early_to_mid_turn += delta
        s.transitions.early_to_mid_turn = max(2, min(15, s.transitions.early_to_mid_turn))

    if rng.random() < rate:
        delta = rng.choice([-3, -2, -1, -1, 0, 1, 1, 2, 3])
        s.transitions.mid_to_late_provinces += delta
        s.transitions.mid_to_late_provinces = max(2, min(8, s.transitions.mid_to_late_provinces))

    # Jitter buy targets (min 1 so every card remains purchasable)
    for card in list(s.buy_targets):
        if rng.random() < rate:
            s.buy_targets[card] += rng.choice([-1, 0, 1])
            s.buy_targets[card] = max(1, min(10, s.buy_targets[card]))

    # Occasionally add/remove a target for a kingdom card
    kingdom_cards = kingdom if kingdom is not None else KINGDOM_CARDS
    if rng.random() < rate / 3:
        card = rng.choice(kingdom_cards)
        if card in s.buy_targets:
            del s.buy_targets[card]  # remove limit
        else:
            s.buy_targets[card] = rng.randint(1, 4)  # add limit

    # Jitter coin thresholds for Province/Duchy
    if rng.random() < rate:
        s.province_max_coins += rng.choice([-1, 0, 1])
        s.province_max_coins = max(8, min(18, s.province_max_coins))
    if rng.random() < rate:
        s.duchy_max_coins += rng.choice([-1, 0, 1])
        s.duchy_max_coins = max(5, min(18, s.duchy_max_coins))

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
    best_model_dir = config.get("best_model_dir", "best_model")
    workers = config.get("workers", 1)
    hall_max_size = config.get("hall_max_size", 6)
    hall_add_threshold = config.get("hall_add_threshold", 0.55)
    vp_margin_weight = config.get("vp_margin_weight", 0.0)

    tier2_count = min(config.get("tier2_count", 15), pop_size)  # top N re-evaluated in tier 2
    tier2_seeds = config.get("tier2_seeds", 400)      # seeds for tier 2 (× 2 = 800 games)

    start_gen = config.get("start_gen", 0)
    initial_population = config.get("initial_population")

    master_rng = random.Random(seed)
    ga_rng = random.Random(master_rng.randint(0, 2**31))

    if initial_population is not None:
        population = initial_population
    else:
        population = init_population(pop_size, ga_rng, kingdom)

    # Hall of fame: diverse set of opponents. Big Money is always present.
    hall: list[Strategy] = config.get("hall", [big_money_strategy(kingdom)])

    log = []
    overall_best_fitness = -1.0
    overall_best_strategy = None
    stagnation_count = 0       # generations since last improvement
    STAGNATION_THRESHOLD = 30  # boost mutation after this many stale generations
    STAGNATION_INJECT = 5      # number of random strategies to inject

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

            # Evaluate all individuals against the hall of fame
            eval_results = evaluate_population_vs_hall(
                population, seed_list, kingdom, hall=hall, workers=workers)

            # Blended fitness: win rate + VP margin + speed bonus
            speed_weight = config.get("speed_weight", 0.0)
            def _blended(r):
                f = r["win_rate"]
                if vp_margin_weight > 0:
                    margin_norm = max(0.0, min(1.0, (r["mean_vp_margin"] + 20) / 40))
                    f = (1 - vp_margin_weight - speed_weight) * r["win_rate"] + vp_margin_weight * margin_norm
                else:
                    f = (1 - speed_weight) * r["win_rate"]
                if speed_weight > 0:
                    speed_norm = max(0.0, min(1.0, (25 - r["mean_turns"]) / 10))
                    f += speed_weight * speed_norm
                return f
            fitnesses = [_blended(r) for r in eval_results]

            # --- Tier 2: re-evaluate top candidates with more games ---
            ranked_t1 = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
            tier2_indices = ranked_t1[:tier2_count]
            tier2_seed_list = make_seed_list(tier2_seeds, master_rng)
            tier2_population = [population[i] for i in tier2_indices]
            tier2_results = evaluate_population_vs_hall(
                tier2_population, tier2_seed_list, kingdom, hall=hall, workers=workers)
            tier2_fitnesses = [_blended(r) for r in tier2_results]

            # Use tier 2 to find the true best
            tier2_best_local = max(range(tier2_count), key=lambda i: tier2_fitnesses[i])
            best_idx = tier2_indices[tier2_best_local]
            best_fitness = tier2_fitnesses[tier2_best_local]
            best_eval_result = tier2_results[tier2_best_local]

            # Stats (tier 1 for population-level metrics)
            mean_fitness = sum(fitnesses) / pop_size
            worst_fitness = min(fitnesses)
            best_strat = population[best_idx]
            best_turns = best_eval_result["mean_turns"]

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
                stagnation_count = 0
            else:
                stagnation_count += 1

            line = (f"Gen {gen:3d} | best={best_fitness:5.0%}  mean={mean_fitness:5.0%}"
                    f"  worst={worst_fitness:5.0%} | turns={best_turns:4.1f}"
                    f" | hall={len(hall)}"
                    f" | early→mid t{best_strat.transitions.early_to_mid_turn}"
                    f"  mid→late p{best_strat.transitions.mid_to_late_provinces}")
            if new_best:
                line += "  *** new best ***"
            print(line)

            # Save best model to disk whenever we find a new best
            if new_best:
                vs_stats = {"win_rate": best_eval_result["win_rate"],
                            "tie_rate": best_eval_result["tie_rate"],
                            "loss_rate": best_eval_result["loss_rate"],
                            "mean_turns": best_turns,
                            "num_games": tier2_seeds * 2,
                            "opponent": f"hall({len(hall)})",
                            "avg_final_deck": best_eval_result.get("avg_final_deck")}
                gen_dir = os.path.join(best_model_dir, f"gen_{gen:03d}")
                save_best_model(best_strat, vs_stats, output_dir=gen_dir)
                # Also save as "latest" for easy access
                save_best_model(best_strat, vs_stats, output_dir=best_model_dir)

            # Add to hall of fame when win rate exceeds threshold
            if (new_best and best_fitness >= hall_add_threshold
                    and gen < start_gen + generations):
                new_member = deepcopy(overall_best_strategy)
                hall.append(new_member)
                if len(hall) > hall_max_size:
                    # Remove oldest non-BM member (index 1, since 0 is Big Money)
                    hall.pop(1)
                overall_best_fitness = -1.0  # reset so GA keeps improving vs new hall
                print(f"  >>> Added to hall of fame (hall={len(hall)}, "
                      f"win_rate={best_fitness:.0%}) <<<")

            # Don't evolve after the last generation
            if gen == start_gen + generations:
                break

            # --- Stagnation detection ---
            stagnant = stagnation_count >= STAGNATION_THRESHOLD
            effective_rate = min(mutation_rate * 2.5, 0.5) if stagnant else mutation_rate
            if stagnant and stagnation_count % STAGNATION_THRESHOLD == 0:
                print(f"  >>> Stagnation detected ({stagnation_count} gens), "
                      f"boosting mutation to {effective_rate:.0%} + "
                      f"injecting {STAGNATION_INJECT} randoms <<<")

            # --- Selection + reproduction ---
            # Elites from tier 2 ranking (most reliable fitness estimates)
            tier2_ranked = [tier2_indices[i] for i in
                           sorted(range(tier2_count), key=lambda i: tier2_fitnesses[i], reverse=True)]
            elites = [deepcopy(population[tier2_ranked[i]]) for i in range(elite_count)]

            new_pop = list(elites)

            # Inject random strategies during stagnation
            if stagnant and stagnation_count % STAGNATION_THRESHOLD == 0:
                for _ in range(STAGNATION_INJECT):
                    new_pop.append(random_strategy(ga_rng, kingdom))

            while len(new_pop) < pop_size:
                p1 = tournament_select(population, fitnesses, tournament_size, ga_rng)
                p2 = tournament_select(population, fitnesses, tournament_size, ga_rng)
                child = crossover(p1, p2, ga_rng)
                child = mutate(child, effective_rate, ga_rng, kingdom)
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
        "hall": hall,
    }
