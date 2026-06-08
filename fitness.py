"""Fitness evaluation for strategies."""

from __future__ import annotations

import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

from engine import play_game, play_game_2p
from strategy import big_money_strategy

try:
    from c_bridge import evaluate_vs_opponent_c
    USE_C_ENGINE = True
except (ImportError, OSError) as e:
    USE_C_ENGINE = False

if TYPE_CHECKING:
    from strategy import Strategy


def make_seed_list(num_games: int, master_rng: random.Random) -> list[int]:
    """Generate a fixed list of RNG seeds for one generation's evaluations."""
    return [master_rng.randint(0, 2**31) for _ in range(num_games)]


def evaluate(strategy: Strategy, seed_list: list[int],
             kingdom: list[str] | None = None) -> dict:
    """Run len(seed_list) games, return fitness stats."""
    results = [play_game(strategy, s, kingdom) for s in seed_list]
    vps = [r["vp"] for r in results]
    turns = [r["turns"] for r in results]

    return {
        "mean_vp": statistics.mean(vps),
        "std_vp": statistics.stdev(vps) if len(vps) > 1 else 0.0,
        "mean_turns": statistics.mean(turns),
        "min_vp": min(vps),
        "max_vp": max(vps),
    }


def evaluate_vs_opponent(strategy: Strategy, seed_list: list[int],
                         kingdom: list[str] | None = None,
                         opponent: Strategy | None = None,
                         need_deck: bool = False) -> dict:
    """Play 2-player games against an opponent. Return win rate.

    If opponent is None, defaults to Big Money.
    Each seed is played twice (strategy as P1 and as P2) to cancel
    first-player advantage.

    Uses the C engine when available (much faster) unless need_deck=True.
    """
    if USE_C_ENGINE and not need_deck:
        return evaluate_vs_opponent_c(strategy, seed_list, kingdom, opponent)

    opp = opponent if opponent is not None else big_money_strategy()
    wins = 0
    ties = 0
    total_turns = 0
    total_vp_margin = 0
    deck_counts: dict[str, int] = {}
    for s in seed_list:
        # Game 1: strategy goes first
        r1 = play_game_2p(strategy, opp, s, kingdom)
        total_turns += r1["turns"]
        total_vp_margin += r1["vp1"] - r1["vp2"]
        if r1["vp1"] > r1["vp2"]:
            wins += 1
        elif r1["vp1"] == r1["vp2"]:
            ties += 1
        for card in r1["deck1"]:
            deck_counts[card] = deck_counts.get(card, 0) + 1

        # Game 2: strategy goes second
        r2 = play_game_2p(opp, strategy, s, kingdom)
        total_turns += r2["turns"]
        total_vp_margin += r2["vp2"] - r2["vp1"]
        if r2["vp2"] > r2["vp1"]:
            wins += 1
        elif r2["vp2"] == r2["vp1"]:
            ties += 1
        for card in r2["deck2"]:
            deck_counts[card] = deck_counts.get(card, 0) + 1

    n = len(seed_list) * 2  # twice as many games
    avg_deck = {card: round(count / n, 1) for card, count in
                sorted(deck_counts.items(), key=lambda x: -x[1])}
    return {
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": (n - wins - ties) / n,
        "mean_turns": total_turns / n,
        "mean_vp_margin": total_vp_margin / n,
        "avg_final_deck": avg_deck,
    }


def _eval_one(strategy: Strategy, seed_list: list[int],
              kingdom: list[str] | None, opponent: Strategy | None) -> dict:
    """Evaluate a single strategy — top-level function for multiprocessing."""
    return evaluate_vs_opponent(strategy, seed_list, kingdom, opponent=opponent)


def evaluate_population(population: list[Strategy], seed_list: list[int],
                        kingdom: list[str] | None = None,
                        opponent: Strategy | None = None,
                        workers: int = 1) -> list[dict]:
    """Evaluate all strategies, optionally in parallel.

    workers=1 means sequential, workers>1 uses a process pool.
    """
    if workers <= 1:
        return [evaluate_vs_opponent(s, seed_list, kingdom, opponent=opponent)
                for s in population]

    fn = partial(_eval_one, seed_list=seed_list, kingdom=kingdom,
                 opponent=opponent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, population))


def evaluate_vs_hall(strategy: Strategy, seed_list: list[int],
                     hall: list[Strategy],
                     kingdom: list[str] | None = None,
                     need_deck: bool = False) -> dict:
    """Evaluate strategy against each hall member, return averaged stats.

    Divides seeds across hall members so total game count stays manageable.
    """
    if not hall:
        return evaluate_vs_opponent(strategy, seed_list, kingdom, opponent=None,
                                    need_deck=need_deck)

    # Divide seeds among hall members (min 4 per opponent)
    seeds_per_opp = max(4, len(seed_list) // len(hall))

    all_results = []
    for i, opponent in enumerate(hall):
        # Use a slice of seeds for this opponent
        start = (i * seeds_per_opp) % len(seed_list)
        opp_seeds = seed_list[start:start + seeds_per_opp]
        if len(opp_seeds) < seeds_per_opp:
            opp_seeds += seed_list[:seeds_per_opp - len(opp_seeds)]
        result = evaluate_vs_opponent(strategy, opp_seeds, kingdom, opponent=opponent,
                                      need_deck=need_deck)
        all_results.append(result)

    # Average across all opponents
    avg_win = sum(r["win_rate"] for r in all_results) / len(all_results)
    avg_tie = sum(r["tie_rate"] for r in all_results) / len(all_results)
    avg_loss = sum(r["loss_rate"] for r in all_results) / len(all_results)
    avg_turns = sum(r["mean_turns"] for r in all_results) / len(all_results)
    avg_vp_margin = sum(r["mean_vp_margin"] for r in all_results) / len(all_results)
    return {
        "win_rate": avg_win,
        "tie_rate": avg_tie,
        "loss_rate": avg_loss,
        "mean_turns": avg_turns,
        "mean_vp_margin": avg_vp_margin,
        "avg_final_deck": all_results[0].get("avg_final_deck"),
    }


def _eval_one_vs_hall(strategy: Strategy, seed_list: list[int],
                      kingdom: list[str] | None,
                      hall: list[Strategy]) -> dict:
    """Evaluate a single strategy vs hall — top-level for multiprocessing."""
    return evaluate_vs_hall(strategy, seed_list, hall, kingdom)


def evaluate_population_vs_hall(population: list[Strategy],
                                seed_list: list[int],
                                kingdom: list[str] | None = None,
                                hall: list[Strategy] | None = None,
                                workers: int = 1) -> list[dict]:
    """Evaluate all strategies against a hall of fame, optionally in parallel."""
    if not hall:
        return evaluate_population(population, seed_list, kingdom, workers=workers)

    if workers <= 1:
        return [evaluate_vs_hall(s, seed_list, hall, kingdom) for s in population]

    fn = partial(_eval_one_vs_hall, seed_list=seed_list, kingdom=kingdom,
                 hall=hall)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, population))


# Backwards-compatible alias
evaluate_vs_big_money = evaluate_vs_opponent
