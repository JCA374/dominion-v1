"""Fitness evaluation for strategies."""

from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING

from engine import play_game, play_game_2p
from strategy import big_money_strategy

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
                         opponent: Strategy | None = None) -> dict:
    """Play 2-player games against an opponent. Return win rate.

    If opponent is None, defaults to Big Money.
    Each seed is played twice (strategy as P1 and as P2) to cancel
    first-player advantage.
    """
    opp = opponent if opponent is not None else big_money_strategy()
    wins = 0
    ties = 0
    total_turns = 0
    for s in seed_list:
        # Game 1: strategy goes first
        r1 = play_game_2p(strategy, opp, s, kingdom)
        total_turns += r1["turns"]
        if r1["vp1"] > r1["vp2"]:
            wins += 1
        elif r1["vp1"] == r1["vp2"]:
            ties += 1

        # Game 2: strategy goes second
        r2 = play_game_2p(opp, strategy, s, kingdom)
        total_turns += r2["turns"]
        if r2["vp2"] > r2["vp1"]:
            wins += 1
        elif r2["vp2"] == r2["vp1"]:
            ties += 1

    n = len(seed_list) * 2  # twice as many games
    return {
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": (n - wins - ties) / n,
        "mean_turns": total_turns / n,
    }


# Backwards-compatible alias
evaluate_vs_big_money = evaluate_vs_opponent
