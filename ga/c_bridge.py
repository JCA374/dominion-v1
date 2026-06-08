"""Python-C bridge for the fast Dominion engine.

Loads dominion.so via ctypes and provides drop-in replacements for
evaluate_vs_opponent() that run ~50-100x faster than pure Python.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import TYPE_CHECKING

from core.cards import (
    ALL_CARDS, CARD_ID, NUM_CARDS, PASS_ID, STOP_ID,
    CARD_COST, CARD_COINS, CARD_VP, CARD_DRAW,
    CARD_ACTIONS, CARD_BUYS, CARD_TYPE_ID, CARD_SPECIAL_ID,
    KINGDOM_CARDS,
)

if TYPE_CHECKING:
    from core.strategy import Strategy

# ── Build and load the shared library ──

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
_so_path = os.path.join(_root, "dominion.so")
_c_path = os.path.join(_root, "c", "dominion.c")


def _ensure_built():
    """Auto-build dominion.so if missing or stale."""
    if not os.path.exists(_so_path) or \
       os.path.getmtime(_c_path) > os.path.getmtime(_so_path):
        subprocess.check_call(["make", "-C", os.path.join(_root, "c")],
                              stdout=subprocess.DEVNULL)


_ensure_built()
_lib = ctypes.CDLL(_so_path)

# ── Set up function signatures ──

_lib.init_cards.argtypes = [ctypes.POINTER(ctypes.c_int)]
_lib.init_cards.restype = None

_lib.play_games_batch.argtypes = [
    ctypes.POINTER(ctypes.c_int),      # strat1
    ctypes.POINTER(ctypes.c_int),      # strat2
    ctypes.POINTER(ctypes.c_uint64),   # seeds
    ctypes.c_int,                      # num_games
    ctypes.POINTER(ctypes.c_int),      # kingdom_ids
    ctypes.c_int,                      # kingdom_n
    ctypes.POINTER(ctypes.c_int),      # out_vp1
    ctypes.POINTER(ctypes.c_int),      # out_vp2
    ctypes.POINTER(ctypes.c_int),      # out_turns
]
_lib.play_games_batch.restype = None

# ── Initialize card data once ──

_card_data = (ctypes.c_int * (NUM_CARDS * 8))()
for _name, _cid in CARD_ID.items():
    if _cid >= NUM_CARDS:
        continue
    _base = _cid * 8
    _card_data[_base + 0] = CARD_COST[_cid]
    _card_data[_base + 1] = CARD_COINS[_cid]
    _card_data[_base + 2] = CARD_VP[_cid]
    _card_data[_base + 3] = CARD_DRAW[_cid]
    _card_data[_base + 4] = CARD_ACTIONS[_cid]
    _card_data[_base + 5] = CARD_BUYS[_cid]
    _card_data[_base + 6] = CARD_TYPE_ID[_cid]
    _card_data[_base + 7] = CARD_SPECIAL_ID[_cid]
_lib.init_cards(_card_data)


# ── Strategy serialization ──

# Strategy layout constants (must match dominion.c)
STRATEGY_SIZE = 191


def _name_to_id(name: str) -> int:
    """Convert card name to integer ID, handling PASS and STOP."""
    if name == "PASS":
        return PASS_ID
    if name == "STOP":
        return STOP_ID
    return CARD_ID[name]


def _write_list(buf: list[int], offset: int, names: list[str], max_len: int) -> None:
    """Write a -1 terminated list of card IDs into buffer at offset."""
    i = 0
    for name in names:
        if i >= max_len - 1:  # leave room for sentinel
            break
        buf[offset + i] = _name_to_id(name)
        i += 1
    buf[offset + i] = -1


def strategy_to_ints(strategy: Strategy) -> ctypes.Array:
    """Serialize a Strategy to a flat int array for the C engine."""
    buf = [0] * STRATEGY_SIZE

    # Scalar fields
    buf[0] = strategy.transitions.early_to_mid_turn
    buf[1] = strategy.transitions.mid_to_late_provinces
    buf[2] = strategy.chapel_max_trash
    buf[3] = strategy.province_max_coins
    buf[4] = strategy.duchy_max_coins

    # Priority lists
    _write_list(buf, 5, strategy.early_buy_priority, 20)
    _write_list(buf, 25, strategy.mid_buy_priority, 20)
    _write_list(buf, 45, strategy.late_buy_priority, 20)
    _write_list(buf, 65, strategy.early_nonterminal_priority, 12)
    _write_list(buf, 77, strategy.early_terminal_priority, 12)
    _write_list(buf, 89, strategy.mid_nonterminal_priority, 12)
    _write_list(buf, 101, strategy.mid_terminal_priority, 12)
    _write_list(buf, 113, strategy.late_nonterminal_priority, 12)
    _write_list(buf, 125, strategy.late_terminal_priority, 12)
    _write_list(buf, 137, strategy.early_chapel_trash, 6)
    _write_list(buf, 143, strategy.mid_chapel_trash, 6)
    _write_list(buf, 149, strategy.late_chapel_trash, 6)
    _write_list(buf, 155, strategy.throne_room_priority, 12)
    _write_list(buf, 167, strategy.mine_trash_priority, 4)

    # Buy targets: pairs of (card_id, max_count), -1 terminated
    offset = 171
    for card_name, max_count in strategy.buy_targets.items():
        if card_name in CARD_ID and offset < 189:
            buf[offset] = CARD_ID[card_name]
            buf[offset + 1] = max_count
            offset += 2
    buf[offset] = -1

    arr = (ctypes.c_int * STRATEGY_SIZE)(*buf)
    return arr


# ── Kingdom serialization ──

def _kingdom_to_ids(kingdom: list[str] | None) -> tuple[ctypes.Array, int]:
    """Convert kingdom card names to a ctypes int array."""
    if kingdom is None:
        kingdom = KINGDOM_CARDS
    ids = [CARD_ID[name] for name in kingdom]
    arr = (ctypes.c_int * len(ids))(*ids)
    return arr, len(ids)


# ── Drop-in replacement for evaluate_vs_opponent ──

def evaluate_vs_opponent_c(strategy: Strategy, seed_list: list[int],
                           kingdom: list[str] | None = None,
                           opponent: Strategy | None = None) -> dict:
    """Play 2-player games against an opponent using the C engine.

    Same interface as fitness.evaluate_vs_opponent but ~50-100x faster.
    Does not return avg_final_deck (returns None for that field).
    """
    from core.strategy import big_money_strategy
    opp = opponent if opponent is not None else big_money_strategy()

    strat1 = strategy_to_ints(strategy)
    strat2 = strategy_to_ints(opp)
    kingdom_arr, kingdom_n = _kingdom_to_ids(kingdom)

    num_seeds = len(seed_list)
    seeds_arr = (ctypes.c_uint64 * num_seeds)(*seed_list)

    # Output arrays: 2 results per seed (seat swap)
    n_results = num_seeds * 2
    out_vp1 = (ctypes.c_int * n_results)()
    out_vp2 = (ctypes.c_int * n_results)()
    out_turns = (ctypes.c_int * n_results)()

    _lib.play_games_batch(strat1, strat2, seeds_arr, num_seeds,
                          kingdom_arr, kingdom_n,
                          out_vp1, out_vp2, out_turns)

    # Aggregate results
    wins = 0
    ties = 0
    total_turns = 0
    total_vp_margin = 0

    for i in range(n_results):
        vp1 = out_vp1[i]
        vp2 = out_vp2[i]
        total_turns += out_turns[i]
        total_vp_margin += vp1 - vp2
        if vp1 > vp2:
            wins += 1
        elif vp1 == vp2:
            ties += 1

    return {
        "win_rate": wins / n_results,
        "tie_rate": ties / n_results,
        "loss_rate": (n_results - wins - ties) / n_results,
        "mean_turns": total_turns / n_results,
        "mean_vp_margin": total_vp_margin / n_results,
        "avg_final_deck": None,
    }
