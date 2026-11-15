"""Training on the fast engine: engine/c/engine.c + c/ga_agent.c built into dominion.so.

evaluate_vs_opponent_c() is the C path of fitness.evaluate_vs_opponent (same games as the
Python agent, ~50x faster).
"""
from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING

import core  # noqa: F401  (puts engine/ on sys.path)
from core.agent import PASS, STOP, _ids
from core.cards import CARD_ID, KINGDOM_CARDS, NUM_CARDS
from dominion_engine import fast

if TYPE_CHECKING:
    from core.strategy import Strategy

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib = ctypes.CDLL(fast.build(os.path.join(_root, "dominion.so"),
                              [os.path.join(_root, "c", "ga_agent.c")]))

_I = ctypes.POINTER(ctypes.c_int)
_lib.ga_strategy_size.restype = ctypes.c_int
_lib.ga_play_one.argtypes = [_I, _I, ctypes.c_uint64, _I, ctypes.c_int, _I,
                             ctypes.POINTER(ctypes.c_uint64)]
_lib.ga_play_batch.argtypes = [_I, _I, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, _I,
                               ctypes.c_int, _I, _I, _I, _I, _I]

# Genome layout (= c/ga_agent.c)
LIST = 40
STRATEGY_SIZE = _lib.ga_strategy_size()
_BUY, _NT, _T, _CHAPEL = 5, 5 + 3 * LIST, 5 + 4 * LIST, 5 + 5 * LIST
_THRONE, _MINE, _TARGETS = 5 + 8 * LIST, 5 + 9 * LIST, 5 + 10 * LIST
assert _TARGETS + NUM_CARDS == STRATEGY_SIZE


def _write(buf: list[int], offset: int, names: list[str]) -> None:
    ids = _ids(names)[:LIST - 1]
    buf[offset:offset + len(ids)] = ids
    buf[offset + len(ids)] = -1


def strategy_to_ints(strategy: Strategy) -> ctypes.Array:
    buf = [-1] * STRATEGY_SIZE
    t = strategy.transitions
    buf[0:5] = [t.early_to_mid_turn, t.mid_to_late_provinces, t.mid_to_late_turn,
                strategy.chapel_max_trash, strategy.terminal_slack]
    for i, lst in enumerate([strategy.early_buy_priority, strategy.mid_buy_priority,
                             strategy.late_buy_priority]):
        _write(buf, _BUY + i * LIST, lst)
    _write(buf, _NT, strategy.nonterminal_priority)
    _write(buf, _T, strategy.terminal_priority)
    for i, lst in enumerate([strategy.early_chapel_trash, strategy.mid_chapel_trash,
                             strategy.late_chapel_trash]):
        _write(buf, _CHAPEL + i * LIST, lst)
    _write(buf, _THRONE, strategy.throne_room_priority)
    _write(buf, _MINE, strategy.mine_trash_priority)
    for name, n in strategy.buy_targets.items():
        if name in CARD_ID:
            buf[_TARGETS + CARD_ID[name]] = n
    return (ctypes.c_int * STRATEGY_SIZE)(*buf)


def _kingdom(kingdom: list[str] | None) -> tuple[ctypes.Array, int]:
    ids = [CARD_ID[c] for c in (kingdom if kingdom is not None else KINGDOM_CARDS[:10])]
    return (ctypes.c_int * len(ids))(*ids), len(ids)


def play_one_c(s0: Strategy, s1: Strategy, seed: int, kingdom: list[str] | None) -> dict:
    """One game, s0 in seat 0 (for parity tests)."""
    out = (ctypes.c_int * 7)()
    trace = ctypes.c_uint64()
    k, kn = _kingdom(kingdom)
    _lib.ga_play_one(strategy_to_ints(s0), strategy_to_ints(s1), seed, k, kn, out, ctypes.byref(trace))
    return {"scores": [out[0], out[1]], "turns": out[2], "actions": [out[3], out[4]],
            "winners": [i for i in range(2) if out[5] >> i & 1], "capped": bool(out[6] & 1),
            "error": bool(out[6] & 2), "trace": trace.value}


def evaluate_vs_opponent_c(strategy: Strategy, seed_list: list[int],
                           kingdom: list[str] | None = None,
                           opponent: Strategy | None = None) -> dict:
    """Each seed played from both seats. Same result keys as fitness.evaluate_vs_opponent."""
    from core.strategy import big_money_strategy
    opp = opponent if opponent is not None else big_money_strategy()
    k, kn = _kingdom(kingdom)
    n = len(seed_list)
    seeds = (ctypes.c_uint64 * n)(*seed_list)
    vp1, vp2, turns, actions, result = [(ctypes.c_int * (2 * n))() for _ in range(5)]
    _lib.ga_play_batch(strategy_to_ints(strategy), strategy_to_ints(opp), seeds, n, k, kn,
                       vp1, vp2, turns, actions, result)
    games = 2 * n
    wins = sum(1 for r in result if r == 1)
    ties = sum(1 for r in result if r == 0)
    total_turns = sum(turns)
    return {
        "win_rate": wins / games,
        "tie_rate": ties / games,
        "loss_rate": (games - wins - ties) / games,
        "mean_turns": total_turns / games,
        "mean_vp_margin": (sum(vp1) - sum(vp2)) / games,
        "mean_actions_per_turn": sum(actions) / total_turns if total_turns else 0.0,
        "avg_final_deck": None,
    }


_lib.ga_search_batch.argtypes = [_I, _I, _I, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, _I,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_uint64, ctypes.c_int, _I, _I]


def search_vs(strategy: Strategy, opponent: Strategy, seed_list: list[int],
              kingdom: list[str] | None = None, rollouts: int = 50, top: int = 3,
              margin: int = 0, opp_model: Strategy | None = None, search_seed: int = 1,
              imagine: bool = False) -> dict:
    """Rollout search on top of `strategy` (c/ga_agent.c search_decide) against plain `opponent`,
    each seed from both seats. opp_model: the search's model of the opponent (default: the
    opponent itself). imagine: the search doesn't see the opponent's cards and imagines them as a
    mirror of its own deck (as on Steam). Score = (wins + shared/2) / games."""
    k, kn = _kingdom(kingdom)
    n = len(seed_list)
    seeds = (ctypes.c_uint64 * n)(*seed_list)
    result, stats = (ctypes.c_int * (2 * n))(), (ctypes.c_int * 2)()
    _lib.ga_search_batch(strategy_to_ints(strategy), strategy_to_ints(opp_model or opponent),
                         strategy_to_ints(opponent), seeds, n, k, kn, rollouts, top, margin,
                         search_seed, int(imagine), result, stats)
    wins = sum(1 for r in result if r == 1)
    ties = sum(1 for r in result if r == 0)
    return {"score": (wins + ties / 2) / (2 * n), "games": 2 * n,
            "searched": stats[0], "changed": stats[1]}


_lib.ga_search_buy.restype = ctypes.c_int
_lib.ga_search_buy.argtypes = [_I, _I, _I, ctypes.c_int, _I, ctypes.c_int, _I, ctypes.c_int, _I,
                               ctypes.c_int, _I, ctypes.c_int, _I, ctypes.c_int, _I, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_uint64, _I]


def search_buy(strategy: Strategy, opp_model: Strategy, supply: list[int], me: int,
               hand: list[int], play: list[int], deck: list[int], discard: list[int], opp_cards: list[int],
               my_turns: int, opp_turns: int, coins: int, buys: int, opp_discard: list[int] = (),
               rollouts: int = 100,
               top: int = 3, margin: int = 5, seed: int = 1) -> tuple[int, dict]:
    """One buy chosen by rollout search from a player's view (card ids; deck = my draw pile, order
    unknown; discard = my discard pile; opp_cards = the opponent's cards as tracked). -1 = buy nothing."""
    arr = lambda xs: (ctypes.c_int * max(1, len(xs)))(*xs)
    stats = (ctypes.c_int * 2)()
    c = _lib.ga_search_buy(strategy_to_ints(strategy), strategy_to_ints(opp_model), arr(supply), me,
                           arr(hand), len(hand), arr(play), len(play), arr(deck), len(deck),
                           arr(discard), len(discard),
                           arr(opp_cards), len(opp_cards), arr(list(opp_discard)), len(opp_discard),
                           my_turns, opp_turns, coins, buys,
                           rollouts, top, margin, seed & ((1 << 64) - 1), stats)
    return c, {"searched": stats[0], "changed": stats[1]}
