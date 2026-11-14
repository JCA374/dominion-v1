"""Play genomes against each other on the shared Python engine (engine/dominion_engine).

The rules live in the engine; this file only seats GA agents. Training uses the C twin
(ga/c_bridge.py), which plays the same games.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import core  # noqa: F401
from core.agent import GAAgent
from core.cards import KINGDOM_CARDS
from dominion_engine import play
from dominion_engine.cards import NAME

if TYPE_CHECKING:
    from core.strategy import Strategy


def play_game_2p(strategy1: Strategy, strategy2: Strategy, rng_seed: int,
                 kingdom: list[str] | None = None) -> dict:
    """One game, strategy1 moves first. kingdom None = every kingdom card."""
    a1, a2 = GAAgent(strategy1), GAAgent(strategy2)
    r = play(kingdom if kingdom is not None else KINGDOM_CARDS, [a1, a2], rng_seed)
    return {
        "vp1": r.scores[0], "vp2": r.scores[1],
        "turns": r.turns[0],
        "winners": r.winners,          # [0], [1] or [0, 1] (shared)
        "deck1": [NAME[c] for c in r.decks[0]], "deck2": [NAME[c] for c in r.decks[1]],
        "actions1": a1.actions_played, "actions2": a2.actions_played,
        "trace": r.trace,
    }
