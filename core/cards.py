"""Card facts in the shape the genome code uses, taken from the shared engine (engine/)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import core  # noqa: F401  (puts engine/ on sys.path)
from dominion_engine import cards as _engine_cards


class CardType(Enum):
    TREASURE = auto()
    VICTORY = auto()   # also Curse, as far as the genome is concerned
    ACTION = auto()


@dataclass(frozen=True)
class Card:
    name: str
    card_type: CardType
    cost: int
    coins: int = 0
    vp: int = 0
    cards_drawn: int = 0
    actions: int = 0
    buys: int = 0


def _card(c) -> Card:
    t = CardType.ACTION if c.is_action else CardType.TREASURE if c.is_treasure else CardType.VICTORY
    return Card(c.name, t, c.cost, c.coins, c.vp, c.cards, c.actions, c.buys)


ALL_CARDS: dict[str, Card] = {c.name: _card(c) for c in _engine_cards.CARDS}

TREASURE_CARDS = ["Copper", "Silver", "Gold"]
VICTORY_CARDS = ["Estate", "Duchy", "Province"]
KINGDOM_CARDS = list(_engine_cards.KINGDOM_CARDS)   # all 26 of the base set
ACTION_CARDS = [c for c in KINGDOM_CARDS if ALL_CARDS[c].card_type == CardType.ACTION]
NONTERMINAL_ACTIONS = [c for c in ACTION_CARDS if ALL_CARDS[c].actions > 0]
TERMINAL_ACTIONS = [c for c in ACTION_CARDS if ALL_CARDS[c].actions == 0]
ATTACK_CARDS = [c.name for c in _engine_cards.CARDS if c.is_attack]
BUYABLE_CARDS = TREASURE_CARDS + VICTORY_CARDS + KINGDOM_CARDS

CARD_ID = dict(_engine_cards.ID)
CARD_NAME = dict(_engine_cards.NAME)
NUM_CARDS = _engine_cards.NUM_CARDS
