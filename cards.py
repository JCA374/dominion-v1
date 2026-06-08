"""Card definitions for simplified Dominion."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CardType(Enum):
    TREASURE = auto()
    VICTORY = auto()
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
    special: Optional[str] = None


# --- Card Definitions ---

COPPER = Card("Copper", CardType.TREASURE, cost=0, coins=1)
SILVER = Card("Silver", CardType.TREASURE, cost=3, coins=2)
GOLD = Card("Gold", CardType.TREASURE, cost=6, coins=3)

ESTATE = Card("Estate", CardType.VICTORY, cost=2, vp=1)
DUCHY = Card("Duchy", CardType.VICTORY, cost=5, vp=3)
PROVINCE = Card("Province", CardType.VICTORY, cost=8, vp=6)

VILLAGE = Card("Village", CardType.ACTION, cost=3, cards_drawn=1, actions=2)
SMITHY = Card("Smithy", CardType.ACTION, cost=4, cards_drawn=3)
MARKET = Card("Market", CardType.ACTION, cost=5, cards_drawn=1, actions=1, buys=1, coins=1)
LABORATORY = Card("Laboratory", CardType.ACTION, cost=5, cards_drawn=2, actions=1)
FESTIVAL = Card("Festival", CardType.ACTION, cost=5, actions=2, buys=1, coins=2)
CHAPEL = Card("Chapel", CardType.ACTION, cost=2, special="chapel")
THRONE_ROOM = Card("Throne Room", CardType.ACTION, cost=4, special="throne_room")
COUNCIL_ROOM = Card("Council Room", CardType.ACTION, cost=5, cards_drawn=4, buys=1)
MONEYLENDER = Card("Moneylender", CardType.ACTION, cost=4, special="moneylender")
GARDENS = Card("Gardens", CardType.VICTORY, cost=4, special="gardens")
MINE = Card("Mine", CardType.ACTION, cost=5, special="mine")
MERCHANT = Card("Merchant", CardType.ACTION, cost=3, cards_drawn=1, actions=1, special="merchant")

ALL_CARDS: dict[str, Card] = {c.name: c for c in [
    COPPER, SILVER, GOLD,
    ESTATE, DUCHY, PROVINCE,
    VILLAGE, SMITHY, MARKET, LABORATORY, FESTIVAL, CHAPEL,
    THRONE_ROOM, COUNCIL_ROOM, MONEYLENDER, GARDENS,
    MINE, MERCHANT,
]}

TREASURE_CARDS = ["Copper", "Silver", "Gold"]
VICTORY_CARDS = ["Estate", "Duchy", "Province"]
KINGDOM_CARDS = [
    "Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel",
    "Throne Room", "Council Room", "Moneylender", "Gardens",
    "Mine", "Merchant",
]
ACTION_CARDS = [c for c in KINGDOM_CARDS if ALL_CARDS[c].card_type == CardType.ACTION]
NONTERMINAL_ACTIONS = [c for c in ACTION_CARDS if ALL_CARDS[c].actions > 0]
TERMINAL_ACTIONS = [c for c in ACTION_CARDS if ALL_CARDS[c].actions == 0]

BUYABLE_CARDS = TREASURE_CARDS + VICTORY_CARDS + KINGDOM_CARDS

# --- Integer Card IDs (for C engine) ---

CARD_ID = {
    "Copper": 0, "Silver": 1, "Gold": 2,
    "Estate": 3, "Duchy": 4, "Province": 5,
    "Village": 6, "Smithy": 7, "Market": 8, "Laboratory": 9,
    "Festival": 10, "Chapel": 11, "Throne Room": 12,
    "Council Room": 13, "Moneylender": 14, "Gardens": 15,
    "Mine": 16, "Merchant": 17,
}
NUM_CARDS = 18
PASS_ID = 18
STOP_ID = 19
CARD_NAME = {v: k for k, v in CARD_ID.items()}
CARD_NAME[PASS_ID] = "PASS"
CARD_NAME[STOP_ID] = "STOP"

# Special card codes
SPECIAL_NONE = 0
SPECIAL_CHAPEL = 1
SPECIAL_THRONE_ROOM = 2
SPECIAL_MONEYLENDER = 3
SPECIAL_GARDENS = 4
SPECIAL_MINE = 5
SPECIAL_MERCHANT = 6

_SPECIAL_MAP = {
    None: SPECIAL_NONE,
    "chapel": SPECIAL_CHAPEL,
    "throne_room": SPECIAL_THRONE_ROOM,
    "moneylender": SPECIAL_MONEYLENDER,
    "gardens": SPECIAL_GARDENS,
    "mine": SPECIAL_MINE,
    "merchant": SPECIAL_MERCHANT,
}

# Card type codes
TYPE_TREASURE = 1
TYPE_VICTORY = 2
TYPE_ACTION = 3

_TYPE_MAP = {
    CardType.TREASURE: TYPE_TREASURE,
    CardType.VICTORY: TYPE_VICTORY,
    CardType.ACTION: TYPE_ACTION,
}

# Flat data arrays indexed by card ID (for C engine)
_ordered = [ALL_CARDS[CARD_NAME[i]] for i in range(NUM_CARDS)]
CARD_COST = [c.cost for c in _ordered]
CARD_COINS = [c.coins for c in _ordered]
CARD_VP = [c.vp for c in _ordered]
CARD_DRAW = [c.cards_drawn for c in _ordered]
CARD_ACTIONS = [c.actions for c in _ordered]
CARD_BUYS = [c.buys for c in _ordered]
CARD_TYPE_ID = [_TYPE_MAP[c.card_type] for c in _ordered]
CARD_SPECIAL_ID = [_SPECIAL_MAP[c.special] for c in _ordered]
