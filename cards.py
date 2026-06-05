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

ALL_CARDS: dict[str, Card] = {c.name: c for c in [
    COPPER, SILVER, GOLD,
    ESTATE, DUCHY, PROVINCE,
    VILLAGE, SMITHY, MARKET, LABORATORY, FESTIVAL, CHAPEL,
    THRONE_ROOM, COUNCIL_ROOM, MONEYLENDER, GARDENS,
]}

TREASURE_CARDS = ["Copper", "Silver", "Gold"]
VICTORY_CARDS = ["Estate", "Duchy", "Province"]
KINGDOM_CARDS = [
    "Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel",
    "Throne Room", "Council Room", "Moneylender", "Gardens",
]
ACTION_CARDS = [c for c in KINGDOM_CARDS if ALL_CARDS[c].card_type == CardType.ACTION]

BUYABLE_CARDS = TREASURE_CARDS + VICTORY_CARDS + KINGDOM_CARDS
