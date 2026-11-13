"""Card facts a player needs (part of the public interface, like the protocol)."""
from __future__ import annotations

from core.cards import ALL_CARDS, CardType

BASE = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]


def known(name: str) -> bool:
    """True if this repo's engine (and so a GA model) knows the card."""
    return name in ALL_CARDS


def is_action(name: str) -> bool:
    return name in ALL_CARDS and ALL_CARDS[name].card_type == CardType.ACTION


def cost(name: str) -> int | None:
    return ALL_CARDS[name].cost if name in ALL_CARDS else None
