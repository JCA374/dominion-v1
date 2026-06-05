"""Game engine for simplified Dominion."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cards import ALL_CARDS, CardType, KINGDOM_CARDS

if TYPE_CHECKING:
    from strategy import Strategy


@dataclass
class GameState:
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    play_area: list[str] = field(default_factory=list)
    trash: list[str] = field(default_factory=list)
    supply: dict[str, int] = field(default_factory=dict)
    actions: int = 0
    buys: int = 0
    coins: int = 0
    turn: int = 0
    rng: random.Random = field(default_factory=random.Random)


def default_supply(kingdom: list[str] | None = None,
                    num_players: int = 1) -> dict[str, int]:
    """Standard supply counts. Adjusts Estate/Copper for number of players."""
    if kingdom is None:
        kingdom = KINGDOM_CARDS
    starting_coppers = 7 * num_players
    starting_estates = 3 * num_players
    supply = {
        "Copper": 60 - starting_coppers,
        "Silver": 40,
        "Gold": 30,
        "Estate": 12 - starting_estates,
        "Duchy": 12,
        "Province": 12,
    }
    for name in kingdom:
        supply[name] = 10
    return supply


def new_game(rng_seed: int, kingdom: list[str] | None = None) -> GameState:
    """Create initial game state: 7 Copper + 3 Estate, shuffled, draw 5."""
    state = GameState()
    state.rng = random.Random(rng_seed)
    state.supply = default_supply(kingdom)
    state.deck = ["Copper"] * 7 + ["Estate"] * 3
    state.rng.shuffle(state.deck)
    draw_cards(state, 5)
    return state


def draw_cards(state: GameState, n: int) -> None:
    """Draw n cards from deck to hand. Reshuffle discard into deck if needed."""
    for _ in range(n):
        if not state.deck:
            if not state.discard:
                return  # nothing left to draw
            state.deck = state.discard
            state.discard = []
            state.rng.shuffle(state.deck)
        state.hand.append(state.deck.pop())


def play_action_phase(state: GameState, strategy: Strategy) -> None:
    """Play action cards from hand following action_priority while actions > 0."""
    while state.actions > 0:
        played = False
        for card_name in strategy.action_priority:
            if card_name in state.hand and state.actions > 0:
                state.hand.remove(card_name)
                state.play_area.append(card_name)
                state.actions -= 1

                card = ALL_CARDS[card_name]
                state.actions += card.actions
                state.coins += card.coins
                state.buys += card.buys
                draw_cards(state, card.cards_drawn)

                if card.special == "chapel":
                    play_chapel(state, strategy)

                played = True
                break  # re-scan from top of priority list
        if not played:
            break


def play_chapel(state: GameState, strategy: Strategy) -> None:
    """Trash up to 4 cards from hand following chapel_trash_priority."""
    trashed = 0
    for card_name in strategy.chapel_trash_priority:
        if card_name == "STOP":
            break
        while card_name in state.hand and trashed < 4:
            state.hand.remove(card_name)
            state.trash.append(card_name)
            trashed += 1


def play_buy_phase(state: GameState, strategy: Strategy) -> None:
    """Auto-play all treasures, then buy cards following phase buy priority."""
    # Auto-play treasures from hand
    treasures = [c for c in state.hand if ALL_CARDS[c].card_type == CardType.TREASURE]
    for t in treasures:
        state.hand.remove(t)
        state.play_area.append(t)
        state.coins += ALL_CARDS[t].coins

    # Determine current phase
    from strategy import get_current_phase, get_buy_priority
    phase = get_current_phase(state.turn, state.supply["Province"], strategy.transitions)
    buy_priority = get_buy_priority(strategy, phase)

    # Buy cards
    while state.buys > 0:
        bought = False
        for card_name in buy_priority:
            if card_name == "PASS":
                state.buys = 0
                break
            if (card_name in state.supply
                    and state.supply[card_name] > 0
                    and ALL_CARDS[card_name].cost <= state.coins):
                state.supply[card_name] -= 1
                state.discard.append(card_name)
                state.coins -= ALL_CARDS[card_name].cost
                state.buys -= 1
                bought = True
                break  # re-scan from top of priority list
        if not bought:
            break


def cleanup(state: GameState) -> None:
    """Discard hand + play area, draw 5 new cards."""
    state.discard.extend(state.hand)
    state.discard.extend(state.play_area)
    state.hand.clear()
    state.play_area.clear()
    draw_cards(state, 5)


def is_game_over(state: GameState, turn_cap: int = 40) -> bool:
    """Check end conditions: Province empty, 3 piles empty, or turn cap."""
    if state.supply.get("Province", 0) == 0:
        return True
    empty_piles = sum(1 for v in state.supply.values() if v == 0)
    if empty_piles >= 3:
        return True
    if state.turn >= turn_cap:
        return True
    return False


def count_vp(state: GameState) -> int:
    """Sum VP across all cards the player owns."""
    all_cards = state.deck + state.hand + state.discard + state.play_area
    return sum(ALL_CARDS[c].vp for c in all_cards)


def play_game(strategy: Strategy, rng_seed: int,
              kingdom: list[str] | None = None) -> dict:
    """Run a full game, return results dict."""
    state = new_game(rng_seed, kingdom)

    while not is_game_over(state):
        state.turn += 1
        state.actions = 1
        state.buys = 1
        state.coins = 0

        play_action_phase(state, strategy)
        play_buy_phase(state, strategy)
        cleanup(state)

    return {
        "vp": count_vp(state),
        "turns": state.turn,
    }


def _new_player(rng: random.Random) -> GameState:
    """Create a player state with starting deck, no supply (shared externally)."""
    state = GameState()
    state.rng = rng
    state.deck = ["Copper"] * 7 + ["Estate"] * 3
    state.rng.shuffle(state.deck)
    draw_cards(state, 5)
    return state


def play_game_2p(strategy1: Strategy, strategy2: Strategy, rng_seed: int,
                 kingdom: list[str] | None = None) -> dict:
    """Run a 2-player game with shared supply. Players alternate turns."""
    rng = random.Random(rng_seed)
    supply = default_supply(kingdom, num_players=2)

    # Each player gets their own RNG derived from the master
    p1 = _new_player(random.Random(rng.randint(0, 2**31)))
    p2 = _new_player(random.Random(rng.randint(0, 2**31)))
    p1.supply = supply  # shared reference — both see the same pile
    p2.supply = supply

    strategies = [strategy1, strategy2]
    players = [p1, p2]
    round_num = 0

    while True:
        round_num += 1
        for player, strategy in zip(players, strategies):
            # Check game end before each player's turn
            if is_game_over(player, turn_cap=40):
                def _all_cards(p):
                    return p.deck + p.hand + p.discard + p.play_area
                return {
                    "vp1": count_vp(p1),
                    "vp2": count_vp(p2),
                    "turns": round_num - 1,
                    "deck1": _all_cards(p1),
                    "deck2": _all_cards(p2),
                }

            player.turn = round_num  # per-player turn (for phase transitions)
            player.actions = 1
            player.buys = 1
            player.coins = 0

            play_action_phase(player, strategy)
            play_buy_phase(player, strategy)
            cleanup(player)
