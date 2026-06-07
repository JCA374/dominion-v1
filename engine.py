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
        if ALL_CARDS[name].card_type == CardType.VICTORY:
            supply[name] = 12  # victory kingdom cards match base victory count
        else:
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


# ---------------------------------------------------------------------------
# Low-level primitives — used by both GA strategy-driven and interactive play
# ---------------------------------------------------------------------------

def resolve_action(state: GameState, card_name: str) -> list[str]:
    """Play a single action card: apply all effects, return newly drawn cards.

    Handles everything except chapel trashing — the caller decides how to
    handle that (strategy-driven or interactive).
    """
    state.hand.remove(card_name)
    state.play_area.append(card_name)
    state.actions -= 1

    card = ALL_CARDS[card_name]
    state.actions += card.actions
    state.coins += card.coins
    state.buys += card.buys

    hand_before_len = len(state.hand)
    draw_cards(state, card.cards_drawn)
    newly_drawn = state.hand[hand_before_len:]

    return list(newly_drawn)


def apply_action_effects(state: GameState, card_name: str) -> list[str]:
    """Apply a card's action effects (draw/actions/coins/buys) without moving
    the card or costing an action. Used for Throne Room's second play.
    Returns list of newly drawn cards."""
    card = ALL_CARDS[card_name]
    state.actions += card.actions
    state.coins += card.coins
    state.buys += card.buys

    hand_before_len = len(state.hand)
    draw_cards(state, card.cards_drawn)
    return list(state.hand[hand_before_len:])


def auto_play_treasures(state: GameState) -> list[str]:
    """Move all treasures from hand to play area, add coins. Return names played.

    Applies Merchant bonus: +$1 per Merchant in play area for the first Silver played.
    """
    treasures = [c for c in state.hand if ALL_CARDS[c].card_type == CardType.TREASURE]
    for t in treasures:
        state.hand.remove(t)
        state.play_area.append(t)
        state.coins += ALL_CARDS[t].coins
    # Merchant bonus: each Merchant in play grants +$1 on the first Silver
    if "Silver" in treasures:
        merchant_count = sum(1 for c in state.play_area if c == "Merchant")
        state.coins += merchant_count
    return treasures


def buy_card(state: GameState, card_name: str) -> None:
    """Buy a single card from supply. Caller must verify affordability/supply."""
    state.supply[card_name] -= 1
    state.discard.append(card_name)
    state.coins -= ALL_CARDS[card_name].cost
    state.buys -= 1


def trash_card(state: GameState, card_name: str) -> None:
    """Trash a single card from hand."""
    state.hand.remove(card_name)
    state.trash.append(card_name)


# ---------------------------------------------------------------------------
# Strategy-driven phases — used by GA / AI opponents
# ---------------------------------------------------------------------------

def _play_action_tier(state: GameState, strategy: Strategy,
                      priority: list[str], phase: str) -> None:
    """Play actions from a single tier (nonterminal or terminal)."""
    while state.actions > 0:
        played = False
        for card_name in priority:
            if card_name in state.hand and state.actions > 0:
                resolve_action(state, card_name)

                if ALL_CARDS[card_name].special == "chapel":
                    play_chapel(state, strategy, phase)
                elif ALL_CARDS[card_name].special == "moneylender":
                    play_moneylender(state)
                elif ALL_CARDS[card_name].special == "throne_room":
                    play_throne_room(state, strategy, phase)
                elif ALL_CARDS[card_name].special == "mine":
                    play_mine(state, strategy)

                played = True
                break  # re-scan from top of priority list
        if not played:
            break


def play_action_phase(state: GameState, strategy: Strategy) -> None:
    """Play action cards: all non-terminals first, then terminals."""
    from strategy import get_current_phase, get_action_priorities
    phase = get_current_phase(state.turn, state.supply["Province"], strategy.transitions)
    nt_priority, t_priority = get_action_priorities(strategy, phase)

    _play_action_tier(state, strategy, nt_priority, phase)
    _play_action_tier(state, strategy, t_priority, phase)


def play_moneylender(state: GameState) -> bool:
    """Trash a Copper from hand and gain +$3. Returns True if trashed."""
    if "Copper" in state.hand:
        trash_card(state, "Copper")
        state.coins += 3
        return True
    return False


def play_mine(state: GameState, strategy: Strategy) -> tuple[str, str] | None:
    """Trash a Treasure from hand, gain a Treasure costing up to $3 more to hand.

    Uses strategy.mine_trash_priority to decide which treasure to trash.
    Returns (trashed, gained) or None if nothing happened.
    """
    for treasure_name in strategy.mine_trash_priority:
        if treasure_name not in state.hand:
            continue
        trashed_cost = ALL_CARDS[treasure_name].cost
        max_gain_cost = trashed_cost + 3
        # Find best treasure to gain (highest cost up to max_gain_cost)
        for gain_name in ["Gold", "Silver", "Copper"]:
            gain_cost = ALL_CARDS[gain_name].cost
            if gain_cost <= max_gain_cost and gain_cost > trashed_cost and state.supply.get(gain_name, 0) > 0:
                trash_card(state, treasure_name)
                state.supply[gain_name] -= 1
                state.hand.append(gain_name)  # Mine puts gained card in hand
                return (treasure_name, gain_name)
        # No upgrade available for this treasure, try next in priority
    return None


def play_throne_room(state: GameState, strategy: Strategy,
                     phase: str = "mid") -> str | None:
    """Choose the best action from hand and play it twice.

    Uses strategy.throne_room_priority to pick the target.
    Returns the name of the doubled card, or None if no valid target.
    """
    # Pick highest-priority action in hand using throne_room_priority
    target = None
    for card_name in strategy.throne_room_priority:
        if card_name in state.hand and ALL_CARDS[card_name].card_type == CardType.ACTION:
            target = card_name
            break
    if target is None:
        return None

    # Move target to play area (no action cost — Throne Room already paid)
    state.hand.remove(target)
    state.play_area.append(target)

    card = ALL_CARDS[target]
    for _ in range(2):
        apply_action_effects(state, target)

        if card.special == "chapel":
            play_chapel(state, strategy, phase)
        elif card.special == "moneylender":
            play_moneylender(state)
        elif card.special == "mine":
            play_mine(state, strategy)

    return target


def play_chapel(state: GameState, strategy: Strategy,
                phase: str = "early") -> None:
    """Trash up to chapel_max_trash cards from hand following phase-specific trash priority."""
    from strategy import get_chapel_trash_priority
    chapel_trash_priority = get_chapel_trash_priority(strategy, phase)
    max_trash = min(strategy.chapel_max_trash, 4)
    trashed = 0
    for card_name in chapel_trash_priority:
        if card_name == "STOP" or trashed >= max_trash:
            break
        while card_name in state.hand and trashed < max_trash:
            trash_card(state, card_name)
            trashed += 1


def play_buy_phase(state: GameState, strategy: Strategy) -> None:
    """Auto-play all treasures, then buy cards following phase buy priority."""
    auto_play_treasures(state)

    # Determine current phase
    from strategy import get_current_phase, get_buy_priority
    phase = get_current_phase(state.turn, state.supply["Province"], strategy.transitions)
    buy_priority = get_buy_priority(strategy, phase)

    # Count owned cards for buy target checks
    buy_targets = strategy.buy_targets
    if buy_targets:
        owned: dict[str, int] = {}
        for c in state.deck + state.hand + state.discard + state.play_area:
            owned[c] = owned.get(c, 0) + 1
    else:
        owned = {}

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
                # Coin threshold check: skip if we have too many coins
                if card_name == "Province" and state.coins > strategy.province_max_coins:
                    continue
                if card_name == "Duchy" and state.coins > strategy.duchy_max_coins:
                    continue
                # Check buy target limit
                if card_name in buy_targets:
                    if owned.get(card_name, 0) >= buy_targets[card_name]:
                        continue  # skip, already at target
                buy_card(state, card_name)
                owned[card_name] = owned.get(card_name, 0) + 1
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
    vp = sum(ALL_CARDS[c].vp for c in all_cards)
    # Gardens: 1 VP per 10 cards in deck
    gardens_count = all_cards.count("Gardens")
    if gardens_count:
        vp += gardens_count * (len(all_cards) // 10)
    return vp


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
