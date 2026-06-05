"""Interactive play mode — play Dominion against evolved AI models.

Kept separate from GA code so the genetic algorithm can be updated
independently. Uses the same engine primitives (GameState, draw_cards,
cleanup, is_game_over, count_vp) to ensure gameplay logic is identical.
"""

from __future__ import annotations

import os
import random
import re
import sys

from cards import ALL_CARDS, CardType, KINGDOM_CARDS
from engine import (
    GameState, default_supply, draw_cards, cleanup,
    is_game_over, count_vp, play_action_phase, play_buy_phase,
    resolve_action, apply_action_effects, auto_play_treasures,
    buy_card, trash_card, play_moneylender, _new_player,
)
from cards import TREASURE_CARDS
from strategy import load_strategy, describe, Strategy, big_money_strategy


# ---------------------------------------------------------------------------
# Model discovery & ranking
# ---------------------------------------------------------------------------

def discover_models(model_dir: str = "best_model") -> list[dict]:
    """Find all saved strategy.json files under model_dir.

    Returns a list of dicts sorted best-first (highest generation number):
        [{"path": ..., "gen": int, "label": str}, ...]
    The top-level best_model/strategy.json is treated as gen=infinity (best).
    """
    models: list[dict] = []

    # Top-level model (the overall best)
    top = os.path.join(model_dir, "strategy.json")
    if os.path.isfile(top):
        models.append({"path": top, "gen": float("inf"), "label": "Best (latest)"})

    # Per-generation snapshots
    if os.path.isdir(model_dir):
        for entry in os.listdir(model_dir):
            m = re.match(r"gen_(\d+)", entry)
            if m:
                path = os.path.join(model_dir, entry, "strategy.json")
                if os.path.isfile(path):
                    gen = int(m.group(1))
                    models.append({"path": path, "gen": gen,
                                   "label": f"Gen {gen}"})

    # Sort best-first (highest generation)
    models.sort(key=lambda x: x["gen"], reverse=True)
    return models


def select_opponent(models: list[dict]) -> Strategy:
    """Display models and let user pick an opponent."""
    print("\n╔══════════════════════════════════════╗")
    print("║       SELECT YOUR OPPONENT           ║")
    print("╠══════════════════════════════════════╣")

    # Always include Big Money as a baseline option
    print("║  0. Big Money (baseline)             ║")
    for i, m in enumerate(models, 1):
        label = m["label"]
        print(f"║  {i}. {label:<33s}║")
    print("╚══════════════════════════════════════╝")

    while True:
        try:
            choice = input("\nPick opponent [0]: ").strip()
            if choice == "":
                choice = "0"
            idx = int(choice)
            if idx == 0:
                print("  -> Big Money selected\n")
                return big_money_strategy()
            if 1 <= idx <= len(models):
                m = models[idx - 1]
                strat = load_strategy(m["path"])
                print(f"  -> {m['label']} selected\n")
                return strat
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _card_str(name: str) -> str:
    """Compact card representation with cost."""
    card = ALL_CARDS[name]
    tag = ""
    if card.card_type == CardType.TREASURE:
        tag = "$"
    elif card.card_type == CardType.VICTORY:
        tag = "V"
    elif card.card_type == CardType.ACTION:
        tag = "A"
    return f"{name}({tag}{card.cost})"


def show_state(state: GameState, label: str = "You") -> None:
    """Print current game state for the human player."""
    print(f"\n{'─' * 50}")
    print(f"  Turn {state.turn}  |  {label}")
    print(f"{'─' * 50}")

    # Show drawn cards (the hand IS what was drawn at start of turn)
    hand = sorted(state.hand, key=lambda c: (ALL_CARDS[c].card_type.value, c))
    print(f"  Drew: {', '.join(_card_str(c) for c in hand)}")
    print(f"  Actions: {state.actions}  |  Buys: {state.buys}  |  Coins: {state.coins}")

    # Deck/discard counts
    total_owned = len(state.deck) + len(state.hand) + len(state.discard) + len(state.play_area)
    print(f"  Deck: {len(state.deck)}  |  Discard: {len(state.discard)}  |  Total owned: {total_owned}")


def show_supply(supply: dict[str, int]) -> None:
    """Print the supply piles."""
    print(f"\n  {'Supply':─^46}")
    treasures = ["Copper", "Silver", "Gold"]
    victories = ["Estate", "Duchy", "Province"]
    actions = [k for k in supply if k not in treasures + victories]

    def fmt(names):
        return "  ".join(f"{n}: {supply[n]}" for n in names if n in supply)

    print(f"  Treasure: {fmt(treasures)}")
    print(f"  Victory:  {fmt(victories)}")
    print(f"  Actions:  {fmt(actions)}")


# ---------------------------------------------------------------------------
# Human turn — interactive action / buy / chapel phases
# ---------------------------------------------------------------------------

def human_action_phase(state: GameState) -> None:
    """Let the human choose which action cards to play."""
    while state.actions > 0:
        action_cards = [c for c in state.hand
                        if ALL_CARDS[c].card_type == CardType.ACTION]
        if not action_cards:
            break

        unique = sorted(set(action_cards))
        print(f"\n  Actions remaining: {state.actions}")
        print(f"  Action cards in hand: {', '.join(_card_str(c) for c in unique)}")
        for i, name in enumerate(unique, 1):
            card = ALL_CARDS[name]
            effects = []
            if card.cards_drawn:
                effects.append(f"+{card.cards_drawn} card{'s' if card.cards_drawn > 1 else ''}")
            if card.actions:
                effects.append(f"+{card.actions} action{'s' if card.actions > 1 else ''}")
            if card.buys:
                effects.append(f"+{card.buys} buy{'s' if card.buys > 1 else ''}")
            if card.coins:
                effects.append(f"+{card.coins} coin{'s' if card.coins > 1 else ''}")
            if card.special:
                effects.append(card.special)
            print(f"    {i}. {name} — {', '.join(effects)}")
        print(f"    0. Done (skip remaining actions)")

        try:
            choice = input("  Play action [0]: ").strip()
            if choice == "" or choice == "0":
                break
            idx = int(choice)
            if 1 <= idx <= len(unique):
                card_name = unique[idx - 1]
                _play_single_action(state, card_name)
            else:
                print("  Invalid choice.")
        except (ValueError, EOFError):
            break


def _play_single_action(state: GameState, card_name: str) -> None:
    """Resolve playing a single action card via engine primitive, then display."""
    card = ALL_CARDS[card_name]
    newly_drawn = resolve_action(state, card_name)

    # Display effects
    effects = []
    if card.cards_drawn:
        effects.append(f"+{card.cards_drawn} cards")
    if card.actions:
        effects.append(f"+{card.actions} actions")
    if card.buys:
        effects.append(f"+{card.buys} buys")
    if card.coins:
        effects.append(f"+{card.coins} coins")
    print(f"  Played {card_name}: {', '.join(effects)}")
    if newly_drawn:
        print(f"  Drew: {', '.join(_card_str(c) for c in newly_drawn)}")

    if card.special == "chapel":
        human_chapel(state)
    elif card.special == "moneylender":
        if play_moneylender(state):
            print("  Trashed Copper, gained +$3")
        else:
            print("  (no Copper in hand to trash)")
    elif card.special == "throne_room":
        human_throne_room(state)
    elif card.special == "mine":
        human_mine(state)

    # Show updated hand
    hand = sorted(state.hand, key=lambda c: (ALL_CARDS[c].card_type.value, c))
    print(f"  Hand now: {', '.join(_card_str(c) for c in hand)}")
    print(f"  Actions: {state.actions}  |  Coins: {state.coins}  |  Buys: {state.buys}")


def human_chapel(state: GameState) -> None:
    """Let human pick up to 4 cards to trash from hand."""
    trashed = 0
    while trashed < 4 and state.hand:
        unique = sorted(set(state.hand))
        print(f"\n  Chapel — trash up to {4 - trashed} more card(s) from hand:")
        for i, name in enumerate(unique, 1):
            count = state.hand.count(name)
            suffix = f" (x{count})" if count > 1 else ""
            print(f"    {i}. {_card_str(name)}{suffix}")
        print(f"    0. Done trashing")

        try:
            choice = input("  Trash [0]: ").strip()
            if choice == "" or choice == "0":
                break
            idx = int(choice)
            if 1 <= idx <= len(unique):
                card_name = unique[idx - 1]
                trash_card(state, card_name)
                trashed += 1
                print(f"  Trashed {card_name}")
            else:
                print("  Invalid choice.")
        except (ValueError, EOFError):
            break


def human_throne_room(state: GameState) -> None:
    """Let human pick an action card from hand to play twice."""
    action_cards = [c for c in state.hand
                    if ALL_CARDS[c].card_type == CardType.ACTION
                    and c != "Throne Room"]
    if not action_cards:
        print("  (no action card to double)")
        return

    unique = sorted(set(action_cards))
    print(f"\n  Throne Room — choose an action to play twice:")
    for i, name in enumerate(unique, 1):
        card = ALL_CARDS[name]
        effects = []
        if card.cards_drawn:
            effects.append(f"+{card.cards_drawn} cards")
        if card.actions:
            effects.append(f"+{card.actions} actions")
        if card.buys:
            effects.append(f"+{card.buys} buys")
        if card.coins:
            effects.append(f"+{card.coins} coins")
        if card.special:
            effects.append(card.special)
        print(f"    {i}. {name} — {', '.join(effects)}")

    while True:
        try:
            choice = input("  Double which action? ").strip()
            idx = int(choice)
            if 1 <= idx <= len(unique):
                target = unique[idx - 1]
                break
            print("  Invalid choice.")
        except (ValueError, EOFError):
            # Default to first option
            target = unique[0]
            break

    # Move target to play area, play effects twice
    state.hand.remove(target)
    state.play_area.append(target)

    target_card = ALL_CARDS[target]
    for i in range(2):
        tag = "1st" if i == 0 else "2nd"
        newly_drawn = apply_action_effects(state, target)
        effects = []
        if target_card.cards_drawn:
            effects.append(f"+{target_card.cards_drawn} cards")
        if target_card.actions:
            effects.append(f"+{target_card.actions} actions")
        if target_card.buys:
            effects.append(f"+{target_card.buys} buys")
        if target_card.coins:
            effects.append(f"+{target_card.coins} coins")
        print(f"  [{tag}] {', '.join(effects)}")
        if newly_drawn:
            print(f"  Drew: {', '.join(_card_str(c) for c in newly_drawn)}")

        if target_card.special == "chapel":
            human_chapel(state)
        elif target_card.special == "moneylender":
            if play_moneylender(state):
                print("  Trashed Copper, gained +$3")
            else:
                print("  (no Copper to trash)")
        elif target_card.special == "mine":
            human_mine(state)


def human_mine(state: GameState) -> None:
    """Let human pick a treasure to trash and upgrade via Mine."""
    treasures_in_hand = [c for c in state.hand
                         if ALL_CARDS[c].card_type == CardType.TREASURE
                         and c != "Gold"]  # Gold can't be upgraded
    if not treasures_in_hand:
        print("  (no treasure to upgrade)")
        return

    unique = sorted(set(treasures_in_hand), key=lambda c: ALL_CARDS[c].cost)
    print(f"\n  Mine — trash a treasure to gain one costing up to $3 more:")
    for i, name in enumerate(unique, 1):
        cost = ALL_CARDS[name].cost
        max_cost = cost + 3
        upgrades = [t for t in ["Silver", "Gold"]
                    if ALL_CARDS[t].cost <= max_cost and ALL_CARDS[t].cost > cost
                    and state.supply.get(t, 0) > 0]
        upgrade_str = f" -> {' or '.join(upgrades)}" if upgrades else " (no upgrade available)"
        print(f"    {i}. {_card_str(name)}{upgrade_str}")
    print(f"    0. Skip")

    try:
        choice = input("  Trash which treasure? [0]: ").strip()
        if choice == "" or choice == "0":
            return
        idx = int(choice)
        if 1 <= idx <= len(unique):
            trash_name = unique[idx - 1]
            trashed_cost = ALL_CARDS[trash_name].cost
            max_gain = trashed_cost + 3
            # Find best gain
            for gain_name in ["Gold", "Silver"]:
                gc = ALL_CARDS[gain_name].cost
                if gc <= max_gain and gc > trashed_cost and state.supply.get(gain_name, 0) > 0:
                    trash_card(state, trash_name)
                    state.supply[gain_name] -= 1
                    state.hand.append(gain_name)
                    print(f"  Trashed {trash_name}, gained {gain_name} to hand")
                    return
            print("  (no valid upgrade in supply)")
        else:
            print("  Invalid choice.")
    except (ValueError, EOFError):
        pass


def human_buy_phase(state: GameState) -> None:
    """Auto-play treasures via engine, then let human choose what to buy."""
    treasures = auto_play_treasures(state)

    if treasures:
        counts: dict[str, int] = {}
        for t in treasures:
            counts[t] = counts.get(t, 0) + 1
        parts = [f"{n}x {c}" for c, n in sorted(counts.items())]
        print(f"\n  Played treasures: {', '.join(parts)} -> {state.coins} coins")

    while state.buys > 0:
        # Build list of affordable cards with supply
        affordable = []
        for name, count in state.supply.items():
            if count > 0 and ALL_CARDS[name].cost <= state.coins:
                affordable.append(name)
        affordable.sort(key=lambda c: -ALL_CARDS[c].cost)

        if not affordable:
            print("  Nothing affordable — ending buy phase.")
            break

        print(f"\n  Coins: {state.coins}  |  Buys: {state.buys}")
        print(f"  Affordable cards:")
        for i, name in enumerate(affordable, 1):
            card = ALL_CARDS[name]
            info = f"cost {card.cost}"
            if card.vp:
                info += f", {card.vp} VP"
            if card.coins:
                info += f", +{card.coins} coins"
            if card.cards_drawn:
                info += f", +{card.cards_drawn} cards"
            if card.actions:
                info += f", +{card.actions} actions"
            if card.buys:
                info += f", +{card.buys} buys"
            if card.special:
                info += f", {card.special}"
            print(f"    {i}. {name} — {info}  [{state.supply[name]} left]")
        print(f"    0. Done buying")

        try:
            choice = input("  Buy [0]: ").strip()
            if choice == "" or choice == "0":
                break
            idx = int(choice)
            if 1 <= idx <= len(affordable):
                card_name = affordable[idx - 1]
                buy_card(state, card_name)
                print(f"  Bought {card_name}!")
            else:
                print("  Invalid choice.")
        except (ValueError, EOFError):
            break


# ---------------------------------------------------------------------------
# AI turn — uses the same engine functions
# ---------------------------------------------------------------------------

def ai_turn(state: GameState, strategy: Strategy) -> None:
    """Execute an AI turn using the standard engine, then report what happened."""
    hand_before = list(state.hand)
    supply_before = dict(state.supply)

    play_action_phase(state, strategy)
    play_buy_phase(state, strategy)

    # Report what the AI did
    played_actions = [c for c in state.play_area
                      if ALL_CARDS[c].card_type == CardType.ACTION]
    bought = []
    for name in state.supply:
        diff = supply_before[name] - state.supply[name]
        for _ in range(diff):
            bought.append(name)

    print(f"\n{'─' * 50}")
    print(f"  Turn {state.turn}  |  AI Opponent")
    print(f"{'─' * 50}")
    if played_actions:
        print(f"  Played: {', '.join(played_actions)}")
    if bought:
        print(f"  Bought: {', '.join(bought)}")
    if not played_actions and not bought:
        print("  (did nothing notable)")

    cleanup(state)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def play_interactive(opponent: Strategy,
                     kingdom: list[str] | None = None,
                     seed: int | None = None) -> None:
    """Play an interactive 2-player game: human vs AI."""
    if kingdom is None:
        kingdom = KINGDOM_CARDS
    if seed is None:
        seed = random.randint(0, 2**31)

    rng = random.Random(seed)
    supply = default_supply(kingdom, num_players=2)

    # Create players using the same engine helper as play_game_2p
    human = _new_player(random.Random(rng.randint(0, 2**31)))
    human.supply = supply
    ai = _new_player(random.Random(rng.randint(0, 2**31)))
    ai.supply = supply  # shared supply

    print("\n" + "=" * 50)
    print("  GAME START")
    print("=" * 50)
    print(f"  Kingdom: {', '.join(kingdom)}")
    print(f"  Provinces: {supply['Province']}")

    round_num = 0
    while True:
        round_num += 1

        # --- Human turn ---
        if is_game_over(human, turn_cap=40):
            break
        human.turn = round_num
        human.actions = 1
        human.buys = 1
        human.coins = 0

        show_supply(supply)
        show_state(human, "Your Turn")
        human_action_phase(human)
        human_buy_phase(human)
        cleanup(human)

        # --- AI turn ---
        if is_game_over(ai, turn_cap=40):
            break
        ai.turn = round_num
        ai.actions = 1
        ai.buys = 1
        ai.coins = 0

        ai_turn(ai, opponent)

    # Game over
    human_vp = count_vp(human)
    ai_vp = count_vp(ai)

    print("\n" + "=" * 50)
    print("  GAME OVER")
    print("=" * 50)
    print(f"  Turns played: {round_num - 1}")
    print(f"  Your VP:      {human_vp}")
    print(f"  AI VP:        {ai_vp}")
    print()
    if human_vp > ai_vp:
        print("  *** YOU WIN! ***")
    elif ai_vp > human_vp:
        print("  *** AI WINS ***")
    else:
        print("  *** TIE ***")
    print()

    # Show final decks
    def deck_summary(state: GameState) -> str:
        all_cards = state.deck + state.hand + state.discard + state.play_area
        counts: dict[str, int] = {}
        for c in all_cards:
            counts[c] = counts.get(c, 0) + 1
        return ", ".join(f"{n}x {c}" for c, n in
                         sorted(counts.items(), key=lambda x: -x[1]))

    print(f"  Your deck:  {deck_summary(human)}")
    print(f"  AI deck:    {deck_summary(ai)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("╔══════════════════════════════════════╗")
    print("║     DOMINION — Play vs AI            ║")
    print("╚══════════════════════════════════════╝")

    models = discover_models()
    if not models:
        print("\nNo saved models found in best_model/.")
        print("Using Big Money as opponent.\n")
        opponent = big_money_strategy()
    else:
        opponent = select_opponent(models)

    # Show opponent strategy overview
    print(describe(opponent))
    print()

    while True:
        play_interactive(opponent)
        again = input("\nPlay again vs same opponent? [Y/n]: ").strip().lower()
        if again and again != "y":
            break


if __name__ == "__main__":
    main()
