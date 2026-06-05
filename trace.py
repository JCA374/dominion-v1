"""Game trace — watch the AI model play turn-by-turn.

Usage:
    python trace.py              # trace best model vs Big Money
    python trace.py --seed 123   # specific seed for reproducibility
    python trace.py --vs self    # trace best model vs itself
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from cards import ALL_CARDS, CardType, KINGDOM_CARDS
from engine import (
    GameState, default_supply, draw_cards, cleanup,
    is_game_over, count_vp, resolve_action, apply_action_effects,
    auto_play_treasures, buy_card, trash_card,
    play_moneylender, play_chapel, _new_player,
)
from strategy import (
    Strategy, load_strategy, big_money_strategy,
    get_current_phase, get_buy_priority,
)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

WIDTH = 90          # total terminal width
COL = 42            # width of each player column
GAP = WIDTH - 2 * COL  # center gap


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _left(text: str) -> str:
    """Left-aligned line (player 1)."""
    return text[:COL].ljust(COL)


def _right(text: str) -> str:
    """Right-aligned line (player 2)."""
    return " " * (WIDTH - COL) + text[:COL]


def _wrap(text: str, max_width: int, indent: int = 4) -> list[str]:
    """Break text into lines that fit within max_width."""
    if len(text) <= max_width:
        return [text]
    lines = []
    while len(text) > max_width:
        # Find a break point (comma or space)
        cut = max_width
        for sep in [", ", " "]:
            pos = text.rfind(sep, 0, max_width)
            if pos > indent:
                cut = pos + len(sep)
                break
        lines.append(text[:cut])
        text = " " * indent + text[cut:]
    if text.strip():
        lines.append(text)
    return lines


def _center(text: str) -> str:
    """Center-aligned line."""
    return text.center(WIDTH)


def _hand_str(hand: list[str]) -> str:
    counts = Counter(hand)
    parts = []
    for card, n in sorted(counts.items(), key=lambda x: -ALL_CARDS[x[0]].cost):
        tag = {"TREASURE": "$", "VICTORY": "V", "ACTION": "A"}[ALL_CARDS[card].card_type.name]
        if n > 1:
            parts.append(f"{n}x{card}({tag})")
        else:
            parts.append(f"{card}({tag})")
    return ", ".join(parts)


def _deck_summary(state: GameState) -> str:
    all_cards = state.deck + state.hand + state.discard + state.play_area
    counts = Counter(all_cards)
    parts = [f"{n}x{c}" for c, n in counts.most_common()]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Traced phases — collect lines instead of printing directly
# ---------------------------------------------------------------------------

def traced_action_phase(state: GameState, strategy: Strategy) -> list[str]:
    """Play action phase, return trace lines."""
    lines = []
    while state.actions > 0:
        played = False
        for card_name in strategy.action_priority:
            if card_name in state.hand and state.actions > 0:
                newly_drawn = resolve_action(state, card_name)

                card = ALL_CARDS[card_name]
                effects = []
                if card.cards_drawn:
                    effects.append(f"+{card.cards_drawn} card{'s' if card.cards_drawn > 1 else ''}")
                if card.actions:
                    effects.append(f"+{card.actions} action{'s' if card.actions > 1 else ''}")
                if card.buys:
                    effects.append(f"+{card.buys} buy{'s' if card.buys > 1 else ''}")
                if card.coins:
                    effects.append(f"+${card.coins}")
                effect_str = ", ".join(effects)
                lines.append(f"PLAY {card_name} ({effect_str})")

                if newly_drawn:
                    lines.append(f"  drew: {', '.join(newly_drawn)}")

                if card.special == "chapel":
                    lines.extend(traced_chapel(state, strategy))
                elif card.special == "moneylender":
                    if play_moneylender(state):
                        lines.append("  TRASH Copper (+$3)")
                    else:
                        lines.append("  (no Copper to trash)")
                elif card.special == "throne_room":
                    lines.extend(traced_throne_room(state, strategy))

                played = True
                break
        if not played:
            break
    return lines


def traced_chapel(state: GameState, strategy: Strategy) -> list[str]:
    lines = []
    max_trash = min(strategy.chapel_max_trash, 4)
    trashed = 0
    for card_name in strategy.chapel_trash_priority:
        if card_name == "STOP" or trashed >= max_trash:
            break
        while card_name in state.hand and trashed < max_trash:
            trash_card(state, card_name)
            trashed += 1
            lines.append(f"TRASH {card_name}")
    if trashed == 0:
        lines.append("TRASH nothing")
    return lines


def traced_throne_room(state: GameState, strategy: Strategy) -> list[str]:
    """Choose best action from hand, play it twice, return trace lines."""
    lines = []
    # Pick highest-priority action using throne_room_priority
    target = None
    for card_name in strategy.throne_room_priority:
        if card_name in state.hand and ALL_CARDS[card_name].card_type == CardType.ACTION:
            target = card_name
            break
    if target is None:
        lines.append("  (no action to double)")
        return lines

    lines.append(f"  doubles: {target}")
    state.hand.remove(target)
    state.play_area.append(target)

    target_card = ALL_CARDS[target]
    for i in range(2):
        tag = "1st" if i == 0 else "2nd"
        newly_drawn = apply_action_effects(state, target)

        effects = []
        if target_card.cards_drawn:
            effects.append(f"+{target_card.cards_drawn} card{'s' if target_card.cards_drawn > 1 else ''}")
        if target_card.actions:
            effects.append(f"+{target_card.actions} action{'s' if target_card.actions > 1 else ''}")
        if target_card.buys:
            effects.append(f"+{target_card.buys} buy{'s' if target_card.buys > 1 else ''}")
        if target_card.coins:
            effects.append(f"+${target_card.coins}")
        lines.append(f"  [{tag}] {', '.join(effects)}")

        if newly_drawn:
            lines.append(f"    drew: {', '.join(newly_drawn)}")

        if target_card.special == "chapel":
            lines.extend(traced_chapel(state, strategy))
        elif target_card.special == "moneylender":
            if play_moneylender(state):
                lines.append("    TRASH Copper (+$3)")
            else:
                lines.append("    (no Copper to trash)")

    return lines


def traced_buy_phase(state: GameState, strategy: Strategy) -> list[str]:
    lines = []
    treasures = auto_play_treasures(state)

    if treasures:
        counts = Counter(treasures)
        parts = [f"{n}x{c}" for c, n in counts.most_common()]
        lines.append(f"${state.coins} ({', '.join(parts)})")

    phase = get_current_phase(state.turn, state.supply["Province"], strategy.transitions)
    buy_priority = get_buy_priority(strategy, phase)

    buy_targets = strategy.buy_targets
    if buy_targets:
        owned: dict[str, int] = {}
        for c in state.deck + state.hand + state.discard + state.play_area:
            owned[c] = owned.get(c, 0) + 1
    else:
        owned = {}

    bought_any = False
    while state.buys > 0:
        bought = False
        for card_name in buy_priority:
            if card_name == "PASS":
                if not bought_any:
                    lines.append("BUY nothing (PASS)")
                state.buys = 0
                break
            if (card_name in state.supply
                    and state.supply[card_name] > 0
                    and ALL_CARDS[card_name].cost <= state.coins):
                if card_name in buy_targets:
                    if owned.get(card_name, 0) >= buy_targets[card_name]:
                        continue
                cost = ALL_CARDS[card_name].cost
                buy_card(state, card_name)
                owned[card_name] = owned.get(card_name, 0) + 1
                target_str = ""
                if card_name in buy_targets:
                    target_str = f" [{owned[card_name]}/{buy_targets[card_name]}]"
                lines.append(f"BUY {card_name} (${cost}){target_str}")
                bought = True
                bought_any = True
                break
        if not bought:
            if not bought_any:
                lines.append(f"BUY nothing (${state.coins})")
            break

    return lines


# ---------------------------------------------------------------------------
# Turn rendering
# ---------------------------------------------------------------------------

def _render_turn(lines: list[str], side: str) -> None:
    """Print turn lines aligned to left or right side."""
    for line in lines:
        wrapped = _wrap("  " + line, COL)
        for part in wrapped:
            if side == "left":
                print(_left(part))
            else:
                print(_right(part))


def _play_turn(player: GameState, strategy: Strategy, phase: str) -> list[str]:
    """Execute one turn and return all trace lines."""
    lines = []

    # Hand
    lines.append(f"Hand: {_hand_str(player.hand)}")
    total = len(player.deck) + len(player.hand) + len(player.discard) + len(player.play_area)
    lines.append(f"Deck: {total} cards ({len(player.deck)} draw)")

    # Action phase
    lines.extend(traced_action_phase(player, strategy))

    # Buy phase
    lines.extend(traced_buy_phase(player, strategy))

    cleanup(player)
    return lines


# ---------------------------------------------------------------------------
# Traced 2-player game
# ---------------------------------------------------------------------------

def trace_game(strategy1: Strategy, strategy2: Strategy,
               label1: str = "Model", label2: str = "Opponent",
               kingdom: list[str] | None = None,
               seed: int | None = None) -> dict:
    if kingdom is None:
        kingdom = KINGDOM_CARDS
    if seed is None:
        seed = random.randint(0, 2**31)

    rng = random.Random(seed)
    supply = default_supply(kingdom, num_players=2)

    p1 = _new_player(random.Random(rng.randint(0, 2**31)))
    p2 = _new_player(random.Random(rng.randint(0, 2**31)))
    p1.supply = supply
    p2.supply = supply

    labels = [label1, label2]
    strategies = [strategy1, strategy2]
    players = [p1, p2]
    sides = ["left", "right"]

    # Header
    print()
    print("=" * WIDTH)
    print(_center(f"GAME TRACE  (seed={seed})"))
    print(_center(f"Kingdom: {', '.join(kingdom)}"))
    print("=" * WIDTH)
    print()

    # Column headers
    print(_left(f"  {label1}") + _right(f"  {label2}").lstrip())
    print(_left("  " + "─" * (COL - 2)) + _right("  " + "─" * (COL - 2)).lstrip())
    print()

    round_num = 0
    while True:
        round_num += 1

        for i, (player, strategy) in enumerate(zip(players, strategies)):
            if is_game_over(player, turn_cap=40):
                print()
                return _game_result(p1, p2, round_num - 1, label1, label2)

            player.turn = round_num
            player.actions = 1
            player.buys = 1
            player.coins = 0

            phase = get_current_phase(
                player.turn, player.supply["Province"],
                strategy.transitions,
            )
            provs = player.supply["Province"]

            # Turn header in center for first player, skip for second
            if i == 0:
                print(_center(f"── Turn {round_num} · {provs} Prov left ──"))

            # Phase label
            side = sides[i]
            turn_lines = [f"[{phase}]"]
            turn_lines.extend(_play_turn(player, strategy, phase))
            _render_turn(turn_lines, side)
            print()


def _game_result(p1, p2, turns, label1, label2):
    vp1 = count_vp(p1)
    vp2 = count_vp(p2)

    print("=" * WIDTH)
    print(_center(f"GAME OVER  ·  {turns} turns"))
    print("=" * WIDTH)
    print()
    print(_left(f"  {label1}: {vp1} VP") + _right(f"  {label2}: {count_vp(p2)} VP").lstrip())
    print()

    # Deck breakdowns side by side
    d1 = _deck_summary(p1)
    d2 = _deck_summary(p2)
    # Split long deck summaries across lines
    for deck, side in [(d1, "left"), (d2, "right")]:
        words = deck.split(", ")
        line = ""
        for w in words:
            if line and len(line) + len(w) + 2 > COL - 4:
                if side == "left":
                    print(_left("  " + line))
                else:
                    print(_right("  " + line))
                line = w
            else:
                line = line + ", " + w if line else w
        if line:
            if side == "left":
                print(_left("  " + line))
            else:
                print(_right("  " + line))

    print()
    if vp1 > vp2:
        print(_center(f">>> {label1} wins! <<<"))
    elif vp2 > vp1:
        print(_center(f">>> {label2} wins! <<<"))
    else:
        print(_center(">>> Tie! <<<"))
    print()

    return {"vp1": vp1, "vp2": vp2, "turns": turns}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trace a game played by the AI model")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--model", default="best_model/strategy.json",
                        help="Path to strategy JSON")
    parser.add_argument("--vs", default="bigmoney",
                        choices=["bigmoney", "self"],
                        help="Opponent type: bigmoney or self")
    args = parser.parse_args()

    strategy = load_strategy(args.model)

    if args.vs == "self":
        opponent = load_strategy(args.model)
        opp_label = "Model (copy)"
    else:
        opponent = big_money_strategy()
        opp_label = "Big Money"

    trace_game(strategy, opponent,
               label1="Model", label2=opp_label,
               seed=args.seed)


if __name__ == "__main__":
    main()
