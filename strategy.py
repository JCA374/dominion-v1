"""Strategy genome, phase selection, and description for phase-aware GA."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict

from cards import BUYABLE_CARDS, ACTION_CARDS


@dataclass
class Transitions:
    early_to_mid_turn: int       # range [2, 15]
    mid_to_late_provinces: int   # range [0, 8]


@dataclass
class Strategy:
    early_buy_priority: list[str]
    mid_buy_priority: list[str]
    late_buy_priority: list[str]
    action_priority: list[str]
    chapel_trash_priority: list[str]
    transitions: Transitions


def get_current_phase(turn: int, provinces_remaining: int,
                      transitions: Transitions) -> str:
    """Return 'early', 'mid', or 'late' based on turn and province count."""
    if turn <= transitions.early_to_mid_turn:
        return "early"
    elif provinces_remaining > transitions.mid_to_late_provinces:
        return "mid"
    else:
        return "late"


def get_buy_priority(strategy: Strategy, phase: str) -> list[str]:
    """Return the buy priority list for the given phase."""
    if phase == "early":
        return strategy.early_buy_priority
    elif phase == "mid":
        return strategy.mid_buy_priority
    else:
        return strategy.late_buy_priority


def random_strategy(rng: random.Random) -> Strategy:
    """Generate a random strategy with shuffled priority lists."""
    def shuffled(lst):
        copy = list(lst)
        rng.shuffle(copy)
        return copy

    # Each buy priority: shuffled buyable cards with PASS always included
    def random_buy_priority():
        cards = shuffled(BUYABLE_CARDS)
        pos = rng.randint(0, len(cards))
        cards.insert(pos, "PASS")
        return cards

    # Chapel trash priority: card types we might want to trash + STOP
    trashable = ["Estate", "Copper", "Duchy", "Silver", "STOP"]
    chapel_priority = shuffled(trashable)

    return Strategy(
        early_buy_priority=random_buy_priority(),
        mid_buy_priority=random_buy_priority(),
        late_buy_priority=random_buy_priority(),
        action_priority=shuffled(ACTION_CARDS),
        chapel_trash_priority=chapel_priority,
        transitions=Transitions(
            early_to_mid_turn=rng.randint(2, 15),
            mid_to_late_provinces=rng.randint(0, 8),
        ),
    )


def _full_buy_priority(preferred: list[str]) -> list[str]:
    """Build a complete buy priority: preferred cards first, remaining BUYABLE_CARDS after, PASS at end."""
    rest = [c for c in BUYABLE_CARDS if c not in preferred and c != "PASS"]
    return preferred + rest + ["PASS"]


def big_money_strategy() -> Strategy:
    """Seed archetype: Province > Gold > Silver, no actions."""
    return Strategy(
        early_buy_priority=_full_buy_priority(["Silver", "Gold"]),
        mid_buy_priority=_full_buy_priority(["Province", "Gold", "Silver"]),
        late_buy_priority=_full_buy_priority(["Province", "Duchy", "Gold", "Estate", "Silver"]),
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
    )


def engine_strategy() -> Strategy:
    """Seed archetype: Chapel engine — thin deck early, build mid, green late."""
    return Strategy(
        early_buy_priority=_full_buy_priority(["Chapel", "Silver", "Village"]),
        mid_buy_priority=_full_buy_priority(["Gold", "Smithy", "Silver", "Market", "Laboratory"]),
        late_buy_priority=_full_buy_priority(["Province", "Duchy", "Gold", "Estate"]),
        action_priority=["Village", "Festival", "Laboratory", "Market", "Smithy", "Chapel"],
        chapel_trash_priority=["Estate", "STOP", "Copper"],
        transitions=Transitions(early_to_mid_turn=5, mid_to_late_provinces=4),
    )


def describe(strategy: Strategy, fitness: float | None = None) -> str:
    """Human-readable multi-line strategy summary."""
    t = strategy.transitions

    def fmt_list(lst: list[str], limit: int = 6) -> str:
        shown = lst[:limit]
        return " > ".join(shown) + (" > ..." if len(lst) > limit else "")

    lines = []
    header = "=== Strategy"
    if fitness is not None:
        header += f" (fitness {fitness:.1f} VP)"
    header += " ==="
    lines.append(header)

    lines.append(f"EARLY (turns 1-{t.early_to_mid_turn}):   {fmt_list(strategy.early_buy_priority)}")
    lines.append(f"MID   (until {t.mid_to_late_provinces} Prov): {fmt_list(strategy.mid_buy_priority)}")
    lines.append(f"LATE  (<= {t.mid_to_late_provinces} Prov):   {fmt_list(strategy.late_buy_priority)}")
    lines.append(f"Actions: {fmt_list(strategy.action_priority)}")
    lines.append(f"Chapel trash: {fmt_list(strategy.chapel_trash_priority)}")

    return "\n".join(lines)


def summarize(strategy: Strategy, vs_bm: dict) -> str:
    """Plain-English summary of what the evolved strategy does and why it works."""
    from cards import ALL_CARDS, CardType

    t = strategy.transitions
    lines = []
    lines.append("=" * 60)
    lines.append("  BEST TACTIC SUMMARY")
    lines.append("=" * 60)

    # --- Performance ---
    num_games = vs_bm.get("num_games", "?")
    opponent_label = vs_bm.get("opponent", "Big Money")
    lines.append("")
    lines.append(f"  Performance ({num_games} games vs {opponent_label})")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Win rate:  {vs_bm['win_rate']:.0%} win / "
                 f"{vs_bm['tie_rate']:.0%} tie / {vs_bm['loss_rate']:.0%} loss")
    if "mean_turns" in vs_bm:
        lines.append(f"  Avg game:  {vs_bm['mean_turns']:.1f} turns")

    # --- Phase timing ---
    lines.append("")
    lines.append("  Phase Timing")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Early game: turns 1-{t.early_to_mid_turn}")
    lines.append(f"  Mid game:   turn {t.early_to_mid_turn + 1}+ while "
                 f">{t.mid_to_late_provinces} Provinces remain")
    lines.append(f"  Late game:  {t.mid_to_late_provinces} or fewer Provinces left")

    # --- Per-phase tactic ---
    def top_buys(priority: list[str], n: int = 4) -> list[str]:
        """Return top N non-PASS cards from a priority list."""
        return [c for c in priority if c != "PASS"][:n]

    def classify_cards(cards: list[str]) -> dict[str, list[str]]:
        types = {"treasure": [], "victory": [], "action": []}
        for c in cards:
            ct = ALL_CARDS[c].card_type
            if ct == CardType.TREASURE:
                types["treasure"].append(c)
            elif ct == CardType.VICTORY:
                types["victory"].append(c)
            else:
                types["action"].append(c)
        return types

    for phase_name, priority in [("Early", strategy.early_buy_priority),
                                  ("Mid", strategy.mid_buy_priority),
                                  ("Late", strategy.late_buy_priority)]:
        top = top_buys(priority)
        types = classify_cards(top)
        lines.append("")
        lines.append(f"  {phase_name} Phase — top buys: {' > '.join(top)}")
        lines.append(f"  {'─' * 40}")

        # Describe the focus
        focus_parts = []
        if types["action"]:
            focus_parts.append(f"action cards ({', '.join(types['action'])})")
        if types["treasure"]:
            focus_parts.append(f"treasure ({', '.join(types['treasure'])})")
        if types["victory"]:
            focus_parts.append(f"victory ({', '.join(types['victory'])})")
        lines.append(f"  Focus: {', then '.join(focus_parts)}")

        # Check for PASS
        pass_pos = priority.index("PASS") if "PASS" in priority else None
        if pass_pos is not None and pass_pos < len(priority) - 1:
            lines.append(f"  Stops buying after rank {pass_pos + 1} "
                         f"(skips cheaper cards if top picks unaffordable)")

    # --- Action play order ---
    lines.append("")
    lines.append("  Action Play Order")
    lines.append(f"  {'─' * 40}")
    if strategy.action_priority:
        lines.append(f"  {' > '.join(strategy.action_priority)}")
    else:
        lines.append("  (no actions played)")

    # --- Chapel strategy ---
    lines.append("")
    lines.append("  Chapel Trashing")
    lines.append(f"  {'─' * 40}")
    stop_idx = (strategy.chapel_trash_priority.index("STOP")
                if "STOP" in strategy.chapel_trash_priority else None)
    if stop_idx == 0:
        lines.append("  Never trashes (STOP at top)")
    elif stop_idx is not None:
        trash_targets = strategy.chapel_trash_priority[:stop_idx]
        lines.append(f"  Trashes: {' > '.join(trash_targets)}")
    else:
        lines.append(f"  Trashes: {' > '.join(strategy.chapel_trash_priority)}")

    # --- Key insight ---
    lines.append("")
    lines.append("  Key Insight")
    lines.append(f"  {'─' * 40}")

    early_top = top_buys(strategy.early_buy_priority, 3)
    early_types = classify_cards(early_top)
    late_top = top_buys(strategy.late_buy_priority, 3)
    late_types = classify_cards(late_top)

    if early_types["action"]:
        lines.append("  Builds an engine early with action cards, then")
    else:
        lines.append("  Prioritizes money early (pure Big Money style), then")

    if t.mid_to_late_provinces >= 6:
        lines.append("  starts greening aggressively while many Provinces remain.")
    elif t.mid_to_late_provinces >= 3:
        lines.append("  switches to victory cards at a moderate greening point.")
    else:
        lines.append("  waits until very late to buy victory cards (lean deck).")

    # --- Average final deck ---
    avg_deck = vs_bm.get("avg_final_deck")
    if avg_deck:
        lines.append("")
        lines.append("  Average Final Deck")
        lines.append(f"  {'─' * 40}")
        total = sum(avg_deck.values())
        for card, count in avg_deck.items():
            bar = "█" * int(round(count))
            lines.append(f"  {card:>12s}  {count:4.1f}  {bar}")
        lines.append(f"  {'Total':>12s}  {total:4.1f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_best_model(strategy: Strategy, vs_bm: dict,
                    output_dir: str = "best_model") -> None:
    """Save the best strategy as JSON + text summary + buy heatmap."""
    from plotting import plot_buy_heatmap

    os.makedirs(output_dir, exist_ok=True)

    # Save strategy as JSON (machine-readable, can be loaded back)
    data = asdict(strategy)
    if "avg_final_deck" in vs_bm:
        data["avg_final_deck"] = vs_bm["avg_final_deck"]
    with open(os.path.join(output_dir, "strategy.json"), "w") as f:
        json.dump(data, f, indent=2)

    # Save text summary
    summary = summarize(strategy, vs_bm)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # Save buy heatmap
    plot_buy_heatmap(strategy, os.path.join(output_dir, "buy_heatmap.png"))

    print(f"Best model saved to {output_dir}/")


def load_strategy(path: str) -> Strategy:
    """Load a strategy from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return Strategy(
        early_buy_priority=data["early_buy_priority"],
        mid_buy_priority=data["mid_buy_priority"],
        late_buy_priority=data["late_buy_priority"],
        action_priority=data["action_priority"],
        chapel_trash_priority=data["chapel_trash_priority"],
        transitions=Transitions(**data["transitions"]),
    )
