"""Strategy genome, phase selection, and description for phase-aware GA."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field

from core.cards import (BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS, TREASURE_CARDS,
                    VICTORY_CARDS, ALL_CARDS, CardType)


@dataclass
class Transitions:
    early_to_mid_turn: int       # range [2, 15]
    mid_to_late_provinces: int   # range [2, 8]
    mid_to_late_turn: int = 20   # range [5, 30] — fallback if provinces don't drop
    late_to_end_provinces: int = 2  # range [1, 4]


@dataclass
class Strategy:
    early_buy_priority: list[str]
    mid_buy_priority: list[str]
    late_buy_priority: list[str]
    end_buy_priority: list[str] = field(default_factory=list)
    early_nonterminal_priority: list[str] = field(default_factory=list)
    early_terminal_priority: list[str] = field(default_factory=list)
    mid_nonterminal_priority: list[str] = field(default_factory=list)
    mid_terminal_priority: list[str] = field(default_factory=list)
    late_nonterminal_priority: list[str] = field(default_factory=list)
    late_terminal_priority: list[str] = field(default_factory=list)
    end_nonterminal_priority: list[str] = field(default_factory=list)
    end_terminal_priority: list[str] = field(default_factory=list)
    early_chapel_trash: list[str] = field(default_factory=list)
    mid_chapel_trash: list[str] = field(default_factory=list)
    late_chapel_trash: list[str] = field(default_factory=list)
    end_chapel_trash: list[str] = field(default_factory=list)
    transitions: Transitions = field(default_factory=lambda: Transitions(4, 4, 2))
    throne_room_priority: list[str] = field(default_factory=list)  # which action to double (best first)
    mine_trash_priority: list[str] = field(default_factory=list)  # which treasure to upgrade (Copper/Silver)
    chapel_max_trash: int = 4          # 0-4: max cards to trash per Chapel play
    buy_targets: dict[str, int] = field(default_factory=dict)  # card -> max copies to own (empty = no limits)
    province_max_coins: int = 99       # skip Province if coins > this (99 = always buy)
    duchy_max_coins: int = 99          # skip Duchy if coins > this (99 = always buy)
    militia_coin_threshold: int = 5    # discard heuristic: keep money if coins >= this


def get_current_phase(turn: int, provinces_remaining: int,
                      transitions: Transitions) -> str:
    """Return 'early', 'mid', 'late', or 'end' based on turn and province count."""
    if turn <= transitions.early_to_mid_turn:
        return "early"
    elif (provinces_remaining > transitions.mid_to_late_provinces
          and turn < transitions.mid_to_late_turn):
        return "mid"
    elif provinces_remaining > transitions.late_to_end_provinces:
        return "late"
    else:
        return "end"


def get_buy_priority(strategy: Strategy, phase: str) -> list[str]:
    """Return the buy priority list for the given phase."""
    if phase == "early":
        return strategy.early_buy_priority
    elif phase == "mid":
        return strategy.mid_buy_priority
    elif phase == "late":
        return strategy.late_buy_priority
    else:
        return strategy.end_buy_priority


def get_action_priorities(strategy: Strategy, phase: str) -> tuple[list[str], list[str]]:
    """Return (nonterminal_priority, terminal_priority) for the given phase."""
    if phase == "early":
        return strategy.early_nonterminal_priority, strategy.early_terminal_priority
    elif phase == "mid":
        return strategy.mid_nonterminal_priority, strategy.mid_terminal_priority
    elif phase == "late":
        return strategy.late_nonterminal_priority, strategy.late_terminal_priority
    else:
        return strategy.end_nonterminal_priority, strategy.end_terminal_priority


def get_action_priority(strategy: Strategy, phase: str) -> list[str]:
    """Return combined action priority (nonterminals first, then terminals)."""
    nt, t = get_action_priorities(strategy, phase)
    return list(nt) + list(t)


def get_chapel_trash_priority(strategy: Strategy, phase: str) -> list[str]:
    """Return the chapel trash priority list for the given phase."""
    if phase == "early":
        return strategy.early_chapel_trash
    elif phase == "mid":
        return strategy.mid_chapel_trash
    elif phase == "late":
        return strategy.late_chapel_trash
    else:
        return strategy.end_chapel_trash


def _buyable_cards(kingdom: list[str] | None = None) -> list[str]:
    """Buyable cards for a given kingdom selection."""
    if kingdom is None:
        return BUYABLE_CARDS
    return TREASURE_CARDS + VICTORY_CARDS + kingdom


def _action_cards(kingdom: list[str] | None = None) -> list[str]:
    """Action cards for a given kingdom selection."""
    if kingdom is None:
        return ACTION_CARDS
    return [c for c in kingdom if ALL_CARDS[c].card_type == CardType.ACTION]


def _nonterminal_actions(kingdom: list[str] | None = None) -> list[str]:
    """Non-terminal action cards (+actions > 0) for a given kingdom."""
    return [c for c in _action_cards(kingdom) if ALL_CARDS[c].actions > 0]


def _terminal_actions(kingdom: list[str] | None = None) -> list[str]:
    """Terminal action cards (actions == 0) for a given kingdom."""
    return [c for c in _action_cards(kingdom) if ALL_CARDS[c].actions == 0]


def _split_priority(ordered_actions: list[str]) -> tuple[list[str], list[str]]:
    """Split an ordered action list into (nonterminal, terminal) preserving order."""
    nt = [c for c in ordered_actions if c in ALL_CARDS and ALL_CARDS[c].actions > 0]
    t = [c for c in ordered_actions if c in ALL_CARDS and ALL_CARDS[c].actions == 0]
    return nt, t



def random_strategy(rng: random.Random,
                    kingdom: list[str] | None = None) -> Strategy:
    """Generate a random strategy with shuffled priority lists."""
    buyable = _buyable_cards(kingdom)
    actions = _action_cards(kingdom)

    def shuffled(lst):
        copy = list(lst)
        rng.shuffle(copy)
        return copy

    # Each buy priority: shuffled buyable cards with PASS always included
    def random_buy_priority():
        cards = shuffled(buyable)
        pos = rng.randint(0, len(cards))
        cards.insert(pos, "PASS")
        return cards

    # Chapel trash is hardcoded: Estate > Copper > STOP (early/mid), STOP (late/end)
    early_chapel = ["Curse", "Estate", "Copper", "STOP"]
    mid_chapel = ["Curse", "Estate", "Copper", "STOP"]
    late_chapel = ["Curse", "STOP"]
    end_chapel = ["Curse", "STOP"]

    # Buy targets: max copies to own per card (action cards 1-5, others uncapped)
    kingdom_cards = kingdom if kingdom is not None else KINGDOM_CARDS
    buy_targets: dict[str, int] = {}
    for card in kingdom_cards:
        buy_targets[card] = rng.randint(1, 5)

    # Throne Room priority: which action to double (excludes Throne Room itself)
    tr_candidates = [c for c in actions if c != "Throne Room"]
    throne_room_priority = shuffled(tr_candidates)

    # Mine trash priority: which treasure to upgrade first
    mine_trash_priority = shuffled(["Copper", "Silver"])

    return Strategy(
        early_buy_priority=random_buy_priority(),
        mid_buy_priority=random_buy_priority(),
        late_buy_priority=random_buy_priority(),
        end_buy_priority=random_buy_priority(),
        early_nonterminal_priority=shuffled(_nonterminal_actions(kingdom)),
        early_terminal_priority=shuffled(_terminal_actions(kingdom)),
        mid_nonterminal_priority=shuffled(_nonterminal_actions(kingdom)),
        mid_terminal_priority=shuffled(_terminal_actions(kingdom)),
        late_nonterminal_priority=shuffled(_nonterminal_actions(kingdom)),
        late_terminal_priority=shuffled(_terminal_actions(kingdom)),
        end_nonterminal_priority=shuffled(_nonterminal_actions(kingdom)),
        end_terminal_priority=shuffled(_terminal_actions(kingdom)),
        early_chapel_trash=early_chapel,
        mid_chapel_trash=mid_chapel,
        late_chapel_trash=late_chapel,
        end_chapel_trash=end_chapel,
        throne_room_priority=throne_room_priority,
        mine_trash_priority=mine_trash_priority,
        chapel_max_trash=4,
        transitions=Transitions(
            early_to_mid_turn=rng.randint(2, 15),
            mid_to_late_provinces=rng.randint(2, 8),
            mid_to_late_turn=rng.randint(5, 30),
            late_to_end_provinces=rng.randint(1, 4),
        ),
        buy_targets=buy_targets,
        province_max_coins=rng.choice([8, 9, 10, 11, 99]),
        duchy_max_coins=rng.choice([5, 6, 7, 8, 99]),
        militia_coin_threshold=rng.randint(3, 8),
    )


def _full_buy_priority(preferred: list[str],
                       kingdom: list[str] | None = None) -> list[str]:
    """Build a complete buy priority: preferred cards first, remaining buyable after, PASS at end."""
    buyable = _buyable_cards(kingdom)
    rest = [c for c in buyable if c not in preferred and c != "PASS"]
    return [c for c in preferred if c in buyable] + rest + ["PASS"]


def _default_throne_room_priority(kingdom: list[str] | None = None) -> list[str]:
    """Default Throne Room target priority: high-impact cards first."""
    actions = _action_cards(kingdom)
    return [c for c in actions if c != "Throne Room"]


def _prioritize(preferred: list[str], all_cards: list[str]) -> list[str]:
    """Put preferred cards first (in order), then remaining cards."""
    rest = [c for c in all_cards if c not in preferred]
    return [c for c in preferred if c in all_cards] + rest


def _default_mine_trash_priority() -> list[str]:
    """Default Mine trash priority: upgrade Copper first, then Silver."""
    return ["Copper", "Silver"]


def big_money_strategy(kingdom: list[str] | None = None) -> Strategy:
    """Seed archetype: Province > Gold > Silver, no actions.

    Uses explicit PASS-terminated lists so it never buys junk (Copper, Chapel, etc.)
    when it can't afford its preferred cards.
    """
    nt = _nonterminal_actions(kingdom)
    t = _terminal_actions(kingdom)
    return Strategy(
        early_buy_priority=["Gold", "Silver", "PASS"],
        mid_buy_priority=["Province", "Gold", "Silver", "PASS"],
        late_buy_priority=["Province", "Duchy", "Gold", "Estate", "Silver", "PASS"],
        end_buy_priority=["Province", "Duchy", "Estate", "Gold", "Silver", "PASS"],
        early_nonterminal_priority=nt[:],
        early_terminal_priority=t[:],
        mid_nonterminal_priority=nt[:],
        mid_terminal_priority=t[:],
        late_nonterminal_priority=nt[:],
        late_terminal_priority=t[:],
        end_nonterminal_priority=nt[:],
        end_terminal_priority=t[:],
        early_chapel_trash=["STOP"],
        mid_chapel_trash=["STOP"],
        late_chapel_trash=["STOP"],
        end_chapel_trash=["STOP"],
        throne_room_priority=_default_throne_room_priority(kingdom),
        mine_trash_priority=_default_mine_trash_priority(),
        chapel_max_trash=0,
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3, late_to_end_provinces=2),
        buy_targets={},  # no limits — pure Big Money
    )


def gardens_strategy(kingdom: list[str] | None = None) -> Strategy:
    """Seed archetype: Gardens rush — buy lots of cheap cards for VP."""
    nt = _nonterminal_actions(kingdom)
    t = _terminal_actions(kingdom)
    tr_candidates = [c for c in _action_cards(kingdom) if c != "Throne Room"]
    return Strategy(
        early_buy_priority=_full_buy_priority(["Gardens", "Silver", "Throne Room"], kingdom),
        mid_buy_priority=_full_buy_priority(["Gardens", "Silver", "Throne Room", "Copper", "Estate"], kingdom),
        late_buy_priority=_full_buy_priority(["Province", "Gardens", "Duchy", "Estate", "Copper"], kingdom),
        end_buy_priority=_full_buy_priority(["Province", "Duchy", "Estate", "Gardens", "Copper"], kingdom),
        early_nonterminal_priority=nt[:],
        early_terminal_priority=t[:],
        mid_nonterminal_priority=nt[:],
        mid_terminal_priority=t[:],
        late_nonterminal_priority=nt[:],
        late_terminal_priority=t[:],
        end_nonterminal_priority=nt[:],
        end_terminal_priority=t[:],
        early_chapel_trash=["STOP"],
        mid_chapel_trash=["STOP"],
        late_chapel_trash=["STOP"],
        end_chapel_trash=["STOP"],
        throne_room_priority=tr_candidates,
        mine_trash_priority=_default_mine_trash_priority(),
        chapel_max_trash=0,
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=2, late_to_end_provinces=1),
        buy_targets={"Gardens": 8} if kingdom is None or "Gardens" in kingdom else {},
    )


def engine_strategy(kingdom: list[str] | None = None) -> Strategy:
    """Seed archetype: Chapel engine — thin deck early, build mid, green late."""
    actions = _action_cards(kingdom)
    tr_candidates = [c for c in actions if c != "Throne Room"]
    all_targets = {"Chapel": 1, "Smithy": 2, "Village": 3, "Market": 2,
                   "Laboratory": 3, "Festival": 2, "Council Room": 2,
                   "Moneylender": 1, "Throne Room": 2, "Mine": 1, "Merchant": 2}
    kingdom_set = set(kingdom) if kingdom is not None else None
    buy_targets = {k: v for k, v in all_targets.items()
                   if kingdom_set is None or k in kingdom_set}
    # Phase-specific action priorities: Chapel/Moneylender first early, draw first mid/late
    early_actions = _prioritize(["Chapel", "Moneylender", "Village"], actions)
    mid_actions = _prioritize(["Village", "Laboratory", "Smithy", "Market"], actions)
    late_actions = _prioritize(["Village", "Laboratory", "Smithy", "Market"], actions)
    early_nt, early_t = _split_priority(early_actions)
    mid_nt, mid_t = _split_priority(mid_actions)
    late_nt, late_t = _split_priority(late_actions)
    end_actions = _prioritize(["Village", "Laboratory", "Smithy"], actions)
    end_nt, end_t = _split_priority(end_actions)
    return Strategy(
        early_buy_priority=_full_buy_priority(["Chapel", "Silver", "Village"], kingdom),
        mid_buy_priority=_full_buy_priority(["Gold", "Smithy", "Silver", "Market", "Laboratory"], kingdom),
        late_buy_priority=_full_buy_priority(["Province", "Duchy", "Gold", "Estate"], kingdom),
        end_buy_priority=_full_buy_priority(["Province", "Duchy", "Estate"], kingdom),
        early_nonterminal_priority=early_nt,
        early_terminal_priority=early_t,
        mid_nonterminal_priority=mid_nt,
        mid_terminal_priority=mid_t,
        late_nonterminal_priority=late_nt,
        late_terminal_priority=late_t,
        end_nonterminal_priority=end_nt,
        end_terminal_priority=end_t,
        early_chapel_trash=["Curse", "Estate", "Copper", "STOP"],
        mid_chapel_trash=["Curse", "Estate", "STOP", "Copper"],
        late_chapel_trash=["Curse", "STOP"],
        end_chapel_trash=["Curse", "STOP"],
        throne_room_priority=tr_candidates,
        mine_trash_priority=["Copper", "Silver"],
        chapel_max_trash=4,
        transitions=Transitions(early_to_mid_turn=5, mid_to_late_provinces=4, late_to_end_provinces=2),
        buy_targets=buy_targets,
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
    lines.append(f"MID   (>{t.mid_to_late_provinces} Prov or <t{t.mid_to_late_turn}): {fmt_list(strategy.mid_buy_priority)}")
    lines.append(f"LATE  ({t.late_to_end_provinces+1}-{t.mid_to_late_provinces} Prov): {fmt_list(strategy.late_buy_priority)}")
    lines.append(f"END   (<={t.late_to_end_provinces} Prov):    {fmt_list(strategy.end_buy_priority)}")
    lines.append(f"Early NonTerm: {fmt_list(strategy.early_nonterminal_priority)}")
    lines.append(f"Early Terminal: {fmt_list(strategy.early_terminal_priority)}")
    lines.append(f"Mid NonTerm:   {fmt_list(strategy.mid_nonterminal_priority)}")
    lines.append(f"Mid Terminal:   {fmt_list(strategy.mid_terminal_priority)}")
    lines.append(f"Late NonTerm:  {fmt_list(strategy.late_nonterminal_priority)}")
    lines.append(f"Late Terminal:  {fmt_list(strategy.late_terminal_priority)}")
    lines.append(f"End NonTerm:   {fmt_list(strategy.end_nonterminal_priority)}")
    lines.append(f"End Terminal:   {fmt_list(strategy.end_terminal_priority)}")
    if strategy.throne_room_priority:
        lines.append(f"Throne Room doubles: {fmt_list(strategy.throne_room_priority)}")
    if strategy.mine_trash_priority:
        lines.append(f"Mine upgrades: {' > '.join(strategy.mine_trash_priority)}")
    lines.append(f"Early Chapel trash: {fmt_list(strategy.early_chapel_trash)} (max {strategy.chapel_max_trash})")
    lines.append(f"Mid Chapel trash:   {fmt_list(strategy.mid_chapel_trash)}")
    lines.append(f"Late Chapel trash:  {fmt_list(strategy.late_chapel_trash)}")
    if strategy.buy_targets:
        targets = ", ".join(f"{c}:{n}" for c, n in strategy.buy_targets.items() if n < 99)
        lines.append(f"Buy targets: {targets}")
    if strategy.province_max_coins < 99 or strategy.duchy_max_coins < 99:
        lines.append(f"Coin thresholds: Province≤${strategy.province_max_coins}, Duchy≤${strategy.duchy_max_coins}")
    lines.append(f"Militia discard threshold: ${strategy.militia_coin_threshold}")
    return "\n".join(lines)


def summarize(strategy: Strategy, vs_bm: dict) -> str:
    """Plain-English summary of what the evolved strategy does and why it works."""
    from core.cards import ALL_CARDS, CardType

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
                 f">{t.mid_to_late_provinces} Provinces remain and turn <{t.mid_to_late_turn}")
    lines.append(f"  Late game:  {t.mid_to_late_provinces} or fewer Provinces left, "
                 f">{t.late_to_end_provinces} remaining")
    lines.append(f"  End game:   {t.late_to_end_provinces} or fewer Provinces left")

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
                                  ("Late", strategy.late_buy_priority),
                                  ("End", strategy.end_buy_priority)]:
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
    lines.append("  Action Play Order (non-terminals first, then terminals)")
    lines.append(f"  {'─' * 40}")
    for label, nt_list, t_list in [("Early", strategy.early_nonterminal_priority, strategy.early_terminal_priority),
                                    ("Mid", strategy.mid_nonterminal_priority, strategy.mid_terminal_priority),
                                    ("Late", strategy.late_nonterminal_priority, strategy.late_terminal_priority),
                                    ("End", strategy.end_nonterminal_priority, strategy.end_terminal_priority)]:
        if nt_list:
            lines.append(f"  {label} non-term: {' > '.join(nt_list[:6])}")
        if t_list:
            lines.append(f"  {label} terminal: {' > '.join(t_list[:6])}")

    # --- Chapel strategy ---
    lines.append("")
    lines.append("  Chapel Trashing")
    lines.append(f"  {'─' * 40}")
    if strategy.chapel_max_trash == 0:
        lines.append("  Never trashes (max 0)")
    else:
        for label, trash_list in [("Early", strategy.early_chapel_trash),
                                   ("Mid", strategy.mid_chapel_trash),
                                   ("Late", strategy.late_chapel_trash),
                                   ("End", strategy.end_chapel_trash)]:
            stop_idx = (trash_list.index("STOP")
                        if "STOP" in trash_list else None)
            if stop_idx == 0:
                lines.append(f"  {label}: never trashes")
            elif stop_idx is not None:
                trash_targets = trash_list[:stop_idx]
                lines.append(f"  {label}: {' > '.join(trash_targets)} (max {strategy.chapel_max_trash}/play)")
            else:
                lines.append(f"  {label}: {' > '.join(trash_list)} (max {strategy.chapel_max_trash}/play)")

    # --- Buy targets ---
    if strategy.buy_targets:
        lines.append("")
        lines.append("  Buy Targets (max copies)")
        lines.append(f"  {'─' * 40}")
        parts = [f"{card}: {n}" for card, n in strategy.buy_targets.items()]
        lines.append(f"  {', '.join(parts)}")

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
    from viz.plotting import plot_buy_heatmap

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
    # Backward compat: old models without newer genes get defaults
    throne_room_priority = data.get("throne_room_priority",
                                    [c for c in ACTION_CARDS if c != "Throne Room"])
    mine_trash_priority = data.get("mine_trash_priority", ["Copper", "Silver"])

    # Phase-specific action priorities (backward compat: split old flat lists)
    if "early_nonterminal_priority" in data:
        early_nt = data["early_nonterminal_priority"]
        early_t = data["early_terminal_priority"]
        mid_nt = data["mid_nonterminal_priority"]
        mid_t = data["mid_terminal_priority"]
        late_nt = data["late_nonterminal_priority"]
        late_t = data["late_terminal_priority"]
    elif "early_action_priority" in data:
        early_nt, early_t = _split_priority(data["early_action_priority"])
        mid_nt, mid_t = _split_priority(data["mid_action_priority"])
        late_nt, late_t = _split_priority(data["late_action_priority"])
    else:
        ap = data.get("action_priority", ACTION_CARDS[:])
        early_nt, early_t = _split_priority(ap)
        mid_nt, mid_t = _split_priority(ap)
        late_nt, late_t = _split_priority(ap)

    # End phase action priorities (backward compat: default to late)
    end_nt = data.get("end_nonterminal_priority", list(late_nt))
    end_t = data.get("end_terminal_priority", list(late_t))

    # Phase-specific chapel trash (backward compat from single chapel_trash_priority)
    if "early_chapel_trash" in data:
        early_ct = data["early_chapel_trash"]
        mid_ct = data["mid_chapel_trash"]
        late_ct = data["late_chapel_trash"]
    else:
        ct = data.get("chapel_trash_priority", ["Estate", "Copper", "STOP"])
        early_ct = list(ct)
        mid_ct = list(ct)
        late_ct = list(ct)
    end_ct = data.get("end_chapel_trash", ["STOP"])

    # Transitions backward compat
    transitions_data = data["transitions"]
    if "late_to_end_provinces" not in transitions_data:
        transitions_data = dict(transitions_data, late_to_end_provinces=2)
    if "mid_to_late_turn" not in transitions_data:
        transitions_data = dict(transitions_data, mid_to_late_turn=20)

    return Strategy(
        early_buy_priority=data["early_buy_priority"],
        mid_buy_priority=data["mid_buy_priority"],
        late_buy_priority=data["late_buy_priority"],
        end_buy_priority=data.get("end_buy_priority", list(data["late_buy_priority"])),
        early_nonterminal_priority=early_nt,
        early_terminal_priority=early_t,
        mid_nonterminal_priority=mid_nt,
        mid_terminal_priority=mid_t,
        late_nonterminal_priority=late_nt,
        late_terminal_priority=late_t,
        end_nonterminal_priority=end_nt,
        end_terminal_priority=end_t,
        early_chapel_trash=early_ct,
        mid_chapel_trash=mid_ct,
        late_chapel_trash=late_ct,
        end_chapel_trash=end_ct,
        throne_room_priority=throne_room_priority,
        mine_trash_priority=mine_trash_priority,
        chapel_max_trash=data.get("chapel_max_trash", 4),
        transitions=Transitions(**transitions_data),
        buy_targets=data.get("buy_targets", {}),
        province_max_coins=data.get("province_max_coins", 99),
        duchy_max_coins=data.get("duchy_max_coins", 99),
        militia_coin_threshold=data.get("militia_coin_threshold", 5),
    )
