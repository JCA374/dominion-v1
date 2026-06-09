"""Plotting functions for GA evolution visualization."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from core.cards import BUYABLE_CARDS, ALL_CARDS

if TYPE_CHECKING:
    from core.strategy import Strategy


def plot_fitness(log: list[dict], filename: str = "fitness.png") -> None:
    """Best/mean/worst win rate over generations."""
    gens = [e["gen"] for e in log]
    best = [e["best_win_rate"] for e in log]
    mean = [e["mean_win_rate"] for e in log]
    worst = [e["worst_win_rate"] for e in log]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, label="Best", linewidth=2)
    ax.plot(gens, mean, label="Mean", linewidth=1.5, linestyle="--")
    ax.plot(gens, worst, label="Worst", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Win Rate vs Big Money")
    ax.set_title("Fitness Over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_transitions(log: list[dict], filename: str = "transitions.png") -> None:
    """Transition genes of best individual over generations."""
    gens = [e["gen"] for e in log]
    early_mid = [e["early_to_mid_turn"] for e in log]
    mid_late = [e["mid_to_late_provinces"] for e in log]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1, color2 = "#1f77b4", "#ff7f0e"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("early_to_mid_turn", color=color1)
    ax1.plot(gens, early_mid, color=color1, linewidth=2, label="early→mid (turn)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(1, 16)

    ax2 = ax1.twinx()
    ax2.set_ylabel("mid_to_late_provinces", color=color2)
    ax2.plot(gens, mid_late, color=color2, linewidth=2, linestyle="--",
             label="mid→late (provinces left)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-1, 9)

    ax1.set_title("Phase Transition Genes Over Generations")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_buy_heatmap(strategy: Strategy, filename: str = "buy_heatmap.png") -> None:
    """Heatmap showing priority rank of each card per phase."""
    lists = [
        strategy.early_buy_priority,
        strategy.mid_buy_priority,
        strategy.late_buy_priority,
        strategy.end_buy_priority,
    ]

    # Derive card list from the strategy's actual priorities (preserves kingdom selection)
    seen = set()
    cards = []
    for priority_list in lists:
        for card in priority_list:
            if card not in seen:
                seen.add(card)
                cards.append(card)

    # Sort by cost (highest to lowest), PASS at bottom
    non_pass = [c for c in cards if c != "PASS"]
    non_pass.sort(key=lambda c: ALL_CARDS[c].cost, reverse=True)
    cards = non_pass + (["PASS"] if "PASS" in seen else [])

    data = np.full((len(cards), 4), np.nan)
    after_pass = np.zeros((len(cards), 4), dtype=bool)

    for col, priority_list in enumerate(lists):
        past_pass = False
        for rank, card_name in enumerate(priority_list):
            if card_name == "PASS":
                past_pass = True
            if card_name in seen:
                row = cards.index(card_name)
                data[row, col] = rank + 1  # 1-indexed rank
                if past_pass and card_name != "PASS":
                    after_pass[row, col] = True

    buy_targets = strategy.buy_targets

    fig, ax = plt.subplots(figsize=(6, max(4, len(cards) * 0.4)), layout="constrained")
    cmap = plt.cm.RdYlGn_r  # lower rank (higher priority) = green
    n_items = max(len(lst) for lst in lists)

    # Grey out cells after PASS by overlaying them
    display = data.copy()
    display[after_pass] = np.nan  # hide from colormap
    im = ax.imshow(display, cmap=cmap, aspect="auto", vmin=1, vmax=n_items)

    # Draw grey background for after-PASS cells
    for i in range(len(cards)):
        for j in range(4):
            if after_pass[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#e0e0e0", edgecolor="none"))

    ax.set_xticks(range(4))
    t = strategy.transitions
    ax.set_xticklabels([
        f"Early\n(turns 1–{t.early_to_mid_turn - 1})",
        f"Mid\n(turn {t.early_to_mid_turn}+,\n>{t.mid_to_late_provinces} prov\nor turn <{t.mid_to_late_turn})",
        f"Late\n(≤{t.mid_to_late_provinces},\n>{t.late_to_end_provinces} prov left)",
        f"End\n(≤{t.late_to_end_provinces} prov left)",
    ])

    # Left y-axis: card name (cost)
    ax.set_yticks(range(len(cards)))
    ax.set_yticklabels([f"{c} ({ALL_CARDS[c].cost})" if c != "PASS" else "PASS"
                        for c in cards])

    # Right y-axis: buy targets
    if buy_targets:
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(range(len(cards)))
        ax2.set_yticklabels([
            f"max {buy_targets[c]}" if c in buy_targets else ""
            for c in cards
        ])
        ax2.set_ylabel("Buy Targets")

    # Highlight PASS row with a different background
    if "PASS" in cards:
        pass_row = cards.index("PASS")
        ax.axhline(y=pass_row - 0.5, color="gray", linewidth=0.5, linestyle="--")

    # Annotate cells with rank
    for i in range(len(cards)):
        for j in range(4):
            if not np.isnan(data[i, j]):
                if after_pass[i, j]:
                    ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center",
                            fontsize=9, color="#aaaaaa", fontstyle="italic")
                else:
                    ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if data[i, j] > n_items / 2 else "black")

    ax.set_title("Buy Priority Rank by Phase (1 = highest)")
    fig.colorbar(im, ax=ax, label="Rank", shrink=0.6)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_game_trace(strategy: Strategy, seed: int,
                    filename: str = "game_trace.png") -> None:
    """Deck composition (treasure/action/VP count) turn by turn."""
    from core.engine import new_game, play_action_phase, play_buy_phase, cleanup, is_game_over
    from core.cards import ALL_CARDS, CardType

    state = new_game(seed)
    trace = {"turn": [], "treasure": [], "action": [], "victory": [], "total": []}

    while not is_game_over(state):
        state.turn += 1
        state.actions = 1
        state.buys = 1
        state.coins = 0

        play_action_phase(state, strategy)
        play_buy_phase(state, strategy)

        # Count all owned cards
        all_cards = state.deck + state.hand + state.discard + state.play_area
        t = sum(1 for c in all_cards if ALL_CARDS[c].card_type == CardType.TREASURE)
        a = sum(1 for c in all_cards if ALL_CARDS[c].card_type == CardType.ACTION)
        v = sum(1 for c in all_cards if ALL_CARDS[c].card_type == CardType.VICTORY)

        trace["turn"].append(state.turn)
        trace["treasure"].append(t)
        trace["action"].append(a)
        trace["victory"].append(v)
        trace["total"].append(len(all_cards))

        cleanup(state)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(trace["turn"], trace["treasure"], trace["action"], trace["victory"],
                 labels=["Treasure", "Action", "Victory"], alpha=0.8)
    ax.plot(trace["turn"], trace["total"], "k--", linewidth=1, label="Total")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Card Count")
    ax.set_title("Deck Composition Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_all_plots(log: list[dict], best_strategy: Strategy,
                   output_dir: str = ".") -> None:
    """Generate all plots and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    plot_fitness(log, os.path.join(output_dir, "fitness.png"))
    plot_transitions(log, os.path.join(output_dir, "transitions.png"))
    plot_buy_heatmap(best_strategy, os.path.join(output_dir, "buy_heatmap.png"))
    plot_game_trace(best_strategy, 0,
                    os.path.join(output_dir, "game_trace.png"))
    print(f"Plots saved to {output_dir}/")
