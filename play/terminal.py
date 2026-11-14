"""Interactive play mode: you against an evolved model, on the shared engine (engine/).

The engine asks every decision; you answer yours at the prompt, the model answers its own.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter

import core  # noqa: F401
from core.agent import GAAgent
from core.cards import KINGDOM_CARDS
from core.strategy import Strategy, big_money_strategy, describe, load_strategy
from dominion_engine import Game
from dominion_engine import cards as K
from dominion_engine import game as G


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
# Display and prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    G.PLAY_ACTION: "Play an action", G.BUY: "Buy a card", G.THRONE: "Throne Room: play twice",
    G.CHAPEL_TRASH: "Chapel: trash a card", G.MONEYLENDER_TRASH: "Moneylender: trash a Copper for +$3?",
    G.MINE_TRASH: "Mine: trash a treasure", G.MINE_GAIN: "Mine: gain a treasure to hand",
    G.REMODEL_TRASH: "Remodel: trash a card", G.REMODEL_GAIN: "Remodel: gain a card",
    G.WORKSHOP_GAIN: "Workshop: gain a card", G.ARTISAN_GAIN: "Artisan: gain a card to hand",
    G.ARTISAN_TOPDECK: "Artisan: put a card from hand on your deck",
    G.CELLAR_DISCARD: "Cellar: discard a card (draw one per discard)",
    G.HARBINGER_TOPDECK: "Harbinger: put a card from your discard pile on your deck",
    G.VASSAL_PLAY: "Vassal: play the discarded card?", G.LIBRARY_SKIP: "Library: set this action aside?",
    G.SENTRY_CARD: "Sentry: 0 = trash, 1 = discard, 2 = put back", G.SENTRY_TOP: "Sentry: which card goes on top",
    G.POACHER_DISCARD: "Poacher: discard a card", G.MILITIA_DISCARD: "Militia: discard a card",
    G.BUREAUCRAT_TOPDECK: "Bureaucrat: put a Victory card on your deck",
    G.BANDIT_TRASH: "Bandit: trash one of these",
}


def _card_str(c: int) -> str:
    card = K.CARDS[c]
    tag = "A" if card.is_action else "$" if card.is_treasure else "V"
    return f"{card.name}({tag}{card.cost})"


def _counts(cards: list[int]) -> str:
    return ", ".join(f"{n}x {K.NAME[c]}" for c, n in Counter(cards).most_common())


def show_supply(g: Game) -> None:
    piles = [f"{K.NAME[c]}:{n}" for c, n in enumerate(g.supply) if n >= 0]
    print("\n  Supply: " + "  ".join(piles))


class HumanAgent:
    def __init__(self):
        self.turn_seen = -1

    def decide(self, g: Game, d: G.Decision) -> int:
        p = g.players[d.player]
        if d.kind in (G.PLAY_ACTION, G.BUY) and g.players[d.player].turns != self.turn_seen:
            self.turn_seen = p.turns
            show_supply(g)
            print(f"\n{'═' * 50}\n  Your turn {p.turns + 1}   Provinces left: {g.provinces_left}\n{'═' * 50}")
        if d.kind == G.SENTRY_CARD:
            labels = ["trash", "discard", "put back"]
        else:
            labels = [_card_str(o) for o in d.options]
        print(f"\n  Hand: {', '.join(K.NAME[c] for c in p.hand) or '-'}   "
              f"actions {g.actions}  buys {g.buys}  ${g.coins}")
        subject = f" [{K.NAME[d.subject]}]" if d.subject >= 0 and d.subject in K.NAME and d.kind in (
            G.VASSAL_PLAY, G.LIBRARY_SKIP, G.SENTRY_CARD) else ""
        print(f"  {PROMPTS[d.kind]}{subject}:")
        for i, label in enumerate(labels, 1):
            print(f"    {i}. {label}")
        if d.may_pass:
            print("    0. none / done")
        while True:
            try:
                raw = input("  > ").strip() or ("0" if d.may_pass else "1")
                i = int(raw)
                if i == 0 and d.may_pass:
                    return -1
                if 1 <= i <= len(d.options):
                    return d.options[i - 1]
            except (ValueError, EOFError):
                pass
            print("  Invalid choice, try again.")


class NarratedAgent(GAAgent):
    """The model, printing what it plays and buys."""

    def decide(self, g: Game, d: G.Decision) -> int:
        ans = super().decide(g, d)
        if ans >= 0 and d.kind in (G.PLAY_ACTION, G.BUY):
            verb = "plays" if d.kind == G.PLAY_ACTION else "buys"
            print(f"  AI {verb} {K.NAME[ans]}")
        elif d.kind in (G.MILITIA_DISCARD, G.BUREAUCRAT_TOPDECK, G.BANDIT_TRASH):
            print(f"  AI {PROMPTS[d.kind].split(':')[1].strip().lower()}: {K.NAME[ans]}")
        return ans


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def model_kingdom(strategy: Strategy) -> list[str]:
    """The kingdom a model was trained on (its buy lists), padded to 10 at random."""
    known = sorted({c for c in strategy.early_buy_priority + strategy.mid_buy_priority
                    + strategy.late_buy_priority if c in KINGDOM_CARDS})
    rest = [c for c in KINGDOM_CARDS if c not in known]
    return known[:10] + random.sample(rest, max(0, 10 - len(known)))


def play_interactive(opponent: Strategy, kingdom: list[str] | None = None,
                     seed: int | None = None) -> None:
    """You (seat 0) against the model (seat 1)."""
    kingdom = kingdom or model_kingdom(opponent)
    seed = random.randint(0, 2**31) if seed is None else seed
    g = Game(kingdom, seed)
    agents = [HumanAgent(), NarratedAgent(opponent)]
    print("\n" + "=" * 50 + "\n  GAME START\n" + "=" * 50)
    print(f"  Kingdom: {', '.join(sorted(kingdom))}")
    gen = g.run()
    try:
        d = next(gen)
        while True:
            d = gen.send(agents[d.player].decide(g, d))
    except StopIteration as stop:
        r = stop.value
    print("\n" + "=" * 50 + "\n  GAME OVER\n" + "=" * 50)
    print(f"  Turns: you {r.turns[0]}, AI {r.turns[1]}{'  (turn cap reached)' if r.capped else ''}")
    print(f"  Your VP: {r.scores[0]}   AI VP: {r.scores[1]}\n")
    print("  *** TIE ***" if len(r.winners) == 2 else
          "  *** YOU WIN! ***" if r.winners == [0] else "  *** AI WINS ***")
    print(f"\n  Your deck: {_counts(r.decks[0])}\n  AI deck:   {_counts(r.decks[1])}")


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

    print(describe(opponent))
    print()

    while True:
        play_interactive(opponent)
        again = input("\nPlay again vs same opponent? [Y/n]: ").strip().lower()
        if again and again != "y":
            break


if __name__ == "__main__":
    main()
