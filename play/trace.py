"""Game trace — watch the AI model play turn-by-turn (on the shared engine, engine/).

Usage:
    python trace.py                    # best model vs Big Money
    python trace.py --seed 123         # specific seed
    python trace.py --vs self          # best model vs itself
    python trace.py --vs prev          # best model vs previous gen
    python trace.py --vs bigmoney      # best model vs Big Money
    python trace.py --vs 42            # best model vs gen 42
    python trace.py --vs gen_042/strategy.json  # explicit path
    python trace.py --list             # list available generations
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import Counter

import core  # noqa: F401
from core.agent import GAAgent
from core.cards import ALL_CARDS, CardType
from core.strategy import Strategy, big_money_strategy, get_current_phase, load_strategy
from dominion_engine import Game
from dominion_engine import cards as K
from dominion_engine import game as G
from play.terminal import model_kingdom


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

WIDTH = 90          # total terminal width
COL = 42            # width of each player column


def _left(text: str) -> str:
    return text[:COL].ljust(COL)


def _right(text: str) -> str:
    return " " * (WIDTH - COL) + text[:COL]


def _wrap(text: str, max_width: int, indent: int = 4) -> list[str]:
    words, lines, line = text.split(" "), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_width:
            lines.append(line)
            line = " " * indent + w
        else:
            line = f"{line} {w}" if line else w
    return lines + [line] if line else lines


def _center(text: str) -> str:
    return text.center(WIDTH)


def _hand_str(cards: list[int]) -> str:
    parts = []
    for c, n in sorted(Counter(cards).items(), key=lambda x: -K.CARDS[x[0]].cost):
        card = K.CARDS[c]
        tag = "A" if card.is_action else "$" if card.is_treasure else "V"
        parts.append(f"{n}x{card.name}({tag})" if n > 1 else f"{card.name}({tag})")
    return ", ".join(parts)


def _deck_summary_by_type(cards: list[int]) -> tuple[str, str, str]:
    vp, treasure, action = [], [], []
    for c, n in Counter(cards).most_common():
        card = K.CARDS[c]
        entry = f"{n}x{card.name}" if n > 1 else card.name
        (action if card.is_action else treasure if card.is_treasure else vp).append(entry)
    return ", ".join(vp) or "—", ", ".join(treasure) or "—", ", ".join(action) or "—"


VERBS = {G.PLAY_ACTION: "Play", G.THRONE: "Throne Room ×2", G.CHAPEL_TRASH: "Chapel trashes",
         G.MONEYLENDER_TRASH: "Moneylender trashes", G.MINE_TRASH: "Mine trashes", G.MINE_GAIN: "Mine gains",
         G.REMODEL_TRASH: "Remodel trashes", G.REMODEL_GAIN: "Remodel gains", G.WORKSHOP_GAIN: "Workshop gains",
         G.ARTISAN_GAIN: "Artisan gains", G.ARTISAN_TOPDECK: "Artisan topdecks", G.CELLAR_DISCARD: "Cellar discards",
         G.HARBINGER_TOPDECK: "Harbinger topdecks", G.VASSAL_PLAY: "Vassal plays", G.LIBRARY_SKIP: "Library skips",
         G.SENTRY_TOP: "Sentry tops", G.POACHER_DISCARD: "Poacher discards", G.MILITIA_DISCARD: "discards",
         G.BUREAUCRAT_TOPDECK: "topdecks", G.BANDIT_TRASH: "loses"}


# ---------------------------------------------------------------------------
# Traced 2-player game
# ---------------------------------------------------------------------------

def trace_game(strategy1: Strategy, strategy2: Strategy,
               label1: str = "Model", label2: str = "Opponent",
               kingdom: list[str] | None = None,
               seed: int | None = None) -> dict:
    kingdom = kingdom or model_kingdom(strategy1)
    seed = random.randint(0, 2**31) if seed is None else seed
    g = Game(kingdom, seed)
    agents = [GAAgent(strategy1), GAAgent(strategy2)]
    labels, strategies = [label1, label2], [strategy1, strategy2]

    print()
    print("=" * WIDTH)
    print(_center(f"GAME TRACE  (seed={seed})"))
    print(_center(f"Kingdom: {', '.join(sorted(kingdom))}"))
    print("=" * WIDTH)
    print()
    print(_left(f"  {label1}") + _right(f"  {label2}").lstrip())
    print(_left("  " + "─" * (COL - 2)) + _right("  " + "─" * (COL - 2)).lstrip())
    print()

    lines: list[str] = []
    key = None

    def flush():
        if key is None:
            return
        who = key[0]
        if who == 0:
            print(_center(f"── Turn {key[1] + 1} ──"))
        for line in lines:
            for part in _wrap("  " + line, COL):
                print(_left(part) if who == 0 else _right(part))
        print()

    gen = g.run()
    try:
        d = next(gen)
        while True:
            cur = (g.current, g.players[g.current].turns)
            if cur != key:
                flush()
                key, lines = cur, []
                p = g.players[g.current]
                phase = get_current_phase(p.turns + 1, g.provinces_left, strategies[g.current].transitions)
                lines += [f"[{phase}] {g.provinces_left} Prov left",
                          f"Hand: {_hand_str(p.hand + p.play)}"]
            ans = agents[d.player].decide(g, d)
            if d.kind == G.BUY:
                if ans >= 0:
                    lines.append(f"Buy {K.NAME[ans]} (${g.coins})")
                elif not any("Buy" in x for x in lines[2:]):
                    lines.append(f"Buy nothing (${g.coins})")
            elif d.kind == G.SENTRY_CARD:
                lines.append(f"Sentry: {['trash', 'discard', 'keep'][ans]} {K.NAME[d.subject]}")
            elif ans >= 0:
                who = "" if d.player == g.current else f"{labels[d.player]} "
                lines.append(f"{who}{VERBS[d.kind]} {K.NAME[ans]}")
            d = gen.send(ans)
    except StopIteration as stop:
        r = stop.value
    flush()
    return _game_result(r, labels)


def _game_result(r, labels):
    print("=" * WIDTH)
    print(_center(f"GAME OVER  ·  {r.turns[0]} turns"))
    print("=" * WIDTH)
    print()
    print(_left(f"  {labels[0]}: {r.scores[0]} VP") + _right(f"  {labels[1]}: {r.scores[1]} VP").lstrip())
    print()
    d1, d2 = _deck_summary_by_type(r.decks[0]), _deck_summary_by_type(r.decks[1])
    for label, row1, row2 in [("VP", d1[0], d2[0]), ("$", d1[1], d2[1]), ("A", d1[2], d2[2])]:
        print(_left(f"  {label}: {row1}") + _right(f"  {label}: {row2}").lstrip())
    print()
    if len(r.winners) == 2:
        print(_center(">>> Tie! <<<"))
    else:
        tb = " (fewer turns)" if r.scores[0] == r.scores[1] else ""
        print(_center(f">>> {labels[r.winners[0]]} wins{tb}! <<<"))
    print()
    return {"vp1": r.scores[0], "vp2": r.scores[1], "turns": r.turns[0], "winners": r.winners}


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_gens(model_dir: str = "best_model") -> list[int]:
    """Return sorted list of generation numbers found under model_dir."""
    gens = []
    if os.path.isdir(model_dir):
        for entry in os.listdir(model_dir):
            m = re.match(r"gen_(\d+)", entry)
            if m:
                path = os.path.join(model_dir, entry, "strategy.json")
                if os.path.isfile(path):
                    gens.append(int(m.group(1)))
    return sorted(gens)


def _resolve_opponent(vs: str, model_dir: str = "best_model") -> tuple[Strategy, str]:
    """Resolve --vs argument to (strategy, label).

    Accepts: 'bigmoney', 'self', 'prev', a gen number like '42',
    or a path to a strategy.json file.
    """
    if vs == "bigmoney":
        return big_money_strategy(), "Big Money"

    if vs == "self":
        strat = load_strategy(os.path.join(model_dir, "strategy.json"))
        return strat, "Model (copy)"

    if vs == "prev":
        gens = _find_gens(model_dir)
        if len(gens) < 2:
            print("Not enough generations for --vs prev, falling back to Big Money")
            return big_money_strategy(), "Big Money"
        prev_gen = gens[-2]
        path = os.path.join(model_dir, f"gen_{prev_gen:03d}", "strategy.json")
        return load_strategy(path), f"Gen {prev_gen}"

    # Try as a gen number
    if vs.isdigit():
        gen = int(vs)
        # Try both zero-padded and unpadded
        for fmt in [f"gen_{gen:03d}", f"gen_{gen}"]:
            path = os.path.join(model_dir, fmt, "strategy.json")
            if os.path.isfile(path):
                return load_strategy(path), f"Gen {gen}"
        print(f"Gen {gen} not found. Available: {', '.join(str(g) for g in _find_gens(model_dir)[-10:])}")
        raise SystemExit(1)

    # Try as a file path
    if os.path.isfile(vs):
        label = os.path.basename(os.path.dirname(vs)) or vs
        return load_strategy(vs), label

    print(f"Unknown --vs value: {vs}")
    raise SystemExit(1)


def _list_gens(model_dir: str = "best_model") -> None:
    """Print available generations."""
    gens = _find_gens(model_dir)
    if not gens:
        print("No saved generations found.")
        return
    print(f"Available generations ({len(gens)} total):")
    # Show in rows of 10
    for i in range(0, len(gens), 10):
        row = gens[i:i + 10]
        print("  " + "  ".join(f"{g:>5d}" for g in row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _eval_vs_opponents(strategy: Strategy, opponents: list[tuple[Strategy, str]],
                       num_games: int = 100,
                       kingdom: list[str] | None = None,
                       title: str = "EVALUATION") -> None:
    """Play the model against a list of opponents, print results table."""
    from core.engine import play_game_2p

    if kingdom is None:
        kingdom = model_kingdom(strategy)

    if not opponents:
        print("No opponents found.")
        return

    rng = random.Random(42)
    seeds = [rng.randint(0, 2**31) for _ in range(num_games)]

    print()
    print("=" * WIDTH)
    print(_center(title))
    print(_center(f"{num_games} games per opponent (seat-swapped)"))
    print(_center(f"Kingdom: {', '.join(kingdom)}"))
    print("=" * WIDTH)
    print()

    # Header
    print(f"  {'Opponent':<16s} {'Win':>5s} {'Tie':>5s} {'Loss':>5s}"
          f"  {'Turns':>5s}  {'VP±':>5s}  Action Cards (avg)")
    print(f"  {'─' * 78}")

    total_wins = 0
    total_ties = 0
    total_games = 0

    for opp_strat, opp_label in opponents:
        wins = ties = 0
        total_turns = 0
        total_vp_margin = 0
        deck_counts: dict[str, int] = {}
        n_games = 0

        for s in seeds:
            # Game 1: model as P1
            r = play_game_2p(strategy, opp_strat, s, kingdom)
            total_turns += r["turns"]
            total_vp_margin += r["vp1"] - r["vp2"]
            if r["winners"] == [0]: wins += 1
            elif len(r["winners"]) == 2: ties += 1
            for card in r["deck1"]:
                deck_counts[card] = deck_counts.get(card, 0) + 1
            n_games += 1

            # Game 2: model as P2
            r = play_game_2p(opp_strat, strategy, s, kingdom)
            total_turns += r["turns"]
            total_vp_margin += r["vp2"] - r["vp1"]
            if r["winners"] == [1]: wins += 1
            elif len(r["winners"]) == 2: ties += 1
            for card in r["deck2"]:
                deck_counts[card] = deck_counts.get(card, 0) + 1
            n_games += 1

        n = n_games
        losses = n - wins - ties
        avg_turns = total_turns / n
        avg_margin = total_vp_margin / n
        total_wins += wins
        total_ties += ties
        total_games += n

        # Top action cards in final deck
        avg_deck = sorted(deck_counts.items(), key=lambda x: -x[1])
        avg_deck = [(name, count) for name, count in avg_deck
                     if ALL_CARDS[name].card_type == CardType.ACTION and count / n >= 0.5]
        deck_str = ", ".join(f"{count/n:.0f}x{name}" for name, count in avg_deck[:6])

        print(f"  {opp_label:<16s} {wins/n:>4.0%} {ties/n:>5.0%} {losses/n:>5.0%}"
              f"  {avg_turns:>5.1f}  {avg_margin:>+5.1f}  {deck_str}")

    # Summary
    print(f"  {'─' * 78}")
    total_losses = total_games - total_wins - total_ties
    print(f"  {'OVERALL':<16s} {total_wins/total_games:>4.0%}"
          f" {total_ties/total_games:>5.0%} {total_losses/total_games:>5.0%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Trace a game played by the AI model")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--model", default="best_model/strategy.json",
                        help="Path to strategy JSON")
    parser.add_argument("--vs", default="bigmoney",
                        help="Opponent: bigmoney, self, prev, hall, gens, a gen number, or a path")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per opponent for --vs hall/gens (default: 100)")
    parser.add_argument("--list", action="store_true",
                        help="List available generations and exit")
    args = parser.parse_args()

    if args.list:
        _list_gens()
        return

    strategy = load_strategy(args.model)
    model_dir = os.path.dirname(args.model) or "best_model"

    if args.vs == "hall":
        # Load saved hall of fame members
        hall_dir = os.path.join(model_dir, "hall")
        opponents: list[tuple[Strategy, str]] = []
        if os.path.isdir(hall_dir):
            for f in sorted(os.listdir(hall_dir)):
                if f.endswith(".json"):
                    path = os.path.join(hall_dir, f)
                    idx = f.replace("hall_", "").replace(".json", "")
                    label = "Big Money" if idx == "0" else f"Hall #{idx}"
                    opponents.append((load_strategy(path), label))
        if not opponents:
            print("No saved hall found. Adding Big Money as fallback.")
            opponents = [(big_money_strategy(), "Big Money")]
        _eval_vs_opponents(strategy, opponents, num_games=args.games,
                           title="HALL OF FAME EVALUATION")
        return

    if args.vs == "gens":
        # Play against Big Money + every 4th saved generation
        opponents = [(big_money_strategy(), "Big Money")]
        gens = _find_gens(model_dir)
        for gen in gens[::4]:  # every 4th generation
            path = os.path.join(model_dir, f"gen_{gen:03d}", "strategy.json")
            opponents.append((load_strategy(path), f"Gen {gen}"))
        _eval_vs_opponents(strategy, opponents, num_games=args.games,
                           title="EVALUATION VS GENERATIONS (every 4th)")
        return

    opponent, opp_label = _resolve_opponent(args.vs, model_dir)

    trace_game(strategy, opponent,
               label1="Model", label2=opp_label,
               seed=args.seed)


if __name__ == "__main__":
    main()
