"""Key cards: the kingdom cards a model's game is built around, e.g. {"Smithy": 2, "Witch": 1}.

    python -m policy.keycards [MODEL ...]            # print them
    python -m policy.keycards --write MODEL ...      # store them in model.json

A model can play any kingdom that has its key cards, whatever the other cards are; the
model selector (policy/select.py) uses this. For a GA model they are measured, not
guessed: the model plays its training kingdom against Big Money and every kingdom card it
ends with at least KEEP copies of on average is a key card (at most MAX_KEY, the most
bought first), with the rounded average as its count. Any other model kind declares
"key_cards" in its model.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

KEEP = 0.75        # average copies at game end to count as a key card
MAX_KEY = 3
GAMES = 200        # seeds, each played from both seats


def derive(strategy, kingdom: list[str], games: int = GAMES, seed: int = 7) -> dict[str, int]:
    """Measure a GA strategy's key cards on its training kingdom (Python engine)."""
    from core.strategy import big_money_strategy
    from ga.fitness import evaluate_vs_opponent
    rng = random.Random(seed)
    seeds = [rng.getrandbits(32) for _ in range(games)]
    r = evaluate_vs_opponent(strategy, seeds, kingdom, big_money_strategy(kingdom), need_deck=True)
    mean = {c: n for c, n in r["avg_final_deck"].items() if c in kingdom and n >= KEEP}
    top = sorted(mean, key=lambda c: (-mean[c], c))[:MAX_KEY]
    return {c: max(1, round(mean[c])) for c in top}


def derive_for(manifest: dict) -> dict[str, int] | None:
    """Key cards of a model manifest, measured if it is a GA model (else None)."""
    if manifest.get("kind") != "ga-strategy" or not manifest.get("kingdom"):
        return None
    from core.strategy import load_strategy
    return derive(load_strategy(manifest["path"]), manifest["kingdom"])


def fmt(cards: dict[str, int] | None) -> str:
    if cards is None:
        return "?"
    return " + ".join(f"{n}x {c}" for c, n in cards.items()) or "(none: plays money)"


def main(argv=None) -> int:
    from policy.registry import list_models, read_manifest, resolve
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--write", action="store_true", help="store the measured key cards in model.json")
    args = ap.parse_args(argv)
    manifests = [read_manifest(resolve(r)) for r in args.models] if args.models else list_models()
    for m in manifests:
        cards = derive_for(m)
        print(f"{m['name']:<20} {fmt(cards)}")
        if args.write and cards is not None:
            path = os.path.join(m["dir"], "model.json")
            data = json.load(open(path))
            data["key_cards"] = cards
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
