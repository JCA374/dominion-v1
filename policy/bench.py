"""Benchmark: how well a model plays against fixed reference bots on the kingdoms it will meet.

    python -m policy.bench [MODEL ...]           # default: every released model
    python -m policy.bench k1-steam best_model --kingdoms 100 --seeds 10
    python -m policy.bench --json bench.json     # also write the numbers

The hall-of-fame win rate from training only says a model beats other evolved models on
its own training kingdom. This asks the questions that matter for the Steam game:

  opponents   Big Money, Smithy-Big Money, Witch-Big Money (fixed, never change)
  kingdoms    the model's training kingdom, and random Steam-like kingdoms: 10 cards drawn
              from the 26-card base set (2nd edition, as Steam deals them), of which the
              engine plays only the cards it knows. Smithy-BM and Witch-BM get kingdoms
              that contain their card.

Score = (wins + ties/2) / games, every seed played from both seats, on the C engine.
The same kingdoms and seeds are used for every model, so numbers are comparable between
models and runs. "big-money" is always added as a reference row.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from core.cards import KINGDOM_CARDS
from core.strategy import (Strategy, big_money_strategy, load_strategy,
                           smithy_big_money_strategy, witch_big_money_strategy)
from ga.fitness import evaluate_vs_opponent
from policy.registry import list_models, read_manifest, resolve

# Steam Dominion's base set (2nd edition): the pool its random kingdoms are drawn from.
STEAM_BASE_SET = [
    "Cellar", "Chapel", "Moat", "Harbinger", "Merchant", "Vassal", "Village", "Workshop",
    "Bureaucrat", "Gardens", "Militia", "Moneylender", "Poacher", "Remodel", "Smithy",
    "Throne Room", "Bandit", "Council Room", "Festival", "Laboratory", "Library", "Market",
    "Mine", "Sentry", "Witch", "Artisan",
]

# name -> (strategy builder, card the kingdom must contain or None)
OPPONENTS = {
    "Big Money": (big_money_strategy, None),
    "Smithy-BM": (smithy_big_money_strategy, "Smithy"),
    "Witch-BM": (witch_big_money_strategy, "Witch"),
}


def steam_kingdoms(n: int, seed: int, must: str | None = None) -> list[list[str]]:
    """n random 10-card base-set kingdoms, cut down to the cards the engine plays."""
    rng = random.Random(f"{seed}:{must}")
    out = []
    for _ in range(n):
        pool = [c for c in STEAM_BASE_SET if c != must]
        k = rng.sample(pool, 9 if must else 10) + ([must] if must else [])
        out.append(sorted(c for c in k if c in KINGDOM_CARDS))
    return out


def score(strategy: Strategy, opponent, kingdoms: list[list[str]], seeds: list[int]) -> dict:
    wins = ties = games = 0
    for k in kingdoms:
        r = evaluate_vs_opponent(strategy, seeds, k, opponent(k))
        n = 2 * len(seeds)
        wins += round(r["win_rate"] * n)
        ties += round(r["tie_rate"] * n)
        games += n
    return {"score": (wins + ties / 2) / games, "win": wins / games, "tie": ties / games,
            "games": games, "kingdoms": len(kingdoms)}


def bench(strategy: Strategy, train_kingdom: list[str] | None, n_kingdoms: int,
          n_seeds: int, seed: int = 1) -> dict:
    rng = random.Random(seed)
    seeds = [rng.getrandbits(32) for _ in range(n_seeds)]
    out = {}
    for name, (opponent, must) in OPPONENTS.items():
        row = {"steam": score(strategy, opponent, steam_kingdoms(n_kingdoms, seed, must), seeds)}
        if train_kingdom and (must is None or must in train_kingdom):
            row["training"] = score(strategy, opponent, [train_kingdom], seeds * 10)
        out[name] = row
    out["steam_mean"] = sum(out[n]["steam"]["score"] for n in OPPONENTS) / len(OPPONENTS)
    return out


def load(ref: str) -> tuple[str, Strategy, list[str] | None]:
    if ref == "big-money":
        return ref, big_money_strategy(), None
    manifest = read_manifest(resolve(ref))
    if manifest["kind"] != "ga-strategy":
        raise SystemExit(f"{ref}: only ga-strategy models can be benchmarked so far "
                         f"(kind {manifest['kind']})")
    return manifest["name"], load_strategy(manifest["path"]), manifest.get("kingdom")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--kingdoms", type=int, default=200, help="random Steam-like kingdoms per opponent")
    ap.add_argument("--seeds", type=int, default=20, help="games per kingdom (each played from both seats)")
    ap.add_argument("--seed", type=int, default=1, help="draws the kingdoms and game seeds")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    args = ap.parse_args(argv)

    refs = args.models or [m["name"] for m in list_models()]
    refs = ["big-money"] + [r for r in refs if r != "big-money"]
    pct = lambda x: f"{100 * x:5.1f}%"
    names = list(OPPONENTS)
    print(f"{args.kingdoms} Steam-like kingdoms x {args.seeds} seeds x 2 seats per opponent; "
          f"score = win + tie/2\n")
    print(f"{'model':<20}" + "".join(f"{n:>12}" for n in names) + f"{'mean':>9}"
          + "   training kingdom: " + " / ".join(names))
    results = {}
    for ref in refs:
        name, strategy, kingdom = load(ref)
        r = bench(strategy, kingdom, args.kingdoms, args.seeds, args.seed)
        results[name] = {"kingdom": kingdom, **r}
        train = " / ".join(pct(r[n]["training"]["score"]).strip() if "training" in r[n] else "-"
                           for n in names)
        print(f"{name:<20}" + "".join(f"{pct(r[n]['steam']['score']):>12}" for n in names)
              + f"{pct(r['steam_mean']):>9}   {train}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
