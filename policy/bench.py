"""Benchmark runner: how well each model plays against fixed reference bots, stored for the selector.

    python -m policy.bench [MODEL ...]           # default: every model in every model folder
    python -m policy.bench --store               # also write models/<name>/bench.json
    python -m policy.bench --store --missing     # only models with no (or an outdated) bench.json
    python -m policy.bench k1-steam best_model --kingdoms 100 --seeds 10 --json bench.json

The hall-of-fame win rate from training only says a model beats other evolved models on
its own training kingdom. This asks the questions that matter for the Steam game:

  opponents   Big Money, Smithy-Big Money, Witch-Big Money (fixed, never change)
  kingdoms    eligible: random Steam-like kingdoms that contain the model's key cards
                        (policy/keycards.py), the kingdoms the selector would pick it for
              random:   random Steam-like kingdoms, any cards
              training: the model's own training kingdom
              A Steam-like kingdom is 10 cards drawn from the 26-card base set (2nd edition,
              as Steam deals them), of which the engine plays only the cards it knows.
              Smithy-BM and Witch-BM get kingdoms that contain their card.

Score = (wins + ties/2) / games, every seed played from both seats, on the C engine. The
same kingdoms and seeds are used for every model with the same key cards, so numbers are
comparable between models and runs. "big-money" is always printed as a reference row.

bench.json holds "score" (the mean eligible score over the three opponents, which the
selector ranks by), the per-opponent numbers and the key cards it measured.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys

from core.cards import KINGDOM_CARDS
from core.strategy import (Strategy, big_money_strategy, load_strategy,
                           smithy_big_money_strategy, witch_big_money_strategy)
from ga.fitness import evaluate_vs_opponent
from policy.keycards import derive, fmt
from policy.registry import (BENCH_FILE, key_cards, list_models, read_bench, read_manifest,
                             resolve)

BENCH_VERSION = 2   # 2: engine with Steam rules (8 Provinces, all 26 cards, tiebreak)

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


def steam_kingdoms(n: int, seed: int, must=None) -> list[list[str]]:
    """n random 10-card base-set kingdoms containing every card in `must` (a card or a list),
    (the engine knows all 26 cards)."""
    must = [must] if isinstance(must, str) else sorted(set(must or []))
    rng = random.Random(f"{seed}:{must[0] if len(must) == 1 else must or None}")
    pool = [c for c in STEAM_BASE_SET if c not in must]
    out = []
    for _ in range(n):
        k = rng.sample(pool, 10 - len(must)) + must
        out.append(sorted(k))
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
          n_seeds: int, seed: int = 1, keys: dict[str, int] | None = None) -> dict:
    """keys: the model's key cards (None = unknown: eligible means its training kingdom only)."""
    rng = random.Random(seed)
    seeds = [rng.getrandbits(32) for _ in range(n_seeds)]
    in_steam = keys is not None and all(c in STEAM_BASE_SET for c in keys)
    out = {}
    for name, (opponent, must) in OPPONENTS.items():
        row = {"steam": score(strategy, opponent, steam_kingdoms(n_kingdoms, seed, must), seeds)}
        if train_kingdom and (must is None or must in train_kingdom):
            row["training"] = score(strategy, opponent, [train_kingdom], seeds * 10)
        if in_steam:
            need = list(keys) + ([must] if must else [])
            row["eligible"] = score(strategy, opponent, steam_kingdoms(n_kingdoms, seed, need), seeds)
        elif "training" in row:
            row["eligible"] = row["training"]
        out[name] = row
    def mean(part):
        rows = [out[n][part]["score"] for n in OPPONENTS if part in out[n]]
        return sum(rows) / len(rows) if rows else None
    out["steam_mean"] = mean("steam")
    out["eligible_mean"] = mean("eligible") if mean("eligible") is not None else out["steam_mean"]
    return out


def load(ref: str) -> tuple[dict, Strategy | None]:
    if ref == "big-money":
        return {"name": ref, "kingdom": None}, big_money_strategy()
    manifest = read_manifest(resolve(ref))
    if manifest["kind"] != "ga-strategy":
        return manifest, None
    return manifest, load_strategy(manifest["path"])


def store(manifest: dict, keys: dict[str, int] | None, r: dict, config: dict) -> str:
    path = os.path.join(manifest["dir"], BENCH_FILE)
    data = {"bench_version": BENCH_VERSION, "model": manifest["name"],
            "date": datetime.date.today().isoformat(), "config": config,
            "key_cards": keys, "score": r["eligible_mean"], "random_score": r["steam_mean"],
            "opponents": {n: r[n] for n in OPPONENTS}}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    return path


def up_to_date(manifest: dict, config: dict) -> bool:
    b = read_bench(manifest)
    return bool(b) and b.get("bench_version") == BENCH_VERSION and b.get("config") == config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--kingdoms", type=int, default=200, help="random Steam-like kingdoms per opponent")
    ap.add_argument("--seeds", type=int, default=20, help="games per kingdom (each played from both seats)")
    ap.add_argument("--seed", type=int, default=1, help="draws the kingdoms and game seeds")
    ap.add_argument("--json", metavar="FILE", help="also write all results to one JSON file")
    ap.add_argument("--store", action="store_true", help="write bench.json into each model's folder")
    ap.add_argument("--missing", action="store_true",
                    help="skip models whose bench.json is up to date (same settings and version)")
    args = ap.parse_args(argv)
    config = {"kingdoms": args.kingdoms, "seeds": args.seeds, "seed": args.seed}

    refs = args.models or [m["name"] for m in list_models()]
    refs = ["big-money"] + [r for r in refs if r != "big-money"]
    pct = lambda x: f"{100 * x:5.1f}%"
    names = list(OPPONENTS)
    print(f"{args.kingdoms} Steam-like kingdoms x {args.seeds} seeds x 2 seats per opponent; "
          f"score = win + tie/2; eligible = kingdoms with the model's key cards\n")
    print(f"{'model':<18}" + "".join(f"{n:>11}" for n in names) + f"{'eligible':>10}{'random':>9}"
          + "   key cards")
    results = {}
    for ref in refs:
        manifest, strategy = load(ref)
        name = manifest["name"]
        if strategy is None:
            print(f"{name:<18} skipped: a {manifest['kind']} model can't be played on the engine yet; "
                  f"it is ranked by the bench.json it ships, if any", flush=True)
            continue
        if args.missing and ref != "big-money" and up_to_date(manifest, config):
            b = read_bench(manifest)
            print(f"{name:<18}" + " " * 11 * len(names) + f"{pct(b['score']):>10}{pct(b['random_score']):>9}"
                  f"   {fmt(b.get('key_cards'))}   (stored)", flush=True)
            continue
        kingdom = manifest.get("kingdom")
        keys = None if ref == "big-money" else key_cards(manifest)
        if keys is None and kingdom:
            keys = derive(strategy, kingdom)
        r = bench(strategy, kingdom, args.kingdoms, args.seeds, args.seed, keys)
        results[name] = {"kingdom": kingdom, "key_cards": keys, **r}
        print(f"{name:<18}" + "".join(f"{pct(r[n].get('eligible', r[n]['steam'])['score']):>11}" for n in names)
              + f"{pct(r['eligible_mean']):>10}{pct(r['steam_mean']):>9}   "
              + ("-" if ref == "big-money" else fmt(keys)), flush=True)
        if args.store and ref != "big-money":
            store(manifest, keys, r, config)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
