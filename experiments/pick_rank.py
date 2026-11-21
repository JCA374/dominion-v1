"""A head-to-head ranking for the GA base's pick rule, tested against the bots ranking.

    python experiments/pick_rank.py [--pair-kingdoms 30] [--seeds 10] [--out rank.json] [--swap member=strategy.json ...]

1. Round robin: every pair of portfolio-v2 members plays on --pair-kingdoms random kingdoms holding
   both members' must cards (seed "rank-train").
2. Bradley-Terry ratings from those scores give a new order.
3. Test on fresh kingdoms (seed "rank-test"): the base with the new order plays the base with the
   old order (az-v1/data/members.json), paired seats; only kingdoms where the picks differ count
   as non-mirror games.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.strategy import load_strategy  # noqa: E402
from ga.c_bridge import evaluate_vs_opponent_c  # noqa: E402
from policy.bench import STEAM_BASE_SET  # noqa: E402
from policy.portfolio import members  # noqa: E402
from policy.registry import read_manifest, resolve  # noqa: E402

RANK = os.path.join(os.path.dirname(HERE), "az-v1", "data", "members.json")
CTX: dict = {}


def kingdom_with(rng, need):
    pool = [c for c in STEAM_BASE_SET if c not in need]
    return sorted(rng.sample(pool, 10 - len(need)) + list(need))


def _pair(job) -> tuple:
    a, b, i = job
    rng = random.Random(f"rank-train:{a}:{b}:{i}")
    k = kingdom_with(rng, sorted(set(CTX["must"][a]) | set(CTX["must"][b])))
    r = evaluate_vs_opponent_c(CTX["s"][a], [rng.getrandbits(32) for _ in range(CTX["seeds"])], k, CTX["s"][b])
    return a, b, r["win_rate"] + r["tie_rate"] / 2


def pick(rule, kingdom):
    ks = set(kingdom)
    if rule == "table":                   # best mean score vs the other eligible members
        el = [n for n in CTX["s"] if all(c in ks for c in CTX["must"][n])]
        P = CTX["P"]
        return max(el, key=lambda a: sum(P[a][b] for b in el if b != a) / max(1, len(el) - 1))
    return next(n for n in CTX[rule] if all(c in ks for c in CTX["must"][n]))


def _test(i) -> dict:
    rng = random.Random(f"rank-test:{i}")
    k = sorted(rng.sample(STEAM_BASE_SET, 10))
    seeds = [rng.getrandbits(32) for _ in range(CTX["seeds"])]
    new, old = pick(CTX["rule"], k), pick("old", k)
    sc = 0.5 if new == old else None
    if sc is None:
        r = evaluate_vs_opponent_c(CTX["s"][new], seeds, k, CTX["s"][old])
        sc = r["win_rate"] + r["tie_rate"] / 2
    return {"kingdom": k, "new": new, "old": old, "score": sc}


def bradley_terry(names, results, iters=500):
    """results: [(a, b, score of a)]; each counts as one 'game' worth its score. MM updates."""
    w = {n: 0.0 for n in names}
    games = {n: [] for n in names}
    for a, b, s in results:
        w[a] += s
        w[b] += 1 - s
        games[a].append(b)
        games[b].append(a)
    p = {n: 1.0 for n in names}
    for _ in range(iters):
        q = {n: (w[n] + 0.5) / (sum(1 / (p[n] + p[o]) for o in games[n]) + 1 / p[n]) for n in names}
        g = math.exp(sum(math.log(v) for v in q.values()) / len(q))
        p = {n: v / g for n, v in q.items()}
    return {n: math.log(v) for n, v in p.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-kingdoms", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--test-kingdoms", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--swap", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    man = read_manifest(resolve("portfolio-v2"))
    s = members(man)
    for sw in a.swap:
        n, p = sw.split("=", 1)
        s[n] = load_strategy(p)
    with open(RANK) as f:
        old = json.load(f)
    CTX.update(s=s, must={m["name"]: m["must"] for m in man["members"]}, seeds=a.seeds,
               old=sorted(s, key=lambda n: -old[n]))
    names = sorted(s)
    jobs = [(x, y, i) for x, y in itertools.combinations(names, 2) for i in range(a.pair_kingdoms)
            if len(set(CTX["must"][x]) | set(CTX["must"][y])) <= 10]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        res = pool.map(_pair, jobs, chunksize=16)
    rating = bradley_terry(names, res)
    CTX["new"] = sorted(names, key=lambda n: -rating[n])
    acc: dict = {}
    for x, y, sc in res:
        acc.setdefault(x, {}).setdefault(y, []).append(sc)
        acc.setdefault(y, {}).setdefault(x, []).append(1 - sc)
    CTX["P"] = {x: {y: sum(v) / len(v) for y, v in d.items()} for x, d in acc.items()}
    print("new order (BT rating, old bots score):")
    for n in CTX["new"]:
        print(f"  {n:20s} {rating[n]:+.3f}   {100 * old[n]:.1f}%")
    out = {"rating": rating, "order": CTX["new"], "P": CTX["P"], "swaps": a.swap}
    for rule in ("new", "table"):
        CTX["rule"] = rule
        with ctx.Pool(a.workers) as pool:
            rows = pool.map(_test, range(a.test_kingdoms), chunksize=8)
        tot = sum(r["score"] for r in rows) / len(rows)
        diff = [r for r in rows if r["new"] != r["old"]]
        n = 2 * a.seeds
        md = sum(r["score"] for r in diff) / max(1, len(diff))
        se = (md * (1 - md) / max(1, len(diff) * n)) ** 0.5
        print(f"rule {rule}: {len(rows)} fresh kingdoms, vs old base {100 * tot:.1f}%;"
              f" picks differ on {len(diff)}: {100 * md:.1f}% (±{196 * se:.1f})")
        out[rule] = {"test": tot, "test_differ": md, "n_differ": len(diff)}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
