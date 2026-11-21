"""The pairwise-table pick rule (experiments/pick_rank.py --out) against portfolio-v2's trial-game picker.

    python experiments/pick_vs_trial.py runs/meas/rank_v2.json [--kingdoms 300] [--swap member=strategy.json ...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import load_strategy  # noqa: E402
from ga.c_bridge import evaluate_vs_opponent_c  # noqa: E402
from policy.bench import STEAM_BASE_SET  # noqa: E402
from policy.portfolio import choose, members  # noqa: E402
from policy.registry import read_manifest, resolve  # noqa: E402

CTX: dict = {}


def table_pick(kingdom):
    ks, P = set(kingdom), CTX["P"]
    el = [n for n in CTX["s"] if all(c in ks for c in CTX["must"][n])]
    return max(el, key=lambda a: sum(P[a][b] for b in el if b != a) / max(1, len(el) - 1))


def rr_pick(kingdom, trial=None):
    """Round robin among the eligible members on this kingdom (own seeds); best mean score. With `trial` (portfolio
    choose()'s scores vs the bots and its hall), the mean of the two."""
    ks = set(kingdom)
    el = [n for n in CTX["s"] if all(c in ks for c in CTX["must"][n])]
    rng = random.Random("rr:" + ",".join(kingdom))
    seeds = [rng.getrandbits(32) for _ in range(CTX["rr_seeds"])]
    tot = {n: 0.0 for n in el}
    for i, x in enumerate(el):
        for y in el[i + 1:]:
            r = evaluate_vs_opponent_c(CTX["s"][x], seeds, kingdom, CTX["s"][y])
            sc = r["win_rate"] + r["tie_rate"] / 2
            tot[x] += sc
            tot[y] += 1 - sc
    rr = {n: tot[n] / max(1, len(el) - 1) for n in el}
    if trial is not None:
        return max(el, key=lambda n: (rr[n] + trial.get(n, 0)) / 2)
    return max(el, key=lambda n: rr[n])


def _one(i):
    rng = random.Random(f"trial-test:{i}")
    k = sorted(rng.sample(STEAM_BASE_SET, 10))
    seeds = [rng.getrandbits(32) for _ in range(CTX["seeds"])]
    ch = choose(CTX["man"], k, strategies=CTX["s0"])
    b = ch["member"]
    a = rr_pick(k) if CTX["rule"] == "rr" else rr_pick(k, ch["scores"]) if CTX["rule"] == "hybrid" else table_pick(k)
    sc = 0.5
    if a != b:
        r = evaluate_vs_opponent_c(CTX["s"][a], seeds, k, CTX["s0"][b])
        sc = r["win_rate"] + r["tie_rate"] / 2
    return {"kingdom": k, "table": a, "trial": b, "score": sc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rank")
    ap.add_argument("--kingdoms", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--swap", nargs="*", default=[])
    ap.add_argument("--rule", default="table", choices=["table", "rr", "hybrid"])
    ap.add_argument("--rr-seeds", type=int, default=100)
    a = ap.parse_args()
    man = read_manifest(resolve("portfolio-v2"))
    s0 = members(man)
    s = dict(s0)
    for sw in a.swap:
        n, p = sw.split("=", 1)
        s[n] = load_strategy(p)
    with open(a.rank) as f:
        P = json.load(f)["P"]
    CTX.update(rule=a.rule, rr_seeds=a.rr_seeds, man=man, s0=s0, s=s, P=P, seeds=a.seeds, must={m["name"]: m["must"] for m in man["members"]})
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_one, range(a.kingdoms), chunksize=1)
    diff = [r for r in rows if r["table"] != r["trial"]]
    tot = sum(r["score"] for r in rows) / len(rows)
    md = sum(r["score"] for r in diff) / max(1, len(diff))
    se = (md * (1 - md) / max(1, len(diff) * 2 * a.seeds)) ** 0.5
    print(f"{a.rule} rule vs portfolio-v2 trial picker, {len(rows)} kingdoms: {100 * tot:.1f}%;"
          f" picks differ on {len(diff)}: {100 * md:.1f}% (±{196 * se:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
