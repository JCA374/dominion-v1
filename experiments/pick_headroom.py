"""How much does the GA base lose by its fixed pick rule? (az-v1/az/base.py)

    python experiments/pick_headroom.py [--kingdoms 200] [--seeds 10] [--json f] [--swap member=strategy.json ...]

The base plays the best-ranked member (score vs the bots, az-v1/data/members.json) whose must
cards are all dealt. Here every other eligible member plays head to head against that pick on
random kingdoms. Rows are saved so a better rule can be fitted on them and tested on fresh ones.
"""
from __future__ import annotations

import argparse
import json
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


def eligible(kingdom, must):
    ks = set(kingdom)
    return [n for n in CTX["order"] if all(c in ks for c in must[n])]


def _one(i: int) -> dict:
    k, seeds = CTX["kingdoms"][i], CTX["seeds"][i]
    el = eligible(k, CTX["must"])
    pick = el[0]
    out = {}
    for n in el[1:]:
        r = evaluate_vs_opponent_c(CTX["s"][n], seeds, k, CTX["s"][pick])
        out[n] = r["win_rate"] + r["tie_rate"] / 2
    return {"kingdom": k, "pick": pick, "eligible": el, "vs_pick": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kingdoms", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", default="pick-train")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--swap", nargs="*", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    man = read_manifest(resolve("portfolio-v2"))
    s = members(man)
    for sw in a.swap:
        n, p = sw.split("=", 1)
        s[n] = load_strategy(p)
    with open(RANK) as f:
        rank = json.load(f)
    rng = random.Random(a.seed)
    CTX.update(s=s, must={m["name"]: m["must"] for m in man["members"]},
               order=sorted(s, key=lambda n: -rank[n]),
               kingdoms=[sorted(rng.sample(STEAM_BASE_SET, 10)) for _ in range(a.kingdoms)],
               seeds=[[rng.getrandbits(32) for _ in range(a.seeds)] for _ in range(a.kingdoms)])
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_one, range(a.kingdoms), chunksize=1)
    best = [max(r["vs_pick"].values(), default=0.5) for r in rows]
    print(f"{len(rows)} kingdoms, {sum(len(r['vs_pick']) for r in rows) / len(rows):.1f} rivals each")
    print(f"best rival vs pick (selected on these games, biased up): {100 * sum(best) / len(best):.1f}%")
    print(f"kingdoms where a rival scores >= 55%: {sum(b >= .55 for b in best)}")
    tally: dict = {}
    for r in rows:
        for n, v in r["vs_pick"].items():
            tally.setdefault((n, r["pick"]), []).append(v)
    print("pairs (rival vs pick), >= 8 kingdoms, mean >= 52%:")
    for (n, p), v in sorted(tally.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        m = sum(v) / len(v)
        if len(v) >= 8 and m >= .52:
            print(f"  {n:20s} vs {p:20s} {100 * m:5.1f}%  ({len(v)} kingdoms)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
