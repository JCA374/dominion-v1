"""Portfolio-v2 with some members replaced, head to head against portfolio-v2 itself.

    python experiments/swap_versus.py v2-witch=runs/c1-witch/strategy.json ... [--kingdoms 300] [--json f]

Each side picks its member per kingdom exactly as in a real game (policy.portfolio.choose), then
the two play --seeds games from each seat. Prints the overall score of the swapped portfolio and
the score on the kingdoms where the two picks differ (elsewhere it is a mirror match).
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import load_strategy  # noqa: E402
from ga.c_bridge import evaluate_vs_opponent_c  # noqa: E402
from policy.bench import STEAM_BASE_SET, kingdom_seeds  # noqa: E402
from policy.portfolio import choose, members  # noqa: E402
from policy.registry import read_manifest, resolve  # noqa: E402

CTX: dict = {}


def _one(i: int) -> dict:
    a, b, sa, sb, base, kingdoms = CTX["a"], CTX["b"], CTX["sa"], CTX["sb"], CTX["base"], CTX["kingdoms"]
    k = kingdoms[i]
    pa = choose(a, k, strategies=sa)["member"]
    pb = choose(b, k, strategies=sb)["member"]
    r = evaluate_vs_opponent_c(sa[pa], kingdom_seeds(base, i), k, sb[pb])
    return {"kingdom": k, "a": pa, "b": pb, "score": r["win_rate"] + r["tie_rate"] / 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("swaps", nargs="+", help="member=strategy.json")
    ap.add_argument("--base", default="portfolio-v2")
    ap.add_argument("--kingdoms", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()
    base = read_manifest(resolve(a.base))
    sb = members(base)
    sa = dict(sb)
    for sw in a.swaps:
        name, path = sw.split("=", 1)
        assert name in sa, name
        sa[name] = load_strategy(path)
    rng = random.Random(a.seed)
    CTX.update(a=copy.deepcopy(base), b=base, sa=sa, sb=sb,
               base=[rng.getrandbits(32) for _ in range(a.seeds)],
               kingdoms=[sorted(rng.sample(STEAM_BASE_SET, 10)) for _ in range(a.kingdoms)])
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_one, range(a.kingdoms), chunksize=1)
    tot = sum(r["score"] for r in rows) / len(rows)
    diff = [r for r in rows if r["a"] != r["b"]]
    swapped = {sw.split("=")[0] for sw in a.swaps}
    played = [r for r in rows if r["a"] in swapped]
    n = a.seeds * 2
    se = lambda m, k: (m * (1 - m) / (k * n)) ** 0.5 if k else 0
    print(f"all {len(rows)} kingdoms: {100 * tot:.1f}% (±{196 * se(tot, len(rows)):.1f})")
    for label, rs in (("picks differ", diff), ("swapped member plays", played)):
        if rs:
            m = sum(r["score"] for r in rs) / len(rs)
            print(f"{label} ({len(rs)} kingdoms): {100 * m:.1f}% (±{196 * se(m, len(rs)):.1f})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"score": tot, "swaps": a.swaps, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
