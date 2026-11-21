"""Held-out strength of specialist genomes, for "does more training help?".

    python experiments/measure_specialist.py --must Witch --ref models/portfolio-v2/members/v2-witch.json \
        runs/x/snapshots/*.json [--json out.json]

For each genome: score (win + tie/2) against the three bots on fresh kingdoms holding the key cards
(the training objective, on games no run has seen: seed "test"), and head to head against the
reference genome and against v2-generalist on fresh kingdoms holding the key cards. Every genome
plays the same games, so differences between genomes are paired.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import load_strategy  # noqa: E402
from ga.c_bridge import evaluate_vs_opponent_c  # noqa: E402
from ga.specialist import complete, make_plan, mask  # noqa: E402
from policy.bench import OPPONENTS, steam_kingdoms  # noqa: E402
from policy.protocol import NO_REQUEST_YET  # noqa: E402

GEN = "models/portfolio-v2/members/v2-generalist.json"
CTX: dict = {}


def _measure(path: str) -> dict:
    s = mask(complete(load_strategy(path)), NO_REQUEST_YET)
    bots = {}
    for name, k, seeds in CTX["plan"]:
        r = evaluate_vs_opponent_c(s, seeds, k, OPPONENTS[name][0](k))
        bots.setdefault(name, []).append(r["win_rate"] + r["tie_rate"] / 2)
    bots = {n: sum(v) / len(v) for n, v in bots.items()}
    out = {"path": path, "bots": sum(bots.values()) / len(bots), "per_bot": bots}
    for label, opp in CTX["opps"].items():
        tot = 0.0
        for k, seeds in CTX["h2h"]:
            r = evaluate_vs_opponent_c(s, seeds, k, opp)
            tot += r["win_rate"] + r["tie_rate"] / 2
        out[label] = tot / len(CTX["h2h"])
    if CTX.get("net"):
        from experiments.specialist_vs_net import vs_net
        out["vs_net"] = sum(vs_net(s, seeds, k)["score"] for k, seeds in CTX["h2h"]) / len(CTX["h2h"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("genomes", nargs="+")
    ap.add_argument("--must", default="")
    ap.add_argument("--ref", required=True, help="reference genome (the v2 member)")
    ap.add_argument("--kingdoms", type=int, default=60, help="per bot, and for head to head x3")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--json")
    ap.add_argument("--net", action="store_true", help="also vs rl-v1's net (run with rl-v1's .venv python)")
    a = ap.parse_args()
    if a.net:
        from experiments.specialist_vs_net import _rl_mod
        _rl_mod()
        CTX["net"] = True
    must = [c.strip() for c in a.must.split(",") if c.strip()]
    rng = random.Random(f"test:{must}")
    CTX["plan"] = make_plan(must, a.kingdoms, a.seeds, rng)
    CTX["h2h"] = [(k, [rng.getrandbits(32) for _ in range(a.seeds)])
                  for k in steam_kingdoms(3 * a.kingdoms, rng.getrandbits(32), sorted(must))]
    CTX["opps"] = {"vs_ref": mask(complete(load_strategy(a.ref)), NO_REQUEST_YET),
                   "vs_gen": mask(complete(load_strategy(GEN)), NO_REQUEST_YET)}
    paths = [a.ref] + a.genomes
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_measure, paths, chunksize=1)
    print(f"{'genome':60s} {'bots':>6s} {'vs_ref':>7s} {'vs_gen':>7s}" + (f" {'vs_net':>7s}" if a.net else ""))
    for r in res:
        print(f"{r['path'][-60:]:60s} {100 * r['bots']:6.1f} {100 * r['vs_ref']:7.1f} {100 * r['vs_gen']:7.1f}"
              + (f" {100 * r['vs_net']:7.1f}" if a.net else ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
