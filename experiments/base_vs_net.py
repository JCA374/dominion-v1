"""The GA base with a new pick rule (and swapped members) against rl-v1's network, next to the old base.

    ../rl-v1/.venv/bin/python experiments/base_vs_net.py runs/meas/rank_v2.json [--rule table|old]
        [--kingdoms 400] [--seeds 10] [--swap member=strategy.json ...]

The old base is az-v1/az/base.py's pick of the released portfolio-v2 members (rl-v1's opponent,
~26-30% against the net). On every fresh kingdom where the new base plays a different genome,
both bases play the net on the same seeds; elsewhere the two are identical games.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import specialist_vs_net as svn  # noqa: E402

CTX: dict = {}


def vs(ints, kingdom, seeds) -> float:
    r = svn._rl_mod()
    net = r["Seat"](r["NET"], r["base_genome"](kingdom)["ints"], r["params"])
    pts = 0.0
    for seed in seeds:
        for seat in (0, 1):
            seats = [None, None]
            seats[seat] = r["Seat"](r["GA"], ints)
            seats[1 - seat] = net
            pts += (r["outcome"](r["play"](seats, kingdom, seed, rng_seed=seed ^ 0x5EED), seat) + 1) / 2
    return pts / (2 * len(seeds))


def round_robin(el, kingdom):
    """Every eligible member plays every other on this kingdom (own seeds); best mean score wins."""
    from ga.c_bridge import evaluate_vs_opponent_c
    rng = random.Random("rr:" + ",".join(kingdom))
    seeds = [rng.getrandbits(32) for _ in range(CTX["rr_seeds"])]
    st = CTX["strat"]
    tot = {n: 0.0 for n in el}
    for i, x in enumerate(el):
        for y in el[i + 1:]:
            r = evaluate_vs_opponent_c(st[x], seeds, kingdom, st[y])
            s = r["win_rate"] + r["tie_rate"] / 2
            tot[x] += s
            tot[y] += 1 - s
    return max(el, key=lambda n: tot[n])


def _one(i):
    rng = random.Random(f"net-test:{i}")
    from policy.bench import STEAM_BASE_SET
    k = sorted(rng.sample(STEAM_BASE_SET, 10))
    seeds = [rng.getrandbits(32) for _ in range(CTX["seeds"])]
    old = svn._rl_mod()["base_genome"](k)["name"]
    ks = set(k)
    el = [n for n in CTX["ints"] if all(c in ks for c in CTX["must"][n])]
    P = CTX["P"]
    if CTX["rule"] == "table":
        new = max(el, key=lambda a: sum(P[a][b] for b in el if b != a) / max(1, len(el) - 1))
    elif CTX["rule"] == "rr":
        new = round_robin(el, k)
    elif CTX["rule"] == "trial":          # portfolio-v2's own picker (trial games vs bots + hall)
        from policy.portfolio import choose
        new = choose(CTX["man"], k, strategies=CTX["strat"])["member"]
    else:
        new = old
    if new == old and CTX["ints"][new] is CTX["ints0"][old]:
        return {"kingdom": k, "old": old, "new": new}
    return {"kingdom": k, "old": old, "new": new,
            "s_old": vs(CTX["ints0"][old], k, seeds), "s_new": vs(CTX["ints"][new], k, seeds)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rank")
    ap.add_argument("--rule", default="table", choices=["table", "rr", "trial", "old"])
    ap.add_argument("--rr-seeds", type=int, default=10)
    ap.add_argument("--kingdoms", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--swap", nargs="*", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    svn._rl_mod()
    from az.core import portfolio_members
    from core.strategy import load_strategy
    from ga.c_bridge import strategy_to_ints
    ms = portfolio_members()
    ints0 = {m["name"]: m["ints"] for m in ms}
    from policy.portfolio import members
    from policy.registry import read_manifest, resolve
    man = read_manifest(resolve("portfolio-v2"))
    strat = members(man)
    ints = dict(ints0)
    for sw in a.swap:
        n, p = sw.split("=", 1)
        strat[n] = load_strategy(p)
        ints[n] = strategy_to_ints(strat[n])
    with open(a.rank) as f:
        P = json.load(f)["P"]
    CTX.update(man=man, strat=strat, rr_seeds=a.rr_seeds, ints0=ints0, ints=ints, P=P, rule=a.rule, seeds=a.seeds, must={m["name"]: m["must"] for m in ms})
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_one, range(a.kingdoms), chunksize=1)
    played = [r for r in rows if "s_old" in r]
    if played:
        so = sum(r["s_old"] for r in played) / len(played)
        sn = sum(r["s_new"] for r in played) / len(played)
        d = [r["s_new"] - r["s_old"] for r in played]
        md = sum(d) / len(d)
        sd = (sum((x - md) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
        print(f"{len(played)} of {len(rows)} kingdoms play differently: vs net old {100 * so:.1f}%,"
              f" new {100 * sn:.1f}%, diff {100 * md:+.1f} (±{196 * sd / len(d) ** 0.5:.1f})")
        print(f"over all kingdoms: {100 * md * len(played) / len(rows):+.2f} points")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
