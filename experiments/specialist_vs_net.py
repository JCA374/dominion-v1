"""ga/specialist.py with rl-v1's network as an extra training opponent ("train against the best model").

    ../rl-v1/.venv/bin/python experiments/specialist_vs_net.py NAME --must Witch --net-weight 2 [specialist args]

Needs rl-v1 (../rl-v1, its .venv for numpy and rl.so). The net plays every decision of its seat
(argmax), the genome plays through the same ga_agent.c. Opponent "hall:net" is played on kingdoms
holding the key cards and counts --net-weight times in the mean score (a bot counts once).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL = os.path.join(os.path.dirname(HERE), "rl-v1")
sys.path.insert(0, HERE)

import ga.specialist as sp  # noqa: E402
from ga.c_bridge import evaluate_vs_opponent_c, strategy_to_ints  # noqa: E402
from policy.bench import OPPONENTS  # noqa: E402

NET_PATH = os.path.join(RL, "models", "rl_r2_200.npy")
NET_WEIGHT = 1.0
_rl = {}


class NetOpponent:
    pass


def _rl_mod():
    if not _rl:
        sys.path.append(RL)
        import numpy as np
        from rl.core import GA, NET, Seat, base_genome, outcome, play
        _rl.update(np=np, GA=GA, NET=NET, Seat=Seat, base_genome=base_genome, outcome=outcome, play=play,
                   params=np.load(NET_PATH).astype(np.float32))
    return _rl


def vs_net(strategy, seeds, kingdom) -> dict:
    r = _rl_mod()
    me = strategy_to_ints(strategy)
    g = r["base_genome"](kingdom)["ints"]
    pts, margin, n = 0.0, 0.0, 0
    for seed in seeds:
        for seat in (0, 1):
            seats = [None, None]
            seats[seat] = r["Seat"](r["GA"], me)
            seats[1 - seat] = r["Seat"](r["NET"], g, r["params"])
            res = r["play"](seats, kingdom, seed, rng_seed=seed ^ 0x5EED)
            pts += (r["outcome"](res, seat) + 1) / 2
            margin += res["scores"][seat] - res["scores"][1 - seat]
            n += 1
    return {"score": pts / n, "mean_vp_margin": margin / n}


def score(strategy, plan) -> dict:
    per, margin = {}, 0.0
    for name, k, seeds in plan:
        opp = sp.HALL[name[5:]] if name.startswith("hall:") else OPPONENTS[name][0](k)
        if isinstance(opp, NetOpponent):
            r = vs_net(strategy, seeds, k)
            s = r["score"]
        else:
            r = evaluate_vs_opponent_c(strategy, seeds, k, opp)
            s = r["win_rate"] + r["tie_rate"] / 2
        per.setdefault(name, []).append(s)
        margin += r["mean_vp_margin"]
    bots = {n: sum(v) / len(v) for n, v in per.items()}
    w = {n: (NET_WEIGHT if n == "hall:net" else 1.0) for n in bots}
    mean = sum(bots[n] * w[n] for n in bots) / sum(w.values())
    margin /= len(plan)
    fit = (1 - sp.MARGIN_WEIGHT) * mean + sp.MARGIN_WEIGHT * max(0.0, min(1.0, (margin + 20) / 40))
    only = [v for n, v in bots.items() if n in OPPONENTS]
    return {"fitness": fit, "score": mean, "bots": bots, "bot_score": sum(only) / len(only), "margin": margin}


def main() -> int:
    global NET_WEIGHT
    argv = list(sys.argv[1:])
    if "--net-weight" in argv:
        i = argv.index("--net-weight")
        NET_WEIGHT = float(argv[i + 1])
        del argv[i:i + 2]
    sp.score = score
    real_run = sp.run

    def run(*a, hall=None, **kw):
        hall = dict(hall or {})
        hall["net"] = NetOpponent()
        return real_run(*a, hall=hall, **kw)
    sp.run = run
    _rl_mod()                     # load rl.so + net before the pool forks
    return sp.main(argv)


if __name__ == "__main__":
    sys.exit(main())
