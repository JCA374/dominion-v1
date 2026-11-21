"""Specialist trainer for the portfolio model: one genome that wins on *any* kingdom holding its key cards.

    python -m ga.specialist witch --must Witch --gens 600
    python -m ga.specialist generalist --gens 600            # no key card: any kingdom
    python -m ga.specialist witch --must Witch --exclude-blocked   # Steam-safe: no NO_REQUEST_YET cards

The classic trainer (ga/main.py) evolves a genome for one fixed kingdom. A Steam game deals
10 of 26 cards at random, so a specialist here sees fresh kingdoms every generation: 10
cards that always contain its key cards (--must), the rest drawn from all 26 base-set
cards. Every genome knows every card (its lists hold all of them; cards after PASS are not
bought until mutation moves them up).

Fitness = the benchmark's question (policy/bench.py): score (win + tie/2, Steam tiebreak)
against Big Money, Smithy-BM and Witch-BM, each on kingdoms that also hold the bot's card,
plus a small VP-margin term. All genomes of a generation play the same kingdoms and seeds.
Every --val-every generations the top candidates are scored on a fixed validation set
(--val-kingdoms per bot), and the best validated genome is the run's output:

    runs/<name>/strategy.json   best validated genome      runs/<name>/log.csv
    runs/<name>/progress.json   run state for the site's training log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict
import multiprocessing as mp

from core.cards import ACTION_CARDS, ALL_CARDS, BUYABLE_CARDS
from core.strategy import (Strategy, big_money_strategy, engine_strategy, gardens_strategy,
                           load_strategy, random_strategy, smithy_big_money_strategy,
                           witch_big_money_strategy, _single_card_big_money)
from ga.c_bridge import evaluate_vs_opponent_c
from ga.ga import crossover, mutate, tournament_select
from policy.bench import OPPONENTS, steam_kingdoms
from policy.protocol import NO_REQUEST_YET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
MARGIN_WEIGHT = 0.1
HALL: dict[str, Strategy] = {}   # extra evolved opponents (--hall), set before the worker pool forks


# -- genomes over all 26 cards ---------------------------------------------------------

def _pad(lst: list[str], universe: list[str], banned: set[str], marker: str | None) -> list[str]:
    """Keep the order, drop banned cards, append the missing ones (after the marker, if any)."""
    out = [c for c in lst if c not in banned]
    missing = [c for c in universe if c not in out and c not in banned]
    if marker is not None and marker not in out:
        out.append(marker)
    return out + missing


def complete(s: Strategy, banned: set[str] = frozenset()) -> Strategy:
    """The same genome, with every card it may ever meet present in its lists (new cards
    below PASS, so the behaviour is unchanged), and banned cards removed."""
    s = deepcopy(s)
    nt = [c for c in ACTION_CARDS if ALL_CARDS[c].actions > 0]
    t = [c for c in ACTION_CARDS if ALL_CARDS[c].actions == 0]
    for f in ("early_buy_priority", "mid_buy_priority", "late_buy_priority"):
        setattr(s, f, _pad(getattr(s, f), BUYABLE_CARDS, banned, "PASS"))
    s.nonterminal_priority = _pad(s.nonterminal_priority, nt, banned, None)
    s.terminal_priority = _pad(s.terminal_priority, t, banned, None)
    s.throne_room_priority = _pad(s.throne_room_priority,
                                  [c for c in ACTION_CARDS if c != "Throne Room"], banned, None)
    s.buy_targets = {c: n for c, n in s.buy_targets.items() if c not in banned}
    return s


def mask(s: Strategy, banned) -> Strategy:
    """What a player that can't play `banned` cards actually plays: never buys them."""
    banned = set(banned)
    s = deepcopy(s)
    for f in ("early_buy_priority", "mid_buy_priority", "late_buy_priority"):
        setattr(s, f, [c for c in getattr(s, f) if c not in banned])
    return s


def seeds_for(must: list[str], banned: set[str]) -> list[Strategy]:
    """Hand-made starting points: the bots, the old archetypes, Big Money with each key card."""
    out = [big_money_strategy(), smithy_big_money_strategy(), witch_big_money_strategy(),
           engine_strategy(), gardens_strategy()]
    for c in must:
        if ALL_CARDS[c].card_type.name == "ACTION":
            out.append(_single_card_big_money(c, 2, None))
    k4 = os.path.join(ROOT, "models", "k4-20260614", "strategy.json")
    if os.path.exists(k4):
        out.append(load_strategy(k4))
    return [complete(s, banned) for s in out]


# -- fitness ----------------------------------------------------------------------------

def make_plan(must: list[str], n_kingdoms: int, n_seeds: int, rng: random.Random,
              hall=()) -> list[tuple]:
    """[(opponent, kingdom, seeds)]: n_kingdoms per bot, each holding must + the bot's card,
    each with its own seeds (shared seeds replay the same games where the cards don't matter);
    then n_kingdoms per hall member ("hall:<name>"), holding must."""
    plan = []
    opps = [(name, card) for name, (_, card) in OPPONENTS.items()] + [(f"hall:{h}", None) for h in hall]
    for name, card in opps:
        need = sorted(set(must) | ({card} if card else set()))
        for k in steam_kingdoms(n_kingdoms, rng.getrandbits(32), need):
            plan.append((name, k, [rng.getrandbits(32) for _ in range(n_seeds)]))
    return plan


def score(strategy: Strategy, plan: list[tuple]) -> dict:
    """Mean score over all opponents, per-opponent scores, the bots-only mean and the VP margin."""
    per, margin = {}, 0.0
    for name, k, seeds in plan:
        opp = HALL[name[5:]] if name.startswith("hall:") else OPPONENTS[name][0](k)
        r = evaluate_vs_opponent_c(strategy, seeds, k, opp)
        per.setdefault(name, []).append(r["win_rate"] + r["tie_rate"] / 2)
        margin += r["mean_vp_margin"]
    bots = {n: sum(v) / len(v) for n, v in per.items()}
    mean = sum(bots.values()) / len(bots)
    margin /= len(plan)
    fit = (1 - MARGIN_WEIGHT) * mean + MARGIN_WEIGHT * max(0.0, min(1.0, (margin + 20) / 40))
    only = [v for n, v in bots.items() if n in OPPONENTS]
    return {"fitness": fit, "score": mean, "bots": bots, "bot_score": sum(only) / len(only),
            "margin": margin}


def _score(args):
    return score(*args)


# -- the run ----------------------------------------------------------------------------

def run(name: str, must: list[str], gens: int, pop_size: int, n_kingdoms: int, n_seeds: int,
        workers: int, seed: int, banned: set[str], val_every: int, val_kingdoms: int,
        tier2: int = 8, elites: int = 3, rate: float = 0.18, resume: bool = False,
        hall: dict[str, Strategy] | None = None, snapshots: bool = False) -> dict:
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    HALL.clear()
    HALL.update(hall or {})
    val_plan = make_plan(must, val_kingdoms, 10, random.Random(f"val:{seed}:{must}"), HALL)

    pop = seeds_for(must, banned)
    if resume and os.path.exists(os.path.join(out, "strategy.json")):
        pop.append(complete(load_strategy(os.path.join(out, "strategy.json")), banned))
    while len(pop) < pop_size:
        pop.append(complete(random_strategy(rng), banned))
    pop = pop[:pop_size]

    best, best_val, history = None, -1.0, []
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["gen", "best_fitness", "mean_fitness", "tier2_score", "val_score", "seconds"])
    stale, t0 = 0, time.time()
    progress = {"name": name, "must": must, "banned": sorted(banned), "gens": gens,
                "pop": pop_size, "kingdoms_per_bot": n_kingdoms, "seeds": n_seeds,
                "val_kingdoms_per_bot": val_kingdoms, "hall": sorted(hall or {}),
                "started": time.strftime("%Y-%m-%d %H:%M")}

    with mp.get_context("fork").Pool(workers) as pool:
        for gen in range(1, gens + 1):
            plan = make_plan(must, n_kingdoms, n_seeds, rng, HALL)
            res = pool.map(_score, [(s, plan) for s in pop], chunksize=1)
            fit = [r["fitness"] for r in res]
            order = sorted(range(len(pop)), key=lambda i: -fit[i])[:tier2]
            plan2 = make_plan(must, n_kingdoms * 2, n_seeds * 2, rng, HALL)
            res2 = pool.map(_score, [(pop[i], plan2) for i in order], chunksize=1)
            ranked = [order[j] for j in sorted(range(len(order)), key=lambda j: -res2[j]["fitness"])]
            top2 = max(res2, key=lambda r: r["fitness"])

            val = ""
            if gen % val_every == 0 or gen == gens:
                cands = [pop[i] for i in ranked[:3]]
                vres = pool.map(_score, [(s, val_plan) for s in cands], chunksize=1)
                j = max(range(len(cands)), key=lambda j: vres[j]["score"])
                val = vres[j]["score"]
                if val > best_val:
                    best_val, best, stale = val, deepcopy(cands[j]), 0
                    with open(os.path.join(out, "strategy.json"), "w") as f:
                        json.dump(asdict(best), f, indent=1)
                    with open(os.path.join(out, "val.json"), "w") as f:
                        json.dump({"gen": gen, **vres[j]}, f, indent=1)
                else:
                    stale += 1
                if snapshots:
                    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
                    with open(os.path.join(out, "snapshots", f"g{gen:05d}.json"), "w") as f:
                        json.dump(asdict(best), f, indent=1)
                history.append({"gen": gen, "val": round(val, 4), "best": round(best_val, 4),
                                "bot_score": round(vres[j]["bot_score"], 4),
                                "bots": {k: round(v, 4) for k, v in vres[j]["bots"].items()}})
                print(f"[{name}] gen {gen:4d}  val {100 * val:5.1f}%  best {100 * best_val:5.1f}%  "
                      f"{' '.join(f'{k} {100 * v:4.1f}%' for k, v in vres[j]['bots'].items())}  "
                      f"{time.time() - t0:6.0f}s", flush=True)
                progress.update({"gen": gen, "best_val": round(best_val, 4), "history": history,
                                 "elapsed_s": round(time.time() - t0),
                                 "updated": time.strftime("%Y-%m-%d %H:%M")})
                with open(os.path.join(out, "progress.json"), "w") as f:
                    json.dump(progress, f, indent=1)

            w.writerow([gen, f"{max(fit):.4f}", f"{sum(fit) / len(fit):.4f}",
                        f"{top2['score']:.4f}", f"{val:.4f}" if val != "" else "",
                        f"{time.time() - t0:.0f}"])
            log.flush()
            if gen == gens:
                break

            boost = stale >= 4                     # 4 validations without progress
            r = min(rate * 1.5, 0.35) if boost else rate
            nxt = [deepcopy(pop[i]) for i in ranked[:elites]]
            if best is not None:
                nxt.append(deepcopy(best))
            if boost:
                nxt += [complete(random_strategy(rng), banned) for _ in range(pop_size // 10)]
                stale = 0
            while len(nxt) < pop_size:
                a = tournament_select(pop, fit, 3, rng)
                b = tournament_select(pop, fit, 3, rng)
                child = complete(mutate(crossover(a, b, rng), r, rng), banned)
                nxt.append(child)
            pop = nxt

    log.close()
    progress.update({"done": True, "finished": time.strftime("%Y-%m-%d %H:%M")})
    with open(os.path.join(out, "progress.json"), "w") as f:
        json.dump(progress, f, indent=1)
    return {"best": best, "val": best_val}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="run name: runs/<name>/")
    ap.add_argument("--must", default="", help="key cards, comma-separated (empty: any kingdom)")
    ap.add_argument("--gens", type=int, default=600)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--kingdoms", type=int, default=4, help="kingdoms per bot per generation")
    ap.add_argument("--seeds", type=int, default=6, help="seeds per kingdom (x2 seats)")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-kingdoms", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude-blocked", action="store_true",
                    help="never buy the cards the protocol can't play yet (policy/protocol.py)")
    ap.add_argument("--resume", action="store_true", help="seed with the run's current strategy.json")
    ap.add_argument("--rate", type=float, default=0.18, help="mutation rate")
    ap.add_argument("--snapshots", action="store_true",
                    help="keep the best validated genome of every validation in runs/<name>/snapshots/")
    ap.add_argument("--hall", default="",
                    help="extra opponents: runs/<name> folders or strategy.json files, comma-separated "
                         "(played on kingdoms with the key cards; they count like a bot)")
    a = ap.parse_args(argv)
    must = [c.strip() for c in a.must.split(",") if c.strip()]
    for c in must:
        if c not in ALL_CARDS:
            ap.error(f"unknown card {c!r}")
    banned = set(NO_REQUEST_YET) if a.exclude_blocked else set()
    hall = {}
    for h in (x.strip() for x in a.hall.split(",") if x.strip()):
        path = h if h.endswith(".json") else os.path.join(h, "strategy.json")
        hall[os.path.basename(os.path.normpath(h if not h.endswith(".json") else os.path.dirname(h)))] = \
            load_strategy(path)
    run(a.name, must, a.gens, a.pop, a.kingdoms, a.seeds, a.workers, a.seed, banned,
        a.val_every, a.val_kingdoms, rate=a.rate, resume=a.resume, hall=hall, snapshots=a.snapshots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
