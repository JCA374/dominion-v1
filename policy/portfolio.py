"""Portfolio model (kind "ga-portfolio"): several specialist genomes, and a selector that picks
one for the dealt kingdom by playing them all on it.

    python -m policy.portfolio build portfolio-v1 runs/witch runs/militia ... k4-20260614 [--description ...]
    python -m policy.portfolio choose portfolio-v1 Cellar Chapel Moat ... Witch   # who plays?

models/<name>/model.json lists the members (members/<member>.json, each trained by
ga/specialist.py on random kingdoms that hold its key cards). When a game starts the policy
plays every member on that exact kingdom against the benchmark bots that fit it (Big Money,
plus Smithy-BM / Witch-BM when Smithy / Witch are dealt), on the C engine, and the best
member plays the whole game. Members play as the player will let them: cards in
policy.protocol.NO_REQUEST_YET are never bought, in the trial games too, so the choice is
made for the game the member will really play. About 1 s per kingdom.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys

import core  # noqa: F401
from core.cards import KINGDOM_CARDS
from core.strategy import Strategy, load_strategy
from policy.base import Policy
from policy.protocol import NO_REQUEST_YET, PROTOCOL_VERSION
from policy.registry import MODELS_DIR, ROOT

KIND = "ga-portfolio"
TRIAL_SEEDS = 120     # seeds per bot (x2 seats) in the trial games (40: -0.7 points)


def members(manifest: dict) -> dict[str, Strategy]:
    return {m["name"]: load_strategy(os.path.join(manifest["dir"], m["file"]))
            for m in manifest["members"]}


def choose(manifest: dict, kingdom: list[str], banned=NO_REQUEST_YET, n_seeds: int = TRIAL_SEEDS,
           strategies: dict[str, Strategy] | None = None, seed: int = 7) -> dict:
    """{"member", "scores": {member: score}} for one kingdom (cards the engine doesn't know are
    ignored). Ties go to the member listed first."""
    from ga.c_bridge import evaluate_vs_opponent_c
    from ga.specialist import mask
    from policy.bench import OPPONENTS
    k = sorted({c for c in kingdom if c in KINGDOM_CARDS})
    strategies = strategies if strategies is not None else members(manifest)
    rng = random.Random(f"{seed}:{','.join(k)}")
    seeds = [rng.getrandbits(32) for _ in range(n_seeds)]
    bots = [(n, b) for n, (b, card) in OPPONENTS.items() if card is None or card in k]
    scores = {}
    for name, s in strategies.items():
        s = mask(s, banned)
        tot = 0.0
        for _, bot in bots:
            r = evaluate_vs_opponent_c(s, seeds, k, bot(k))
            tot += r["win_rate"] + r["tie_rate"] / 2
        scores[name] = tot / len(bots)
    best = max(scores, key=lambda n: scores[n])     # dicts keep order: first wins a tie
    return {"member": best, "scores": scores, "bots": [n for n, _ in bots]}


class PortfolioPolicy(Policy):
    kind = KIND

    def __init__(self, manifest: dict):
        super().__init__(manifest)
        self.strategies = members(manifest)
        self.chosen: dict | None = None
        self._for: tuple | None = None
        self._policy = None

    def member(self, obs: dict):
        """The GAPolicy of the member chosen for this observation's kingdom."""
        from policy.ga_policy import GAPolicy
        key = tuple(sorted(obs["kingdom"]))
        if self._policy is None or key != self._for:
            self.chosen = choose(self.manifest, list(key), strategies=self.strategies)
            name = self.chosen["member"]
            self._policy = GAPolicy({"name": name, "kingdom": list(KINGDOM_CARDS)},
                                    self.strategies[name])
            self._for = key
        return self._policy

    def play_action(self, obs):
        return self.member(obs).play_action(obs)

    def buy(self, obs):
        return self.member(obs).buy(obs)

    def trash(self, obs):
        return self.member(obs).trash(obs)

    def discard(self, obs, keep):
        return self.member(obs).discard(obs, keep)

    def throne_target(self, obs):
        return self.member(obs).throne_target(obs)

    def cellar(self, obs):
        return self.member(obs).cellar(obs)

    def poacher(self, obs, n):
        return self.member(obs).poacher(obs, n)

    def remodel(self, obs):
        return self.member(obs).remodel(obs)

    def workshop(self, obs):
        return self.member(obs).workshop(obs)

    def artisan(self, obs):
        return self.member(obs).artisan(obs)

    def mine(self, obs):
        return self.member(obs).mine(obs)

    def moneylender(self, obs):
        return self.member(obs).moneylender(obs)

    def harbinger(self, obs, discard_pile):
        return self.member(obs).harbinger(obs, discard_pile)

    def vassal(self, obs, card):
        return self.member(obs).vassal(obs, card)

    def library(self, obs, card):
        return self.member(obs).library(obs, card)

    def sentry(self, obs, cards):
        return self.member(obs).sentry(obs, cards)

    def bandit(self, obs, cards):
        return self.member(obs).bandit(obs, cards)

    def bureaucrat(self, obs):
        return self.member(obs).bureaucrat(obs)

    def phase(self, obs):
        p = self.member(obs)
        return f"{self.chosen['member']}:{p.phase(obs)}"


# -- building a portfolio from specialist runs ---------------------------------------------

def build(name: str, runs: list[str], description: str = "", force: bool = False) -> str:
    """runs: ga/specialist.py run folders, or names of released ga-strategy models."""
    from dataclasses import asdict
    from ga.specialist import complete
    from policy.registry import read_manifest, resolve
    dst = os.path.join(MODELS_DIR, name)
    if os.path.exists(dst):
        if not force:
            raise FileExistsError(f"{dst} exists (released models are immutable; pick a new name)")
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "members"))
    rows = []
    for run in runs:
        if os.path.exists(os.path.join(run, "progress.json")):        # a ga/specialist.py run
            prog = json.load(open(os.path.join(run, "progress.json")))
            member = os.path.basename(os.path.normpath(run))
            shutil.copy(os.path.join(run, "strategy.json"), os.path.join(dst, "members", member + ".json"))
            rows.append({"name": member, "file": f"members/{member}.json", "must": prog["must"],
                         "banned": prog.get("banned", []), "val_score": prog.get("best_val"),
                         "generations": prog.get("gen")})
            continue
        m = read_manifest(resolve(run))                                # a released ga-strategy model
        if m.get("kind") != "ga-strategy":
            raise ValueError(f"{run}: not a specialist run or a ga-strategy model")
        with open(os.path.join(dst, "members", m["name"] + ".json"), "w") as f:
            json.dump(asdict(complete(load_strategy(m["path"]))), f, indent=1)
        rows.append({"name": m["name"], "file": f"members/{m['name']}.json",
                     "must": sorted(m.get("key_cards") or {}), "banned": [],
                     "released_model": m["name"]})
    manifest = {"protocol": PROTOCOL_VERSION, "name": name, "kind": KIND,
                "kingdom": list(KINGDOM_CARDS), "key_cards": {}, "members": rows,
                "created": datetime.date.today().isoformat(),
                "source": "ga/specialist.py runs: " + ", ".join(r["name"] for r in rows),
                "description": description}
    with open(os.path.join(dst, "model.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    from policy.check import check
    from policy.registry import load_policy
    problems = check(load_policy(dst))
    if problems:
        raise RuntimeError(f"conformance check failed for {name}: {problems[:3]}")
    return dst


def evaluate(manifest: dict, n_kingdoms: int = 100, n_seeds: int = 40, banned=NO_REQUEST_YET,
             compare=("k4-20260614", "k1-steam"), seed: int = 11) -> dict:
    """How good is the per-kingdom choice? On n random Steam kingdoms: the member the trial
    picks, scored on fresh games (n_seeds per bot, both seats), against the best member in
    hindsight (oracle), every member alone and the `compare` models. Score as policy.bench."""
    from ga.c_bridge import evaluate_vs_opponent_c
    from ga.specialist import mask
    from policy.bench import OPPONENTS, STEAM_BASE_SET, kingdom_seeds, load
    rng = random.Random(seed)
    strategies = members(manifest)
    others = {}
    for ref in compare:
        try:
            others[ref] = load(ref)[1]
        except FileNotFoundError:
            pass
    players = {n: mask(s, banned) for n, s in strategies.items()}
    rows, pick_count = [], {}
    base = [rng.getrandbits(32) for _ in range(n_seeds)]
    for i in range(n_kingdoms):
        k = sorted(rng.sample(STEAM_BASE_SET, 10))
        seeds = kingdom_seeds(base, i)
        bots = [b for _, (b, card) in OPPONENTS.items() if card is None or card in k]

        def sc(s):
            return sum((lambda r: r["win_rate"] + r["tie_rate"] / 2)(
                evaluate_vs_opponent_c(s, seeds, k, b(k))) for b in bots) / len(bots)
        true = {n: sc(s) for n, s in players.items()}
        pick = choose(manifest, k, banned, strategies=strategies)["member"]
        pick_count[pick] = pick_count.get(pick, 0) + 1
        rows.append({"kingdom": k, "pick": pick, "picked": true[pick], "oracle": max(true.values()),
                     "members": true, **{f"model:{r}": sc(s) for r, s in others.items()}})
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    out = {"kingdoms": n_kingdoms, "seeds": n_seeds, "banned": list(banned),
           "portfolio": mean("picked"), "oracle": mean("oracle"),
           "members": {n: sum(r["members"][n] for r in rows) / len(rows) for n in players},
           "models": {r: mean(f"model:{r}") for r in others}, "picks": pick_count, "rows": rows}
    return out


def versus(a: dict, b: dict, n_kingdoms: int = 200, n_seeds: int = 20, seed: int = 13) -> dict:
    """Head to head on n random Steam kingdoms: each portfolio picks its own member for the
    kingdom (as in a real game), then the two play n_seeds games from each seat. Score of `a`
    = (wins + ties/2) / games."""
    from ga.c_bridge import evaluate_vs_opponent_c
    from policy.bench import STEAM_BASE_SET, kingdom_seeds
    rng = random.Random(seed)
    sa, sb = members(a), members(b)
    base = [rng.getrandbits(32) for _ in range(n_seeds)]
    tot, rows = 0.0, []
    for i in range(n_kingdoms):
        k = sorted(rng.sample(STEAM_BASE_SET, 10))
        pa = choose(a, k, strategies=sa)["member"]
        pb = choose(b, k, strategies=sb)["member"]
        r = evaluate_vs_opponent_c(sa[pa], kingdom_seeds(base, i), k, sb[pb])
        score = r["win_rate"] + r["tie_rate"] / 2
        tot += score
        rows.append({"kingdom": k, "a": pa, "b": pb, "score": score})
    return {"score": tot / n_kingdoms, "kingdoms": n_kingdoms, "seeds": n_seeds, "rows": rows}


def main(argv=None) -> int:
    from policy.registry import read_manifest, resolve
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="release specialist runs as one portfolio model")
    b.add_argument("name")
    b.add_argument("runs", nargs="+", help="runs/<name> folders of ga/specialist.py, or released models")
    b.add_argument("--description", default="")
    b.add_argument("--force", action="store_true")
    e = sub.add_parser("evaluate", help="choice vs oracle, members alone and old models")
    e.add_argument("model")
    e.add_argument("--kingdoms", type=int, default=100)
    e.add_argument("--seeds", type=int, default=40)
    e.add_argument("--all-cards", action="store_true", help="members may buy every card")
    e.add_argument("--json", metavar="FILE")
    v = sub.add_parser("versus", help="head to head: A's pick vs B's pick on random kingdoms")
    v.add_argument("a")
    v.add_argument("b")
    v.add_argument("--kingdoms", type=int, default=200)
    v.add_argument("--seeds", type=int, default=20)
    v.add_argument("--json", metavar="FILE")
    c = sub.add_parser("choose", help="which member plays this kingdom")
    c.add_argument("model")
    c.add_argument("kingdom", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        dst = build(a.name, a.runs, a.description, a.force)
        print(f"built {a.name} -> {os.path.relpath(dst, ROOT)}")
        return 0
    if a.cmd == "versus":
        r = versus(read_manifest(resolve(a.a)), read_manifest(resolve(a.b)), a.kingdoms, a.seeds)
        print(f"{a.a} vs {a.b}: {100 * r['score']:.1f}% for {a.a} "
              f"({a.kingdoms} random kingdoms x {a.seeds} seeds x 2 seats)")
        if a.json:
            with open(a.json, "w") as f:
                json.dump(r, f, indent=1)
        return 0
    if a.cmd == "evaluate":
        r = evaluate(read_manifest(resolve(a.model)), a.kingdoms, a.seeds,
                     () if a.all_cards else NO_REQUEST_YET)
        pct = lambda x: f"{100 * x:5.1f}%"
        print(f"{a.kingdoms} random Steam kingdoms x {a.seeds} seeds x 2 seats per bot"
              f"{' (all cards)' if a.all_cards else ' (blocked cards never bought)'}")
        print(f"portfolio (trial pick)  {pct(r['portfolio'])}\noracle (best member)    {pct(r['oracle'])}")
        for n, v in sorted(r["models"].items(), key=lambda x: -x[1]):
            print(f"model {n:<17} {pct(v)}")
        for n, v in sorted(r["members"].items(), key=lambda x: -x[1]):
            print(f"  member {n:<15} {pct(v)}   picked {r['picks'].get(n, 0)}x")
        if a.json:
            with open(a.json, "w") as f:
                json.dump(r, f, indent=1)
        return 0
    r = choose(read_manifest(resolve(a.model)), a.kingdom)
    for n, s in sorted(r["scores"].items(), key=lambda x: -x[1]):
        print(f"{'->' if n == r['member'] else '  '} {n:<16} {100 * s:5.1f}%")
    print(f"vs {', '.join(r['bots'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
