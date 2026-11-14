"""Conformance check: ask a model a set of real-game questions and validate every answer.

    python -m policy.check [MODEL ...]      # default: every released model

Run it before releasing a model or before pointing the autopilot at it.
"""
from __future__ import annotations

import argparse
import random
import sys

from core.cards import ALL_CARDS, CardType
from policy.protocol import check_answer, observation
from policy.registry import list_models, load_policy

COSTS = {name: card.cost for name, card in ALL_CARDS.items()}


def scenarios(kingdom: list[str], n: int = 60, seed: int = 1) -> list[dict]:
    """Requests spread over a whole game: opening, mid game, endgame, attacks."""
    rng = random.Random(seed)
    known = [c for c in kingdom if c in ALL_CARDS]
    actions = [c for c in known if ALL_CARDS[c].card_type == CardType.ACTION]
    pool = ["Copper"] * 6 + ["Silver"] * 3 + ["Gold", "Estate", "Duchy", "Province"] + actions * 2
    reqs = [{"type": "info"}]
    for i in range(n):
        turn = 1 + i * 30 // n
        owned = {"Copper": 7, "Estate": 3}
        for c in rng.sample(pool, k=min(len(pool), turn)):
            owned[c] = owned.get(c, 0) + 1
        hand = rng.sample(pool, k=5)
        if actions and i % 3 == 0:
            hand[0] = rng.choice(actions)
        obs = observation(turn=turn, provinces_left=max(0, 8 - turn // 4), kingdom=kingdom,
                          owned=owned, hand=hand, actions=rng.choice([0, 1, 2]),
                          buys=rng.choice([1, 1, 2]), coins=rng.randint(0, 11),
                          empty_piles=rng.sample(known, k=1) if known and i % 7 == 0 else [])
        reqs.append({"type": "action", "obs": obs})
        reqs.append({"type": "buy", "obs": obs})
        reqs.append({"type": "trash", "obs": obs})
        reqs.append({"type": "throne_room", "obs": {**obs, "hand": ["Throne Room"] + hand[:4]}})
        reqs.append({"type": "discard", "keep": 3, "obs": obs})
        for kind in ("cellar", "remodel", "workshop", "artisan", "mine", "moneylender", "bureaucrat"):
            reqs.append({"type": kind, "obs": obs})
        reqs.append({"type": "poacher", "n": rng.choice([1, 2]), "obs": obs})
        reqs.append({"type": "harbinger", "discard_pile": rng.sample(pool, k=4), "obs": obs})
        reqs.append({"type": "vassal", "card": rng.choice(pool), "obs": obs})
        reqs.append({"type": "library", "card": rng.choice(pool), "obs": obs})
        reqs.append({"type": "sentry", "cards": rng.sample(pool, k=2), "obs": obs})
        reqs.append({"type": "bandit", "cards": rng.sample(["Silver", "Gold", "Copper", "Estate"], k=2),
                     "obs": obs})
    return reqs


def check(policy, kingdom: list[str] | None = None) -> list[str]:
    problems = []
    for req in scenarios(kingdom or policy.kingdom):
        answer = policy.handle(req)
        for p in check_answer(req, answer, COSTS):
            problems.append(f"{req['type']}: {p}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*")
    args = ap.parse_args(argv)
    refs = args.models or [m["name"] for m in list_models()]
    bad = 0
    for ref in refs:
        policy = load_policy(ref)
        problems = check(policy)
        status = "ok" if not problems else f"{len(problems)} problems"
        print(f"{policy.name:24s} {policy.kind:12s} {status}")
        for p in problems[:10]:
            print("   ", p)
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
