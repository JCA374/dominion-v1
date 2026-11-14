"""Model selector: which model should play a dealt kingdom (CLI: python -m policy.select).

    python -m policy.select Cellar Chapel Moat Village Smithy Militia Market Witch Mine Artisan
    python -m policy.select --best-kingdom       # the model + kingdom to play when nothing is given

    from policy import select_model, rank_models
    select_model(kingdom)["name"]

Rules, in order:
  0. A portfolio model (kind "ga-portfolio", policy/portfolio.py) plays any kingdom: it holds
     specialists for all 26 cards and picks one by playing them all on the dealt kingdom.
     Best stored score first.
  1. A model trained on exactly this kingdom plays it.
  2. Otherwise a model whose key cards (policy/keycards.py) are all in the kingdom can play
     it, whatever the other cards are; the one with the best stored benchmark score
     (bench.json "score", policy/bench.py) plays.
  3. If no model's key cards are all there, the model with the largest share of its key
     cards in the kingdom plays, best score first. There is no Big Money fallback: the
     answer is always a real model.
Models marked "retired": true in model.json are never chosen automatically (they can still
be played by name). A model with no known key cards counts as eligible only for its exact
training kingdom; one with no benchmark ranks below the benchmarked ones.
"""
from __future__ import annotations

import argparse
import sys

from policy.registry import key_cards, list_models, read_bench

PORTFOLIO, EXACT, KEY_CARDS, PARTIAL = 0, 1, 2, 3
TIER_NAME = {PORTFOLIO: "portfolio: picks a specialist per kingdom", EXACT: "trained on this kingdom",
             KEY_CARDS: "key cards all dealt",
             PARTIAL: "some key cards missing"}


def _norm(cards) -> set[str]:
    return {c.strip().title() for c in cards}


def rank_models(kingdom: list[str], models: list[dict] | None = None) -> list[dict]:
    """Every non-retired model, best first, with why: {"name", "tier", "score", "key_cards",
    "missing", "share", "reason", "manifest"}."""
    dealt = _norm(kingdom)
    rows = []
    for m in models if models is not None else list_models():
        if m.get("retired"):
            continue
        keys = key_cards(m)
        bench = read_bench(m)
        score = bench.get("score") if bench else None
        if m.get("kind") == "ga-portfolio":
            tier, missing, share = PORTFOLIO, [], 1.0
        elif m.get("kingdom") and _norm(m["kingdom"]) == dealt:
            tier, missing, share = EXACT, [], 1.0
        elif keys is None:
            tier, missing, share = PARTIAL, [], 0.0
        else:
            missing = [c for c in keys if c.title() not in dealt]
            total = sum(keys.values())
            share = 1.0 if not total else sum(n for c, n in keys.items() if c.title() in dealt) / total
            tier = KEY_CARDS if not missing else PARTIAL
        if tier == PORTFOLIO:
            reason = TIER_NAME[PORTFOLIO]
        elif tier == EXACT:
            reason = "trained on exactly this kingdom"
        elif tier == KEY_CARDS:
            reason = "key cards all dealt: " + (", ".join(keys) if keys else "none needed")
        elif keys is None:
            reason = "key cards unknown (run policy.bench --store or declare key_cards)"
        else:
            reason = "missing key cards: " + ", ".join(missing)
        rows.append({"name": m["name"], "tier": tier, "score": score, "key_cards": keys,
                     "missing": missing, "share": share, "reason": reason, "manifest": m})
    rows.sort(key=lambda r: (r["tier"], -r["share"] if r["tier"] == PARTIAL else 0,
                             r["score"] is None, -(r["score"] or 0), r["name"]))
    return rows


def select_model(kingdom: list[str], models: list[dict] | None = None) -> dict:
    """The best model for a kingdom (a rank_models row). Raises if there are no models."""
    ranked = rank_models(kingdom, models)
    if not ranked:
        raise LookupError("no models to choose from (all retired, or none released)")
    return ranked[0]


def best_kingdom(models: list[dict] | None = None) -> tuple[dict, list[str]]:
    """When the kingdom is ours to choose: the model that scores best on its own training
    kingdom (stored benchmark), and that kingdom. Models are compared on the opponents every
    one of them was benchmarked against there (Witch-BM only plays kingdoms with a Witch)."""
    cands = [m for m in (models if models is not None else list_models())
             if not m.get("retired") and m.get("kingdom") and m.get("kind") != "ga-portfolio"]
    if not cands:
        raise LookupError("no model with a training kingdom")
    train = {}
    for m in cands:
        bench = read_bench(m) or {}
        train[m["name"]] = {n: o["training"]["score"] for n, o in bench.get("opponents", {}).items()
                            if "training" in o}
    have = [t for t in train.values() if t]
    common = set.intersection(*(set(t) for t in have)) if have else set()

    def score(m):
        t = train[m["name"]]
        return sum(t[n] for n in common) / len(common) if common and t else -1.0
    best = max(cands, key=lambda m: (score(m), m["name"]))
    return best, list(best["kingdom"])


def main(argv=None) -> int:
    from policy.keycards import fmt
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kingdom", nargs="*", help="the dealt kingdom cards (quote two-word names)")
    ap.add_argument("--best-kingdom", action="store_true", help="pick a model and its kingdom")
    args = ap.parse_args(argv)
    if args.best_kingdom or not args.kingdom:
        m, k = best_kingdom()
        print(f"{m['name']} on its training kingdom: {', '.join(k)}")
        return 0
    for i, r in enumerate(rank_models(args.kingdom)):
        score = "   -  " if r["score"] is None else f"{100 * r['score']:5.1f}%"
        print(f"{'->' if i == 0 else '  '} {r['name']:<18} {score}  {fmt(r['key_cards']):<40} {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
