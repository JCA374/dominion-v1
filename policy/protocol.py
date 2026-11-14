"""The decision protocol between a Dominion player (e.g. the Steam autopilot) and a model.

Everything is plain JSON so the same messages work in-process and over a pipe
(see policy/serve.py). The player owns perception and bookkeeping; the model
only answers questions. Full description: docs/PROTOCOL.md.

Observation (sent with every request):
    turn            int        current turn number (1 = first turn)
    provinces_left  int        Provinces left in the supply
    kingdom         [str]      the 10 kingdom cards of this game
    empty_piles     [str]      supply piles known to be empty
    owned           {str:int}  the player's whole deck (all zones), by card name
    hand            [str]      cards in hand right now
    actions, buys, coins  int  counters for the current turn

Requests and answers ({"type": ..., "obs": {...}, ...} -> {...}):
    info                       -> {"protocol", "name", "kind", "kingdom", ...}
    action                     -> {"play": card | null}        next action to play
    buy                        -> {"buy": [card, ...]}         all buys this turn, in order
    trash        (Chapel)      -> {"trash": [card, ...]}
    discard      + "keep": n   -> {"discard": [card, ...]}     attack: discard down to n
    throne_room                -> {"target": card | null}
  protocol 2: the choices of every other base-set card (the engine's own decisions)
    cellar                     -> {"discard": [card, ...]}     then draw that many
    poacher      + "n": k      -> {"discard": [card, ...]}     k = empty supply piles
    remodel                    -> {"trash": card, "gain": card}   gain costs up to trash + 2
    workshop                   -> {"gain": card}               costing up to 4
    artisan                    -> {"gain": card, "topdeck": card}  gain (up to 5) to hand,
                                                               then a hand card onto the deck
    mine                       -> {"trash": treasure | null, "gain": treasure}  up to +3, to hand
    moneylender                -> {"trash": bool}              trash a Copper for +$3
    harbinger    + "discard_pile": [card, ...] -> {"topdeck": card | null}
    vassal       + "card": c   -> {"play": bool}               play the discarded action?
    library      + "card": c   -> {"skip": bool}               set the drawn action aside?
    sentry       + "cards": [a, b] -> {"trash": [...], "discard": [...], "keep": [...]}
                                                               keep: top of the deck first
    bandit       + "cards": [..]   -> {"trash": card | null}  (attacked) revealed treasure to lose
    bureaucrat                 -> {"topdeck": card | null}    (attacked) victory card onto deck
Every answer may also carry "phase" (a short label for logs) or "error".
"""
from __future__ import annotations

PROTOCOL_VERSION = 2   # 2: a request for every card's own choices (cellar ... bureaucrat)

REQUESTS = ("info", "action", "buy", "trash", "discard", "throne_room",
            "cellar", "poacher", "remodel", "workshop", "artisan", "mine", "moneylender",
            "harbinger", "vassal", "library", "sentry", "bandit", "bureaucrat")

# Cards whose own choices no request above can express yet. A model is never told to buy one,
# so a player never meets their prompts. Since protocol 2 every base-set card has its request.
NO_REQUEST_YET: tuple[str, ...] = ()

OBS_DEFAULTS = {
    "turn": 1,
    "provinces_left": 8,
    "kingdom": [],
    "empty_piles": [],
    "owned": {"Copper": 7, "Estate": 3},
    "hand": [],
    "actions": 1,
    "buys": 1,
    "coins": 0,
}


def observation(**fields) -> dict:
    """Build an observation dict, filling in defaults for missing fields."""
    unknown = set(fields) - set(OBS_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown observation fields: {sorted(unknown)}")
    obs = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in OBS_DEFAULTS.items()}
    obs.update(fields)
    obs["hand"] = list(obs["hand"])
    obs["kingdom"] = list(obs["kingdom"])
    obs["empty_piles"] = sorted(obs["empty_piles"])
    obs["owned"] = {k: int(v) for k, v in dict(obs["owned"]).items() if v}
    return obs


def check_answer(request: dict, answer: dict, costs: dict[str, int] | None = None) -> list[str]:
    """Return a list of problems with a model's answer (empty = legal).

    Used by the conformance tests and by `python -m policy.check`; a player can
    call it too before acting on an answer.
    """
    problems = []
    if "error" in answer:
        return [f"model error: {answer['error']}"]
    kind = request["type"]
    obs = request.get("obs", {})
    hand = list(obs.get("hand", []))

    def from_hand(cards, what):
        rest = list(hand)
        for c in cards:
            if c in rest:
                rest.remove(c)
            else:
                problems.append(f"{what} {c!r} is not in hand {hand}")

    if kind == "info":
        for key in ("protocol", "name", "kind"):
            if key not in answer:
                problems.append(f"info lacks {key!r}")
        if answer.get("protocol") != PROTOCOL_VERSION:
            problems.append(f"protocol {answer.get('protocol')} != {PROTOCOL_VERSION}")
    elif kind == "action":
        card = answer.get("play", "missing")
        if card == "missing":
            problems.append("action answer lacks 'play'")
        elif card is not None:
            from_hand([card], "play")
            if obs.get("actions", 0) < 1:
                problems.append("played an action with no actions left")
    elif kind == "buy":
        buys = answer.get("buy")
        if not isinstance(buys, list):
            return ["buy answer lacks a 'buy' list"]
        if len(buys) > obs.get("buys", 1):
            problems.append(f"{len(buys)} buys with only {obs.get('buys', 1)}")
        if costs is not None:
            total = sum(costs.get(c, 0) for c in buys)
            if total > obs.get("coins", 0):
                problems.append(f"buys {buys} cost {total} > {obs.get('coins', 0)} coins")
        empty = set(obs.get("empty_piles", []))
        for c in buys:
            if c in empty:
                problems.append(f"bought {c!r} from an empty pile")
    elif kind == "trash":
        cards = answer.get("trash")
        if not isinstance(cards, list):
            return ["trash answer lacks a 'trash' list"]
        if len(cards) > 4:
            problems.append("Chapel trashes at most 4 cards")
        from_hand(cards, "trash")
    elif kind == "discard":
        cards = answer.get("discard")
        if not isinstance(cards, list):
            return ["discard answer lacks a 'discard' list"]
        need = max(0, len(hand) - request.get("keep", 3))
        if len(cards) != need:
            problems.append(f"discarded {len(cards)} cards, needed {need}")
        from_hand(cards, "discard")
    elif kind == "throne_room":
        card = answer.get("target", "missing")
        if card == "missing":
            problems.append("throne_room answer lacks 'target'")
        elif card is not None:
            rest = list(hand)
            if "Throne Room" in rest:
                rest.remove("Throne Room")
            if card not in rest:
                problems.append(f"target {card!r} is not in hand {hand}")
    elif kind in ("cellar", "poacher"):
        cards = answer.get("discard")
        if not isinstance(cards, list):
            return [f"{kind} answer lacks a 'discard' list"]
        from_hand(cards, "discard")
        if kind == "poacher" and len(cards) != min(request.get("n", 0), len(hand)):
            problems.append(f"poacher discarded {len(cards)}, needed {min(request.get('n', 0), len(hand))}")
    elif kind in ("remodel", "mine"):
        t, g = answer.get("trash", "missing"), answer.get("gain", "missing")
        if t == "missing" or g == "missing":
            return [f"{kind} answer needs 'trash' and 'gain'"]
        if t is not None:
            from_hand([t], "trash")
            if costs is not None and g is not None and costs.get(g, 99) > costs.get(t, 0) + (2 if kind == "remodel" else 3):
                problems.append(f"{kind}: {g} costs too much for {t}")
        elif kind == "remodel" and hand:
            problems.append("remodel must trash a card")
    elif kind in ("workshop", "artisan"):
        g = answer.get("gain", "missing")
        if g == "missing":
            return [f"{kind} answer lacks 'gain'"]
        if costs is not None and g is not None and costs.get(g, 99) > (4 if kind == "workshop" else 5):
            problems.append(f"{kind}: {g} costs too much")
        if kind == "artisan":
            top = answer.get("topdeck", "missing")
            if top == "missing":
                problems.append("artisan answer lacks 'topdeck'")
            elif top is not None and top not in hand + ([g] if g else []):
                problems.append(f"artisan topdeck {top!r} is not in hand")
    elif kind == "moneylender":
        if not isinstance(answer.get("trash"), bool):
            problems.append("moneylender answer needs a bool 'trash'")
        elif answer["trash"] and "Copper" not in hand:
            problems.append("moneylender trashes a Copper that isn't in hand")
    elif kind == "harbinger":
        top = answer.get("topdeck", "missing")
        if top == "missing":
            problems.append("harbinger answer lacks 'topdeck'")
        elif top is not None and top not in request.get("discard_pile", []):
            problems.append(f"harbinger topdeck {top!r} is not in the discard pile")
    elif kind in ("vassal", "library"):
        key = "play" if kind == "vassal" else "skip"
        if not isinstance(answer.get(key), bool):
            problems.append(f"{kind} answer needs a bool {key!r}")
    elif kind == "sentry":
        parts = [answer.get(k) for k in ("trash", "discard", "keep")]
        if not all(isinstance(x, list) for x in parts):
            return ["sentry answer needs 'trash', 'discard' and 'keep' lists"]
        if sorted(sum(parts, [])) != sorted(request.get("cards", [])):
            problems.append(f"sentry answer {answer} doesn't place exactly {request.get('cards')}")
    elif kind == "bandit":
        t = answer.get("trash", "missing")
        if t == "missing":
            problems.append("bandit answer lacks 'trash'")
        elif t is not None and t not in request.get("cards", []):
            problems.append(f"bandit trash {t!r} was not revealed")
    elif kind == "bureaucrat":
        t = answer.get("topdeck", "missing")
        if t == "missing":
            problems.append("bureaucrat answer lacks 'topdeck'")
        elif t is not None:
            from_hand([t], "topdeck")
    else:
        problems.append(f"unknown request type {kind!r}")
    return problems
