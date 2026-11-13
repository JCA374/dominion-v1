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
Every answer may also carry "phase" (a short label for logs) or "error".
"""
from __future__ import annotations

PROTOCOL_VERSION = 1

REQUESTS = ("info", "action", "buy", "trash", "discard", "throne_room")

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
    else:
        problems.append(f"unknown request type {kind!r}")
    return problems
