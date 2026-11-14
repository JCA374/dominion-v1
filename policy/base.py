"""The Policy interface every playable model implements."""
from __future__ import annotations

from policy.protocol import PROTOCOL_VERSION, observation


class Policy:
    """A model that answers protocol requests.

    Subclasses implement the five decision methods. `handle()` turns a JSON
    request into a JSON answer, so a Policy can be served over a pipe as is.
    """

    kind = "abstract"

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.name = manifest.get("name", "unnamed")
        self.kingdom = list(manifest.get("kingdom", []))

    # -- decisions (override) ------------------------------------------------

    def play_action(self, obs: dict) -> str | None:
        raise NotImplementedError

    def buy(self, obs: dict) -> list[str]:
        raise NotImplementedError

    def trash(self, obs: dict) -> list[str]:
        raise NotImplementedError

    def discard(self, obs: dict, keep: int) -> list[str]:
        raise NotImplementedError

    def throne_target(self, obs: dict) -> str | None:
        raise NotImplementedError

    # -- the cards' own choices (protocol 2; see policy/protocol.py) -----------

    def cellar(self, obs: dict) -> list[str]:
        raise NotImplementedError

    def poacher(self, obs: dict, n: int) -> list[str]:
        raise NotImplementedError

    def remodel(self, obs: dict) -> dict:
        raise NotImplementedError

    def workshop(self, obs: dict) -> str | None:
        raise NotImplementedError

    def artisan(self, obs: dict) -> dict:
        raise NotImplementedError

    def mine(self, obs: dict) -> dict:
        raise NotImplementedError

    def moneylender(self, obs: dict) -> bool:
        raise NotImplementedError

    def harbinger(self, obs: dict, discard_pile: list[str]) -> str | None:
        raise NotImplementedError

    def vassal(self, obs: dict, card: str) -> bool:
        raise NotImplementedError

    def library(self, obs: dict, card: str) -> bool:
        raise NotImplementedError

    def sentry(self, obs: dict, cards: list[str]) -> dict:
        raise NotImplementedError

    def bandit(self, obs: dict, cards: list[str]) -> str | None:
        raise NotImplementedError

    def bureaucrat(self, obs: dict) -> str | None:
        raise NotImplementedError

    def phase(self, obs: dict) -> str | None:
        """Optional short label for logs (e.g. 'early')."""
        return None

    # -- protocol ------------------------------------------------------------

    def info(self) -> dict:
        return {"protocol": PROTOCOL_VERSION, "name": self.name, "kind": self.kind,
                "kingdom": self.kingdom,
                **{k: v for k, v in self.manifest.items()
                   if k in ("description", "created", "source", "trained_vs")}}

    def handle(self, request: dict) -> dict:
        kind = request.get("type")
        try:
            if kind == "info":
                return self.info()
            obs = observation(**request.get("obs", {}))
            if kind == "action":
                answer = {"play": self.play_action(obs)}
            elif kind == "buy":
                answer = {"buy": self.buy(obs)}
            elif kind == "trash":
                answer = {"trash": self.trash(obs)}
            elif kind == "discard":
                answer = {"discard": self.discard(obs, int(request.get("keep", 3)))}
            elif kind == "throne_room":
                answer = {"target": self.throne_target(obs)}
            elif kind == "cellar":
                answer = {"discard": self.cellar(obs)}
            elif kind == "poacher":
                answer = {"discard": self.poacher(obs, int(request.get("n", len(obs["empty_piles"]))))}
            elif kind in ("remodel", "artisan", "mine"):
                answer = dict(getattr(self, kind)(obs))
            elif kind == "workshop":
                answer = {"gain": self.workshop(obs)}
            elif kind == "moneylender":
                answer = {"trash": bool(self.moneylender(obs))}
            elif kind == "harbinger":
                answer = {"topdeck": self.harbinger(obs, list(request.get("discard_pile", [])))}
            elif kind == "vassal":
                answer = {"play": bool(self.vassal(obs, request["card"]))}
            elif kind == "library":
                answer = {"skip": bool(self.library(obs, request["card"]))}
            elif kind == "sentry":
                answer = dict(self.sentry(obs, list(request["cards"])))
            elif kind == "bandit":
                answer = {"trash": self.bandit(obs, list(request["cards"]))}
            elif kind == "bureaucrat":
                answer = {"topdeck": self.bureaucrat(obs)}
            else:
                return {"error": f"unknown request type {kind!r}"}
            phase = self.phase(obs)
            if phase is not None:
                answer["phase"] = phase
            return answer
        except Exception as e:  # report, don't crash the player
            return {"error": f"{type(e).__name__}: {e}"}
