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
            else:
                return {"error": f"unknown request type {kind!r}"}
            phase = self.phase(obs)
            if phase is not None:
                answer["phase"] = phase
            return answer
        except Exception as e:  # report, don't crash the player
            return {"error": f"{type(e).__name__}: {e}"}
