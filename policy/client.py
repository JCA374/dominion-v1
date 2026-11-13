"""A Policy that runs in another process and is reached over the JSON-lines pipe.

model.json for such a model:
    {"protocol": 1, "name": "my-net", "kind": "subprocess",
     "command": ["/path/to/venv/bin/python", "-m", "mynet.serve"],
     "cwd": "/path/to/project", "kingdom": [...]}
"""
from __future__ import annotations

import json
import os
import subprocess

from policy.base import Policy
from policy.protocol import PROTOCOL_VERSION


class SubprocessPolicy(Policy):
    kind = "subprocess"

    def __init__(self, manifest: dict):
        super().__init__(manifest)
        cwd = manifest.get("cwd")
        env = dict(os.environ)
        if cwd:
            env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen(manifest["command"], cwd=cwd, env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1)
        self.remote = self.request({"type": "info"})
        if self.remote.get("protocol") != PROTOCOL_VERSION:
            self.close()
            raise RuntimeError(f"{self.name}: remote speaks protocol "
                               f"{self.remote.get('protocol')}, need {PROTOCOL_VERSION}")
        self.kingdom = self.kingdom or list(self.remote.get("kingdom", []))
        self._last_phase = None

    def request(self, msg: dict) -> dict:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: model process exited (code {self.proc.poll()})")
        return json.loads(line)

    def handle(self, request: dict) -> dict:
        if request.get("type") == "info":
            return {**self.remote, "name": self.name, "kind": self.kind}
        return self.request(request)

    def _ask(self, kind: str, obs: dict, key: str, **extra):
        answer = self.request({"type": kind, "obs": obs, **extra})
        if "error" in answer:
            raise RuntimeError(f"{self.name}: {answer['error']}")
        self._last_phase = answer.get("phase")
        return answer[key]

    def play_action(self, obs):
        return self._ask("action", obs, "play")

    def buy(self, obs):
        return self._ask("buy", obs, "buy")

    def trash(self, obs):
        return self._ask("trash", obs, "trash")

    def discard(self, obs, keep):
        return self._ask("discard", obs, "discard", keep=keep)

    def throne_target(self, obs):
        return self._ask("throne_room", obs, "target")

    def phase(self, obs):
        return self._last_phase

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
