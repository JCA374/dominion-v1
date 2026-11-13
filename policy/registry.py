"""Released models: models/<name>/model.json, and how to load any of them.

A model reference can be
  - a released model name             "k1-gen-steam"
  - a model directory                 models/k1-gen-steam, best_model
  - a model.json or a bare strategy.json
  - None                              the name in models/DEFAULT

model.json (the manifest):
  {"protocol": 1, "name": ..., "kind": "ga-strategy" | "subprocess",
   "kingdom": [...], "file": "strategy.json",            # ga-strategy
   "command": ["python", "-m", "..."], "cwd": "...",      # subprocess
   "created": "YYYY-MM-DD", "source": ..., "description": ...}
"""
from __future__ import annotations

import json
import os

from policy.protocol import PROTOCOL_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_FILE = os.path.join(MODELS_DIR, "DEFAULT")
BASE_CARDS = {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse", "PASS"}


def kingdom_of(strategy_data: dict) -> list[str]:
    """The kingdom a GA strategy was trained on (its buy lists hold exactly those cards)."""
    return sorted(set(strategy_data["early_buy_priority"]) - BASE_CARDS)


def default_model() -> str | None:
    if os.path.exists(DEFAULT_FILE):
        return open(DEFAULT_FILE).read().strip() or None
    return None


def list_models() -> list[dict]:
    """Manifests of all released models, sorted by name."""
    out = []
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            path = os.path.join(MODELS_DIR, name, "model.json")
            if os.path.exists(path):
                out.append(read_manifest(path))
    return out


def read_manifest(path: str) -> dict:
    """Read a model.json, or synthesise a manifest for a bare strategy.json / directory."""
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "model.json")):
            path = os.path.join(path, "model.json")
        else:
            path = os.path.join(path, "strategy.json")
    path = os.path.abspath(path)
    with open(path) as f:
        data = json.load(f)
    if os.path.basename(path) == "model.json":
        manifest = dict(data)
        base = os.path.dirname(path)
        if "file" in manifest:
            manifest["path"] = os.path.join(base, manifest["file"])
        manifest.setdefault("cwd", base)
        if manifest.get("protocol", PROTOCOL_VERSION) > PROTOCOL_VERSION:
            raise ValueError(f"{path}: model needs protocol {manifest['protocol']}, "
                             f"this code speaks {PROTOCOL_VERSION}")
        return manifest
    # bare GA strategy.json
    folder = os.path.basename(os.path.dirname(path))
    return {"protocol": PROTOCOL_VERSION, "name": folder, "kind": "ga-strategy",
            "file": os.path.basename(path), "path": path, "kingdom": kingdom_of(data)}


def resolve(ref: str | None) -> str:
    """Turn a model reference into a path."""
    if ref is None:
        ref = default_model()
        if ref is None:
            raise FileNotFoundError("no model given and models/DEFAULT is missing")
    if os.path.exists(ref):
        return ref
    named = os.path.join(MODELS_DIR, ref)
    if os.path.isdir(named):
        return named
    names = [m["name"] for m in list_models()]
    raise FileNotFoundError(f"model {ref!r} not found; released models: {', '.join(names)}")


def load_policy(ref: str | None = None):
    """Load any model as a Policy."""
    manifest = read_manifest(resolve(ref))
    kind = manifest.get("kind")
    if kind == "ga-strategy":
        from policy.ga_policy import GAPolicy
        return GAPolicy(manifest)
    if kind == "subprocess":
        from policy.client import SubprocessPolicy
        return SubprocessPolicy(manifest)
    raise ValueError(f"unknown model kind {kind!r} (known: ga-strategy, subprocess)")
