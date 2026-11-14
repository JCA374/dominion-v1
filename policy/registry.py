"""Released models: models/<name>/model.json, and how to load any of them.

A model reference can be
  - a released model name             "k1-gen-steam"
  - a model directory                 models/k1-gen-steam, best_model
  - a model.json or a bare strategy.json
  - None                              the name in models/DEFAULT

model.json (the manifest):
  {"protocol": 1, "name": ..., "kind": "ga-strategy" | "ga-portfolio" | "subprocess",
   "kingdom": [...], "file": "strategy.json",            # ga-strategy
   "command": ["python", "-m", "..."], "cwd": "...",      # subprocess
   "created": "YYYY-MM-DD", "source": ..., "description": ...,
   "key_cards": {"Smithy": 2, "Witch": 1}}               # optional, see policy/keycards.py

Models are plug-ins: a model is a folder, found in models/ and in every folder listed
in the DOMINION_MODELS environment variable (os.pathsep-separated). A player can keep its
own models folder there, and adding a model never needs a code change. A name in models/
wins over the same name in an extra folder.

Next to model.json a folder may hold bench.json, written by `python -m policy.bench --store`;
the model selector (policy/select.py) reads it.
"""
from __future__ import annotations

import json
import os

from policy.protocol import PROTOCOL_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_FILE = os.path.join(MODELS_DIR, "DEFAULT")
BENCH_FILE = "bench.json"
BASE_CARDS = {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse", "PASS"}


def kingdom_of(strategy_data: dict) -> list[str]:
    """The kingdom a GA strategy was trained on (its buy lists hold exactly those cards)."""
    return sorted(set(strategy_data["early_buy_priority"]) - BASE_CARDS)


def default_model() -> str | None:
    if os.path.exists(DEFAULT_FILE):
        return open(DEFAULT_FILE).read().strip() or None
    return None


def model_dirs() -> list[str]:
    """Folders that hold model folders: models/, then DOMINION_MODELS."""
    extra = [d for d in os.environ.get("DOMINION_MODELS", "").split(os.pathsep) if d]
    return [MODELS_DIR] + [os.path.abspath(d) for d in extra]


def list_models() -> list[dict]:
    """Manifests of all models in all model folders, sorted by name."""
    found = {}
    for base in model_dirs():
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name, "model.json")
            if os.path.exists(path) and name not in found:
                found[name] = read_manifest(path)
    return [found[n] for n in sorted(found)]


def read_bench(manifest: dict) -> dict | None:
    """The stored benchmark of a model (bench.json next to its model.json), if any."""
    path = os.path.join(manifest.get("dir", ""), BENCH_FILE)
    if manifest.get("dir") and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def key_cards(manifest: dict) -> dict[str, int] | None:
    """The kingdom cards a model is built around: declared in model.json, else derived
    by the benchmark runner (bench.json). None if neither knows."""
    if manifest.get("key_cards") is not None:
        return dict(manifest["key_cards"])
    bench = read_bench(manifest)
    if bench and bench.get("key_cards") is not None:
        return dict(bench["key_cards"])
    return None


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
        manifest["dir"] = base
        if manifest.get("protocol", PROTOCOL_VERSION) > PROTOCOL_VERSION:
            raise ValueError(f"{path}: model needs protocol {manifest['protocol']}, "
                             f"this code speaks {PROTOCOL_VERSION}")
        return manifest
    # bare GA strategy.json
    folder = os.path.basename(os.path.dirname(path))
    return {"protocol": PROTOCOL_VERSION, "name": folder, "kind": "ga-strategy",
            "file": os.path.basename(path), "path": path, "kingdom": kingdom_of(data),
            "dir": os.path.dirname(path)}


def resolve(ref: str | None) -> str:
    """Turn a model reference into a path."""
    if ref is None:
        ref = default_model()
        if ref is None:
            raise FileNotFoundError("no model given and models/DEFAULT is missing")
    if os.path.exists(ref):
        return ref
    for base in model_dirs():
        named = os.path.join(base, ref)
        if os.path.exists(os.path.join(named, "model.json")):
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
    if kind == "ga-portfolio":
        from policy.portfolio import PortfolioPolicy
        return PortfolioPolicy(manifest)
    if kind == "subprocess":
        from policy.client import SubprocessPolicy
        return SubprocessPolicy(manifest)
    raise ValueError(f"unknown model kind {kind!r} (known: ga-strategy, ga-portfolio, subprocess)")
