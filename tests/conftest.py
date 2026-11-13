"""Guard: the test suite must never write into the real best_model/ or models/.

(It once did: run_ga defaults best_model_dir to "best_model", which overwrote a
training run's hall of fame.)
"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDED = [os.path.join(ROOT, "best_model", "strategy.json"),
           os.path.join(ROOT, "best_model", "hall"),
           os.path.join(ROOT, "models")]


def _snapshot():
    snap = {}
    for top in GUARDED:
        if os.path.isfile(top):
            snap[top] = os.stat(top).st_mtime_ns
        for dirpath, _, files in os.walk(top):
            for f in files:
                p = os.path.join(dirpath, f)
                snap[p] = os.stat(p).st_mtime_ns
    return snap


@pytest.fixture(autouse=True, scope="session")
def _protect_model_dirs():
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    assert not changed, f"tests modified real model files: {changed[:5]}"
