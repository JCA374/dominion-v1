"""Release a trained model into models/<name>/ so players (the Steam autopilot) can use it.

    python -m policy.release best_model k4-gen49009 --description "..." [--default]

Copies strategy.json, summary.txt and buy_heatmap.png, writes model.json
(kingdom, date, source, headline win rate), and runs the conformance check.
best_model/ keeps changing while the GA trains; models/<name>/ never does.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

from policy.protocol import PROTOCOL_VERSION
from policy.registry import DEFAULT_FILE, MODELS_DIR, ROOT, kingdom_of


def _git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def release(src: str, name: str, description: str = "", make_default: bool = False,
            force: bool = False) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise ValueError("name: lowercase letters, digits, '.', '_', '-'")
    src_file = src if src.endswith(".json") else os.path.join(src, "strategy.json")
    src_dir = os.path.dirname(src_file)
    dst = os.path.join(MODELS_DIR, name)
    if os.path.exists(dst) and not force:
        raise FileExistsError(f"{dst} exists (released models are immutable; pick a new name)")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(src_file, os.path.join(dst, "strategy.json"))
    for extra in ("summary.txt", "buy_heatmap.png"):
        if os.path.exists(os.path.join(src_dir, extra)):
            shutil.copy(os.path.join(src_dir, extra), os.path.join(dst, extra))

    data = json.load(open(src_file))
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "name": name,
        "kind": "ga-strategy",
        "file": "strategy.json",
        "kingdom": kingdom_of(data),
        "created": datetime.date.today().isoformat(),
        "source": f"{os.path.relpath(os.path.abspath(src_file), ROOT)} @ {_git_rev()}",
        "description": description,
    }
    summary = os.path.join(dst, "summary.txt")
    if os.path.exists(summary):
        text = open(summary).read()
        m = re.search(r"Performance \((.+?)\)\s*\n.*?\n\s*Win rate:\s*(.+)", text)
        if m:
            manifest["trained_vs"] = f"{m.group(2).strip()} ({m.group(1)})"
    with open(os.path.join(dst, "model.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    from policy.check import check
    from policy.registry import load_policy
    problems = check(load_policy(dst))
    if problems:
        raise RuntimeError(f"conformance check failed for {name}: {problems[:3]}")
    if make_default:
        with open(DEFAULT_FILE, "w") as f:
            f.write(name + "\n")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="model directory or strategy.json (e.g. best_model)")
    ap.add_argument("name", help="release name, e.g. k4-gen49009")
    ap.add_argument("--description", default="")
    ap.add_argument("--default", action="store_true", help="make it models/DEFAULT")
    ap.add_argument("--force", action="store_true", help="overwrite an existing release")
    args = ap.parse_args(argv)
    dst = release(args.src, args.name, args.description, args.default, args.force)
    print(f"released {args.name} -> {os.path.relpath(dst, ROOT)}"
          + (" (default)" if args.default else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
