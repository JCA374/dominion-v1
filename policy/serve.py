"""Serve a model over stdin/stdout, one JSON request per line, one JSON answer per line.

    python -m policy.serve [MODEL]

This is the out-of-process side of the protocol: anything that speaks it (a
PyTorch model in its own venv, another language) can be plugged into the
autopilot with a "subprocess" model.json. Any model of this repo can be served
too, which is how the protocol is tested.
"""
from __future__ import annotations

import argparse
import json
import sys

from policy.base import Policy
from policy.registry import load_policy


def serve(policy: Policy, inp=sys.stdin, out=sys.stdout) -> None:
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            answer = {"error": f"bad JSON: {e}"}
        else:
            answer = policy.handle(request)
        out.write(json.dumps(answer) + "\n")
        out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", help="model name or path (default: models/DEFAULT)")
    args = ap.parse_args(argv)
    serve(load_policy(args.model))


if __name__ == "__main__":
    main()
