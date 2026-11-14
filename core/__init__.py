"""GA genome and its agent. The game itself is the shared engine in engine/ (git submodule)."""
import os
import sys

_ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)
