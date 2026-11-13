"""Stable interface for playing a trained model from outside this repo.

    from policy import load_policy, observation
    p = load_policy("k1-steam")                 # or a path, or None for models/DEFAULT
    p.buy(observation(coins=6, kingdom=[...]))  # -> ["Gold"]

The protocol is documented in docs/PROTOCOL.md; players should import only
from this package, never from core/ or ga/ directly.
"""
from policy import cards
from policy.base import Policy
from policy.protocol import PROTOCOL_VERSION, check_answer, observation
from policy.registry import default_model, list_models, load_policy

__all__ = ["PROTOCOL_VERSION", "Policy", "cards", "check_answer", "default_model", "list_models",
           "load_policy", "observation"]
