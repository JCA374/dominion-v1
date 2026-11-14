"""Stable interface for playing a trained model from outside this repo.

    from policy import load_policy, observation
    p = load_policy("k1-steam")                 # or a path, or None for models/DEFAULT
    p.buy(observation(coins=6, kingdom=[...]))  # -> ["Gold"]
    select_model(kingdom)["name"]               # the best model for a dealt kingdom

The protocol is documented in docs/PROTOCOL.md; players should import only
from this package, never from core/ or ga/ directly.
"""
from policy import cards
from policy.base import Policy
from policy.protocol import PROTOCOL_VERSION, check_answer, observation
from policy.registry import default_model, key_cards, list_models, load_policy, read_bench
from policy.selector import best_kingdom, rank_models, select_model

__all__ = ["PROTOCOL_VERSION", "Policy", "best_kingdom", "cards", "check_answer", "default_model",
           "key_cards", "list_models", "load_policy", "observation", "rank_models", "read_bench",
           "select_model"]
