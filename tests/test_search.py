"""Rollout search on top of a genome (c/ga_agent.c search_decide, engine de_resume_buy/de_hide)."""
import random

from core.strategy import random_strategy, witch_big_money_strategy
from ga.c_bridge import evaluate_vs_opponent_c, search_vs
from ga.specialist import complete


def test_search_that_never_overrules_plays_the_genomes_games():
    """Rollouts run on copies, so with an impossible margin the real games are unchanged."""
    rng = random.Random(2)
    k = ["Village", "Smithy", "Market", "Witch", "Militia", "Moat", "Cellar", "Gardens", "Mine", "Library"]
    for _ in range(3):
        s, o = complete(random_strategy(rng, k)), witch_big_money_strategy(k)
        seeds = [rng.getrandbits(32) for _ in range(6)]
        r = search_vs(s, o, seeds, k, rollouts=5, margin=101)
        plain = evaluate_vs_opponent_c(s, seeds, k, o)
        assert r["changed"] == 0 and r["searched"] > 0
        assert abs(r["score"] - (plain["win_rate"] + plain["tie_rate"] / 2)) < 1e-9


def test_search_beats_its_genome_when_it_sees_the_opponent():
    rng = random.Random(3)
    k = ["Village", "Smithy", "Market", "Witch", "Militia", "Moat", "Cellar", "Gardens", "Mine", "Library"]
    s = witch_big_money_strategy(k)
    r = search_vs(s, s, [rng.getrandbits(32) for _ in range(40)], k, rollouts=60, margin=5)
    assert r["score"] > 0.53, r
