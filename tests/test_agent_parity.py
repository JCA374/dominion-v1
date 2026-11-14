"""The genome plays identical games as a Python agent (core/agent.py) and a C agent (c/ga_agent.c).

Training runs in C and the policy/play modes run in Python, so this is what makes a model play
on Steam the way it was evaluated. Random genomes on random base-set kingdoms reach every card.
"""
import random

from core.cards import KINGDOM_CARDS
from core.engine import play_game_2p
from core.strategy import big_money_strategy, engine_strategy, random_strategy
from ga.c_bridge import play_one_c


def _cases(n):
    for s in range(n):
        rnd = random.Random(s)
        kingdom = rnd.sample(KINGDOM_CARDS, 10)
        pick = [lambda: random_strategy(rnd, kingdom), lambda: big_money_strategy(kingdom),
                lambda: engine_strategy(kingdom)]
        yield kingdom, rnd.choice(pick)(), random_strategy(rnd, kingdom), rnd.getrandbits(31)


def test_python_and_c_agents_play_the_same_games():
    for kingdom, s0, s1, seed in _cases(400):
        py = play_game_2p(s0, s1, seed, kingdom)
        c = play_one_c(s0, s1, seed, kingdom)
        assert not c["error"]
        assert ([py["vp1"], py["vp2"]], py["turns"], py["winners"], py["trace"],
                [py["actions1"], py["actions2"]]) == \
            (c["scores"], c["turns"], c["winners"], c["trace"], c["actions"]), (kingdom, seed)
