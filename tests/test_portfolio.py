"""Portfolio model: specialist genomes over all 26 cards, and the per-kingdom member choice."""
import json
import random

import pytest

from core.cards import BUYABLE_CARDS, KINGDOM_CARDS
from core.strategy import engine_strategy, random_strategy, witch_big_money_strategy
from ga.c_bridge import play_one_c
from ga.specialist import complete, make_plan, mask, score
from policy.check import check
from policy.portfolio import KIND, PortfolioPolicy, choose
from policy.protocol import NO_REQUEST_YET, observation


def test_complete_lists_every_card_and_plays_the_same_games():
    rnd = random.Random(3)
    for _ in range(40):
        kingdom = rnd.sample(KINGDOM_CARDS, 10)
        s = rnd.choice([witch_big_money_strategy(kingdom), engine_strategy(kingdom)])
        full = complete(s)
        assert set(BUYABLE_CARDS) <= set(full.early_buy_priority)
        seed, opp = rnd.getrandbits(31), random_strategy(rnd, kingdom)
        assert play_one_c(s, opp, seed, kingdom)["trace"] == play_one_c(full, opp, seed, kingdom)["trace"]


def test_banned_cards_are_never_bought():
    s = complete(random_strategy(random.Random(1)), banned=set(NO_REQUEST_YET))
    m = mask(random_strategy(random.Random(2)), NO_REQUEST_YET)
    for lst in (s.early_buy_priority, s.mid_buy_priority, m.late_buy_priority):
        assert not set(lst) & set(NO_REQUEST_YET)


def test_plan_kingdoms_hold_the_key_and_bot_cards():
    plan = make_plan(["Gardens", "Workshop"], 5, 3, random.Random(1))
    for bot, k, seeds in plan:
        assert len(k) == 10 and {"Gardens", "Workshop"} <= set(k) and len(seeds) == 3
        assert bot != "Witch-BM" or "Witch" in k
    assert 0 <= score(witch_big_money_strategy(), plan[:3])["score"] <= 1


@pytest.fixture
def portfolio(tmp_path):
    members = {"bm-witch": witch_big_money_strategy(), "engine": engine_strategy()}
    (tmp_path / "members").mkdir()
    rows = []
    for name, s in members.items():
        from dataclasses import asdict
        (tmp_path / "members" / f"{name}.json").write_text(json.dumps(asdict(complete(s))))
        rows.append({"name": name, "file": f"members/{name}.json", "must": []})
    return {"protocol": 1, "name": "test-portfolio", "kind": KIND, "kingdom": list(KINGDOM_CARDS),
            "key_cards": {}, "members": rows, "dir": str(tmp_path)}


def test_choice_is_deterministic_and_plays_the_chosen_member(portfolio):
    k = ["Cellar", "Chapel", "Moat", "Village", "Smithy", "Militia", "Market", "Witch", "Mine", "Artisan"]
    a, b = choose(portfolio, k, n_seeds=10), choose(portfolio, list(reversed(k)), n_seeds=10)
    assert a == b and a["member"] in ("bm-witch", "engine")
    assert a["bots"] == ["Big Money", "Smithy-BM", "Witch-BM"]
    p = PortfolioPolicy(portfolio)
    assert p.phase(observation(kingdom=k)).startswith(p.chosen["member"] + ":")
    assert not check(p, k)
