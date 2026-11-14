"""The player <-> model protocol (policy/): registry, GA policy, pipe transport."""
import json
import os
import sys
from collections import Counter

import pytest

from policy import PROTOCOL_VERSION, list_models, load_policy, observation
from policy.check import check, scenarios
from policy.registry import MODELS_DIR, default_model, read_manifest
from policy.release import release

K1 = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel",
      "Throne Room", "Mine", "Moneylender", "Merchant"]


def test_default_model_exists():
    assert default_model() is not None
    assert load_policy(None).name == default_model()


@pytest.mark.parametrize("name", [m["name"] for m in list_models()])
def test_released_models_conform(name):
    policy = load_policy(name)
    assert policy.info()["protocol"] == PROTOCOL_VERSION
    assert policy.kingdom, "manifest must record the training kingdom"
    assert check(policy) == []


def test_ga_policy_answers_like_the_agent_in_training():
    """Mid-game, the protocol answer equals the GA agent's own engine decision."""
    import core  # noqa: F401
    from core.agent import GAAgent
    from dominion_engine import cards as K, game as G, play

    policy = load_policy("k1-steam")
    checked = []

    class Checker(GAAgent):
        def decide(self, g, d):
            ans = super().decide(g, d)
            p = g.players[d.player]
            if d.kind == G.BUY and d.player == g.current:
                obs = observation(
                    turn=g.turn_number(d.player), provinces_left=g.provinces_left, kingdom=K1,
                    empty_piles=[K.NAME[c] for c in range(K.NUM_CARDS) if g.supply[c] == 0],
                    owned={K.NAME[c]: n for c, n in Counter(p.all_cards()).items()},
                    hand=[K.NAME[c] for c in p.hand], actions=g.actions, buys=g.buys,
                    coins=g.coins)
                got = policy.buy(obs)
                assert (got[0] if got else None) == (K.NAME[ans] if ans >= 0 else None)
                checked.append(1)
            return ans

    for seed in range(30):
        play(K1, [Checker(policy.strategy), GAAgent(policy.strategy)], seed)
    assert len(checked) > 300


def test_unknown_cards_are_never_bought_and_discarded_first():
    policy = load_policy("k1-steam")
    obs = observation(kingdom=K1[:9] + ["Torturer"], coins=5, hand=["Torturer", "Copper", "Gold",
                                                                      "Silver", "Estate"])
    assert "Torturer" not in policy.buy(obs)
    assert policy.discard(obs, keep=3)[0] == "Torturer"


def test_every_card_can_be_bought_since_protocol_2():
    from core.strategy import big_money_strategy
    from policy.ga_policy import GAPolicy
    s = big_money_strategy()
    s.early_buy_priority = ["Cellar", "Remodel", "Silver", "PASS"]
    policy = GAPolicy({"name": "t", "kingdom": ["Cellar", "Remodel"]}, s)
    assert policy.buy(observation(kingdom=["Cellar", "Remodel"], coins=4)) == ["Cellar"]


def test_handle_reports_errors_instead_of_raising():
    policy = load_policy("k1-steam")
    assert "error" in policy.handle({"type": "nonsense", "obs": {}})
    assert "error" in policy.handle({"type": "buy", "obs": {"bogus_field": 1}})


def test_bare_strategy_json_loads(tmp_path):
    src = os.path.join(MODELS_DIR, "k1-steam", "strategy.json")
    dst = tmp_path / "run" / "strategy.json"
    dst.parent.mkdir()
    dst.write_text(open(src).read())
    m = read_manifest(str(dst))
    assert m["kind"] == "ga-strategy" and sorted(K1) == m["kingdom"]
    assert load_policy(str(dst)).buy(observation(coins=8, kingdom=K1))


def test_newer_protocol_is_refused(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps(
        {"protocol": PROTOCOL_VERSION + 1, "name": "future", "kind": "ga-strategy"}))
    with pytest.raises(ValueError):
        load_policy(str(tmp_path))


def test_release_is_immutable(tmp_path, monkeypatch):
    import policy.registry as reg
    import policy.release as rel
    monkeypatch.setattr(rel, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(rel, "DEFAULT_FILE", str(tmp_path / "DEFAULT"))
    monkeypatch.setattr(reg, "MODELS_DIR", str(tmp_path))
    src = os.path.join(MODELS_DIR, "k1-steam")
    dst = release(src, "test-model", "x", make_default=True)
    assert json.load(open(os.path.join(dst, "model.json")))["kingdom"] == sorted(K1)
    assert (tmp_path / "DEFAULT").read_text().strip() == "test-model"
    with pytest.raises(FileExistsError):
        release(src, "test-model")


def test_subprocess_policy_matches_in_process(tmp_path):
    """A 'subprocess' model (here: this repo's own server) answers exactly like in-process."""
    root = os.path.dirname(MODELS_DIR)
    (tmp_path / "model.json").write_text(json.dumps({
        "protocol": PROTOCOL_VERSION, "name": "piped", "kind": "subprocess",
        "command": [sys.executable, "-m", "policy.serve", "k1-steam"], "cwd": root}))
    remote = load_policy(str(tmp_path))
    local = load_policy("k1-steam")
    try:
        assert remote.kingdom == local.kingdom
        for req in scenarios(K1, n=15):
            if req["type"] == "info":
                continue
            assert remote.handle(req) == local.handle(req)
        assert check(remote) == []
    finally:
        remote.close()


def test_bench_reference_rows():
    """policy.bench: Big Money scores 50% against itself and loses to Smithy-BM and Witch-BM."""
    from core.strategy import big_money_strategy
    from policy.bench import bench, steam_kingdoms

    r = bench(big_money_strategy(), None, n_kingdoms=10, n_seeds=10)
    assert r["Big Money"]["steam"]["score"] == 0.5
    assert r["Smithy-BM"]["steam"]["score"] < 0.4
    assert r["Witch-BM"]["steam"]["score"] < 0.3
    assert all("Witch" in k for k in steam_kingdoms(20, 1, must="Witch"))
    assert steam_kingdoms(5, 1) == steam_kingdoms(5, 1)   # same kingdoms every run
