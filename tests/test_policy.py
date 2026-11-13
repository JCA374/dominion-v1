"""The player <-> model protocol (policy/): registry, GA policy, pipe transport."""
import json
import os
import sys

import pytest

from core.engine import GameState, play_buy_phase
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


def test_ga_buy_matches_training_engine():
    policy = load_policy("k1-steam")
    for req in scenarios(K1, n=40):
        if req["type"] != "buy":
            continue
        obs = req["obs"]
        supply = {c: 10 for c in ["Copper", "Silver", "Gold", "Estate", "Duchy", "Curse"] + K1}
        supply["Province"] = obs["provinces_left"]
        for c in obs["empty_piles"]:
            supply[c] = 0
        st = GameState(supply=supply, coins=obs["coins"], buys=obs["buys"], turn=obs["turn"],
                       deck=[c for c, n in obs["owned"].items() for _ in range(n)])
        play_buy_phase(st, policy.strategy)
        assert policy.buy(obs) == st.discard


def test_unknown_cards_are_never_bought_and_discarded_first():
    policy = load_policy("k1-steam")
    obs = observation(kingdom=K1[:9] + ["Bandit"], coins=5, hand=["Bandit", "Copper", "Gold",
                                                                    "Silver", "Estate"])
    assert "Bandit" not in policy.buy(obs)
    assert policy.discard(obs, keep=3)[0] == "Bandit"


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
