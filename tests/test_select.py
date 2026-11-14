"""Model plug-in folders, key cards, the benchmark store and the model selector."""
import json
import os
import shutil

from policy import list_models, load_policy, rank_models, select_model
from policy.bench import main as bench_main
from policy.registry import MODELS_DIR, key_cards, read_bench, read_manifest
from policy.selector import best_kingdom

K1 = ["Chapel", "Festival", "Laboratory", "Market", "Merchant", "Mine", "Moneylender",
      "Smithy", "Throne Room", "Village"]
OTHER = ["Cellar", "Moat", "Village", "Smithy", "Militia", "Market", "Witch", "Mine",
         "Library", "Merchant"]


def fake(tmp_path, name, kingdom, keys, score=None, retired=False):
    d = tmp_path / name
    d.mkdir()
    m = {"protocol": 1, "name": name, "kind": "ga-strategy", "file": "strategy.json",
         "kingdom": kingdom, "key_cards": keys, "dir": str(d)}
    if retired:
        m["retired"] = True
    if score is not None:
        (d / "bench.json").write_text(json.dumps({"score": score, "opponents": {}}))
    return m


def test_exact_kingdom_beats_a_better_model(tmp_path):
    models = [fake(tmp_path, "exact", K1, {"Laboratory": 3}, 0.40),
              fake(tmp_path, "strong", ["Smithy"], {"Smithy": 1}, 0.90)]
    assert select_model(list(reversed(K1)), models)["name"] == "exact"


def test_key_cards_decide_eligibility_then_score(tmp_path):
    models = [fake(tmp_path, "witch", [], {"Witch": 2, "Market": 1}, 0.80),
              fake(tmp_path, "smithy", [], {"Smithy": 2}, 0.50),
              fake(tmp_path, "lab", [], {"Laboratory": 2}, 0.95)]      # Laboratory not dealt
    ranked = rank_models(OTHER, models)
    assert [r["name"] for r in ranked] == ["witch", "smithy", "lab"]
    assert ranked[2]["missing"] == ["Laboratory"]


def test_no_eligible_model_still_picks_a_real_model(tmp_path):
    models = [fake(tmp_path, "a", [], {"Gardens": 1, "Witch": 1}, 0.30),
              fake(tmp_path, "b", [], {"Laboratory": 2}, 0.90)]
    pick = select_model(OTHER, models)
    assert pick["name"] == "a"            # half its key cards are dealt; b has none
    assert pick["name"] != "big-money"


def test_retired_models_are_never_chosen(tmp_path):
    models = [fake(tmp_path, "old", OTHER, {"Witch": 1}, 0.99, retired=True),
              fake(tmp_path, "new", [], {"Smithy": 1}, 0.10)]
    assert select_model(OTHER, models)["name"] == "new"


def test_unbenchmarked_ranks_below_benchmarked(tmp_path):
    models = [fake(tmp_path, "nobench", [], {"Smithy": 1}),
              fake(tmp_path, "bench", [], {"Smithy": 1}, 0.2)]
    assert select_model(OTHER, models)["name"] == "bench"


def test_best_kingdom_is_a_training_kingdom():
    m, k = best_kingdom()
    assert not m.get("retired") and k == m["kingdom"]


def test_released_models_have_key_cards():
    for m in list_models():
        assert key_cards(m) is not None, m["name"]


def test_plugin_folder_and_bench_store(tmp_path, monkeypatch):
    """A model folder dropped into a DOMINION_MODELS folder is found, played, benchmarked
    and selected, with no code change."""
    plug = tmp_path / "plugins"
    shutil.copytree(os.path.join(MODELS_DIR, "k1-steam"), plug / "my-model")
    data = json.loads((plug / "my-model" / "model.json").read_text())
    data["name"] = "my-model"
    (plug / "my-model" / "model.json").write_text(json.dumps(data))
    monkeypatch.setenv("DOMINION_MODELS", str(plug))

    assert "my-model" in [m["name"] for m in list_models()]
    assert load_policy("my-model").name == "my-model"
    assert bench_main(["my-model", "--store", "--kingdoms", "3", "--seeds", "2"]) == 0
    bench = read_bench(read_manifest(str(plug / "my-model")))
    assert bench["model"] == "my-model" and 0 <= bench["score"] <= 1
    assert bench["key_cards"] == data["key_cards"]
