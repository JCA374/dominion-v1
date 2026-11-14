"""GA tests: genome, operators, fitness, hall of fame, C bridge. Rules are tested in engine/."""

import random

from core.cards import (ALL_CARDS, BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS,
                        NONTERMINAL_ACTIONS, TERMINAL_ACTIONS, CardType, CARD_ID, CARD_NAME, NUM_CARDS)
from core.engine import play_game_2p
from core.strategy import (
    Strategy, Transitions, get_current_phase, random_strategy,
    big_money_strategy, engine_strategy, describe,
    save_best_model, load_strategy, get_action_priority,
    get_nonterminal_priority, get_terminal_priority,
    get_chapel_trash_priority,
)
from ga.fitness import evaluate_vs_opponent, make_seed_list, USE_C_ENGINE
from ga.ga import order_crossover, mutate, crossover, init_population

# The 12 attack-free cards the first engine knew (a small kingdom the short GA runs converge on)
BASE12 = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel", "Throne Room",
          "Council Room", "Moneylender", "Gardens", "Mine", "Merchant"]


def test_phase_selection():
    """Phase selection returns correct phase for edge cases."""
    t = Transitions(early_to_mid_turn=6, mid_to_late_provinces=4,
                    mid_to_late_turn=15)

    assert get_current_phase(1, 12, t) == "early"
    assert get_current_phase(6, 12, t) == "early"
    assert get_current_phase(7, 12, t) == "mid"
    assert get_current_phase(7, 5, t) == "mid"
    assert get_current_phase(7, 4, t) == "late"    # provinces trigger
    assert get_current_phase(7, 3, t) == "late"
    assert get_current_phase(7, 2, t) == "late"
    assert get_current_phase(7, 0, t) == "late"
    assert get_current_phase(20, 4, t) == "late"
    assert get_current_phase(20, 1, t) == "late"
    # Turn-based mid→late trigger (provinces still high but turn >= mid_to_late_turn)
    assert get_current_phase(14, 8, t) == "mid"    # turn 14 < 15, provinces high
    assert get_current_phase(15, 8, t) == "late"   # turn 15 >= 15, forces late
    assert get_current_phase(20, 8, t) == "late"   # turn 20 >= 15, provinces high


def test_random_strategies_valid():
    """100 random strategies have valid genomes."""
    rng = random.Random(123)
    buyable_set = set(BUYABLE_CARDS)
    action_set = set(ACTION_CARDS)

    for _ in range(100):
        s = random_strategy(rng)
        for lst_name in ["early_buy_priority", "mid_buy_priority", "late_buy_priority"]:
            lst = getattr(s, lst_name)
            non_pass = [c for c in lst if c != "PASS"]
            assert set(non_pass) == buyable_set, f"Invalid {lst_name}: {lst}"
            assert len(non_pass) == len(set(non_pass)), f"Duplicates in {lst_name}"
            assert lst.count("PASS") <= 1

        nt_set = {c for c in ACTION_CARDS if ALL_CARDS[c].actions > 0}
        t_set = {c for c in ACTION_CARDS if ALL_CARDS[c].actions == 0}
        assert set(s.nonterminal_priority) == nt_set, f"Invalid nonterminal_priority: {s.nonterminal_priority}"
        assert len(s.nonterminal_priority) == len(nt_set), f"Duplicates in nonterminal_priority"
        assert set(s.terminal_priority) == t_set, f"Invalid terminal_priority: {s.terminal_priority}"
        assert len(s.terminal_priority) == len(t_set), f"Duplicates in terminal_priority"
        assert 0 <= s.terminal_slack <= 3

        for ct_name in ["early_chapel_trash", "mid_chapel_trash", "late_chapel_trash"]:
            ct = getattr(s, ct_name)
            assert "STOP" in ct, f"STOP missing from {ct_name}"

        # Throne Room priority: all actions except Throne Room
        tr_set = action_set - {"Throne Room"}
        assert set(s.throne_room_priority) == tr_set
        assert len(s.throne_room_priority) == len(tr_set)

        # Mine trash priority: Copper and Silver
        assert set(s.mine_trash_priority) == {"Copper", "Silver"}
        assert len(s.mine_trash_priority) == 2

        assert 2 <= s.transitions.early_to_mid_turn <= 15
        assert 0 <= s.transitions.mid_to_late_provinces <= 8


def test_describe():
    """describe() returns expected substrings."""
    s = big_money_strategy()
    d = describe(s, 85.0)
    assert "85.0 VP" in d
    assert "Province" in d
    assert "EARLY" in d
    assert "MID" in d
    assert "LATE" in d


def test_ox_validity():
    """Order crossover produces valid permutations (1000 iterations)."""
    rng = random.Random(99)
    buyable_set = set(BUYABLE_CARDS)

    for _ in range(1000):
        s1 = random_strategy(rng)
        s2 = random_strategy(rng)
        child = order_crossover(s1.early_buy_priority, s2.early_buy_priority, rng)
        non_pass = [c for c in child if c != "PASS"]
        assert set(non_pass) == buyable_set, f"Missing cards in OX child"
        assert len(non_pass) == len(set(non_pass)), f"Duplicates in OX child"
        assert child.count("PASS") <= 1


def test_mutation_validity():
    """Mutation preserves valid permutations."""
    rng = random.Random(77)
    buyable_set = set(BUYABLE_CARDS)

    for _ in range(1000):
        s = random_strategy(rng)
        m = mutate(s, 0.3, rng)
        for lst_name in ["early_buy_priority", "mid_buy_priority", "late_buy_priority"]:
            lst = getattr(m, lst_name)
            non_pass = [c for c in lst if c != "PASS"]
            assert set(non_pass) == buyable_set
            assert len(non_pass) == len(set(non_pass))


def test_short_ga_run():
    """5-generation mini run completes without errors."""
    from ga.ga import run_ga

    result = run_ga({
        "pop_size": 10,
        "generations": 5,
        "games_per_eval": 10,
        "tournament_size": 3,
        "elite_count": 1,
        "mutation_rate": 0.1,
        "seed": 42,
        "kingdom": KINGDOM_CARDS,
        "csv_path": "/tmp/smoke_test_log.csv",
        "best_model_dir": "/tmp/test_short_ga_model",
    })

    assert len(result["log"]) == 5
    assert 0.0 <= result["best_fitness"] <= 1.0
    assert result["best_strategy"] is not None


def _make_strategy(**overrides) -> Strategy:
    """Create a Strategy with sensible defaults. Override any field via kwargs."""
    defaults = dict(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        nonterminal_priority=[c for c in ACTION_CARDS if ALL_CARDS[c].actions > 0],
        terminal_priority=[c for c in ACTION_CARDS if ALL_CARDS[c].actions == 0],
        early_chapel_trash=["Estate", "Copper", "STOP"],
        mid_chapel_trash=["Estate", "Copper", "STOP"],
        late_chapel_trash=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
        throne_room_priority=[c for c in ACTION_CARDS if c != "Throne Room"],
        mine_trash_priority=["Copper", "Silver"],
        chapel_max_trash=4,
        buy_targets={},
    )
    defaults.update(overrides)
    return Strategy(**defaults)


def _action_strategy(card_name):
    """Helper: strategy that plays a single action card."""
    prov_first = ["Province"] + [c for c in BUYABLE_CARDS if c != "Province"]
    if ALL_CARDS[card_name].actions > 0:
        nt, t = [card_name], []
    else:
        nt, t = [], [card_name]
    return _make_strategy(
        early_buy_priority=prov_first,
        mid_buy_priority=prov_first,
        late_buy_priority=prov_first,
        nonterminal_priority=nt, terminal_priority=t,
    )


def _chapel_strategy(trash_priority):
    """Helper: strategy that plays Chapel with given trash priority in all phases."""
    return _make_strategy(
        nonterminal_priority=[], terminal_priority=["Chapel"],
        early_chapel_trash=trash_priority,
        mid_chapel_trash=trash_priority,
        late_chapel_trash=trash_priority,
    )


def test_2p_game_shared_supply():
    """2-player game uses shared supply that both players deplete."""
    s = big_money_strategy()
    result = play_game_2p(s, s, 42)

    assert "vp1" in result
    assert "vp2" in result
    assert "turns" in result
    assert result["vp1"] > 0
    assert result["vp2"] > 0
    assert result["turns"] > 0


def test_2p_game_vp_sum_reasonable():
    """Total VP across both players should be reasonable."""
    s = big_money_strategy()
    result = play_game_2p(s, s, 42)
    total_vp = result["vp1"] + result["vp2"]
    # Both players start with 3 Estate (3 VP each), buy more during game
    assert total_vp >= 6


# ---------------------------------------------------------------------------
# Full turn sequence test
# ---------------------------------------------------------------------------

def test_ga_chapel_crossover_preserves_stop():
    """Crossover of two chapel lists keeps STOP marker."""
    p1 = ["Estate", "Copper", "STOP", "Duchy", "Silver"]
    p2 = ["STOP", "Silver", "Copper", "Estate", "Duchy"]

    rng = random.Random(42)
    for _ in range(20):
        child = order_crossover(p1, p2, rng)
        assert "STOP" in child, f"STOP missing from crossover result: {child}"


def test_ga_chapel_trash_structure():
    """Chapel trash always starts with Curse and ends with STOP."""
    base = _make_strategy(
        early_chapel_trash=["Curse", "Copper", "Estate", "STOP"],
        mid_chapel_trash=["Curse", "Estate", "Copper", "STOP"],
        late_chapel_trash=["Curse", "Estate", "STOP"],
    )

    rng = random.Random(42)
    for _ in range(50):
        mutated = mutate(base, rate=1.0, rng=rng)
        for phase in ["early", "mid", "late"]:
            trash = getattr(mutated, f"{phase}_chapel_trash")
            assert trash[0] == "Curse", f"{phase}: Curse must be first"
            assert trash[-1] == "STOP", f"{phase}: STOP must be last"
            assert "Curse" not in trash[1:-1], f"{phase}: no duplicate Curse"
        assert mutated.chapel_max_trash == 4


# ---------------------------------------------------------------------------
# Save / load best model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Special card tests: Throne Room, Council Room, Moneylender, Gardens, Mine, Merchant
# ---------------------------------------------------------------------------

def test_save_and_load_best_model(tmp_path):
    """Save best model to disk and load it back."""
    strategy = big_money_strategy()
    vs_bm = {"win_rate": 0.42, "tie_rate": 0.16, "loss_rate": 0.42,
             "mean_turns": 22.5, "num_games": 200}

    out = str(tmp_path / "best_model")
    save_best_model(strategy, vs_bm, output_dir=out)

    import os
    assert os.path.exists(os.path.join(out, "strategy.json"))
    assert os.path.exists(os.path.join(out, "summary.txt"))
    assert os.path.exists(os.path.join(out, "buy_heatmap.png"))

    # Load back and verify round-trip
    loaded = load_strategy(os.path.join(out, "strategy.json"))
    assert loaded.early_buy_priority == strategy.early_buy_priority
    assert loaded.mid_buy_priority == strategy.mid_buy_priority
    assert loaded.late_buy_priority == strategy.late_buy_priority
    assert loaded.nonterminal_priority == strategy.nonterminal_priority
    assert loaded.terminal_priority == strategy.terminal_priority
    assert loaded.terminal_slack == strategy.terminal_slack
    assert loaded.early_chapel_trash == strategy.early_chapel_trash
    assert loaded.mid_chapel_trash == strategy.mid_chapel_trash
    assert loaded.late_chapel_trash == strategy.late_chapel_trash
    assert loaded.transitions.early_to_mid_turn == strategy.transitions.early_to_mid_turn
    assert loaded.transitions.mid_to_late_provinces == strategy.transitions.mid_to_late_provinces

    # Summary text contains key info
    with open(os.path.join(out, "summary.txt")) as f:
        text = f.read()
    assert "BEST TACTIC SUMMARY" in text
    assert "42%" in text


# ---------------------------------------------------------------------------
# Terminal-density buy rule tests
# ---------------------------------------------------------------------------

def test_hall_of_fame_adds_member_when_threshold_met(tmp_path):
    """GA adds strategy to hall when win rate >= hall_add_threshold."""
    from unittest.mock import patch
    from ga.ga import run_ga

    def fake_evaluate_population_vs_hall(population, seed_list, kingdom,
                                         hall=None, workers=1):
        results = []
        for _ in population:
            # All strategies "win" at 60% — above 0.55 threshold
            results.append({
                "win_rate": 0.6, "tie_rate": 0.1, "loss_rate": 0.3,
                "mean_turns": 20.0, "mean_vp_margin": 2.0,
                "mean_actions_per_turn": 1.0, "avg_final_deck": {},
            })
        return results

    with patch("ga.ga.evaluate_population_vs_hall", fake_evaluate_population_vs_hall), \
         patch("ga.ga.save_best_model"):
        result = run_ga({
            "pop_size": 6,
            "generations": 5,
            "games_per_eval": 4,
            "tournament_size": 2,
            "elite_count": 1,
            "mutation_rate": 0.1,
            "seed": 42,
            "kingdom": KINGDOM_CARDS,
            "hall_add_threshold": 0.55,
            "hall_max_size": 6,
            "csv_path": str(tmp_path / "log.csv"),
            "best_model_dir": str(tmp_path / "model"),
        })

    # Hall should have grown beyond the initial Big Money member
    assert len(result["hall"]) >= 2


def test_hall_of_fame_respects_max_size(tmp_path):
    """Hall of fame does not exceed hall_max_size."""
    from unittest.mock import patch
    from ga.ga import run_ga

    def fake_evaluate_population_vs_hall(population, seed_list, kingdom,
                                         hall=None, workers=1):
        results = []
        for _ in population:
            results.append({
                "win_rate": 0.8, "tie_rate": 0.1, "loss_rate": 0.1,
                "mean_turns": 20.0, "mean_vp_margin": 5.0,
                "mean_actions_per_turn": 1.0, "avg_final_deck": {},
            })
        return results

    with patch("ga.ga.evaluate_population_vs_hall", fake_evaluate_population_vs_hall), \
         patch("ga.ga.save_best_model"):
        result = run_ga({
            "pop_size": 6,
            "generations": 10,
            "games_per_eval": 4,
            "tournament_size": 2,
            "elite_count": 1,
            "mutation_rate": 0.1,
            "seed": 42,
            "kingdom": KINGDOM_CARDS,
            "hall_add_threshold": 0.55,
            "hall_max_size": 3,
            "csv_path": str(tmp_path / "log.csv"),
            "best_model_dir": str(tmp_path / "model"),
        })

    # Hall should not exceed max size
    assert len(result["hall"]) <= 3


# ---------------------------------------------------------------------------
# Evaluation correctness tests
# ---------------------------------------------------------------------------

def test_eval_big_money_vs_itself_is_balanced():
    """Big Money vs itself should win ~50% (each side is the same)."""
    bm = big_money_strategy()
    rng = random.Random(99)
    seeds = make_seed_list(100, rng)
    result = evaluate_vs_opponent(bm, seeds, KINGDOM_CARDS, opponent=bm)
    # Should be close to 50% with some variance — allow 30%-70%
    assert 0.30 <= result["win_rate"] <= 0.70, (
        f"BM vs itself should be ~50%, got {result['win_rate']:.0%}")


def test_eval_counts_wins_correctly():
    """Verify win/tie/loss always sum to 1.0."""
    bm = big_money_strategy()
    rng = random.Random(42)
    seeds = make_seed_list(50, rng)
    result = evaluate_vs_opponent(bm, seeds, KINGDOM_CARDS, opponent=bm)
    total = result["win_rate"] + result["tie_rate"] + result["loss_rate"]
    assert abs(total - 1.0) < 1e-9, f"Win+tie+loss = {total}, expected 1.0"


def test_vp_margin_returned():
    """evaluate_vs_opponent returns mean_vp_margin as a float."""
    bm = big_money_strategy()
    rng = random.Random(42)
    seeds = make_seed_list(20, rng)
    result = evaluate_vs_opponent(bm, seeds, KINGDOM_CARDS, opponent=bm)
    assert "mean_vp_margin" in result
    assert isinstance(result["mean_vp_margin"], float)


def test_vp_margin_positive_when_winning():
    """A stronger strategy should have positive VP margin against a weaker one."""
    bm = big_money_strategy()
    do_nothing = _make_strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        mid_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        late_buy_priority=["PASS"] + BUYABLE_CARDS[:],
    )
    rng = random.Random(42)
    seeds = make_seed_list(50, rng)
    result = evaluate_vs_opponent(bm, seeds, KINGDOM_CARDS, opponent=do_nothing)
    assert result["mean_vp_margin"] > 0, (
        f"BM should have positive VP margin vs do-nothing, got {result['mean_vp_margin']:.1f}")


def test_big_money_beats_do_nothing_strategy():
    """A strategy that passes on all buys should lose to Big Money."""
    # PASS first = never buy anything
    do_nothing = _make_strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        mid_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        late_buy_priority=["PASS"] + BUYABLE_CARDS[:],
    )
    rng = random.Random(42)
    seeds = make_seed_list(50, rng)
    result = evaluate_vs_opponent(do_nothing, seeds, KINGDOM_CARDS, opponent=None)
    # Big Money should crush a do-nothing strategy
    assert result["win_rate"] < 0.05, (
        f"Do-nothing should lose to Big Money, but won {result['win_rate']:.0%}")


def test_big_money_wins_as_both_p1_and_p2():
    """Verify evaluate_vs_opponent counts wins correctly from both positions."""
    from core.engine import play_game_2p

    bm = big_money_strategy()
    do_nothing = _make_strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        mid_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        late_buy_priority=["PASS"] + BUYABLE_CARDS[:],
    )

    # Manual check: BM as P1 should win
    r1 = play_game_2p(bm, do_nothing, 42, KINGDOM_CARDS)
    assert r1["vp1"] > r1["vp2"], "BM as P1 should beat do-nothing"

    # Manual check: BM as P2 should also win
    r2 = play_game_2p(do_nothing, bm, 42, KINGDOM_CARDS)
    assert r2["vp2"] > r2["vp1"], "BM as P2 should beat do-nothing"


def test_buy_target_zero_blocks_purchase():
    """buy_targets={card: 0} should prevent the strategy from ever buying that card."""
    from core.engine import play_game_2p

    # Strategy that wants Province first but has buy_target=0 for Province
    pgs = ["Province", "Gold", "Silver"] + [
        c for c in BUYABLE_CARDS if c not in ["Province", "Gold", "Silver"]]
    blocked = _make_strategy(
        early_buy_priority=pgs, mid_buy_priority=pgs, late_buy_priority=pgs,
        buy_targets={"Province": 0},  # never buy Province!
    )
    rng = random.Random(42)
    seeds = make_seed_list(20, rng)
    result = evaluate_vs_opponent(blocked, seeds, KINGDOM_CARDS, opponent=None)
    # Can't buy Province → should lose to Big Money almost always
    assert result["win_rate"] < 0.15, (
        f"Province-blocked should lose to BM, but won {result['win_rate']:.0%}")


def test_ga_best_strategy_beats_big_money():
    """GA output must beat Big Money >= 50%."""
    from ga.ga import run_ga

    base_kingdom = BASE12
    result = run_ga({
        "pop_size": 20,
        "generations": 15,
        "games_per_eval": 30,
        "tournament_size": 3,
        "elite_count": 2,
        "mutation_rate": 0.15,
        "seed": 42,
        "kingdom": base_kingdom,
        "hall_add_threshold": 0.55,
        "hall_max_size": 4,
        "csv_path": "/tmp/test_ga_bm.csv",
        "best_model_dir": "/tmp/test_ga_bm_model",
    })

    rng = random.Random(99)
    seeds = make_seed_list(100, rng)
    vs = evaluate_vs_opponent(result["best_strategy"], seeds, base_kingdom,
                              opponent=None, need_deck=True)
    assert vs["win_rate"] >= 0.50, (
        f"GA best should beat Big Money >=50%, got {vs['win_rate']:.0%}")


def test_hall_of_fame_prevents_bm_drift():
    """GA with hall of fame (BM always in hall) must still beat Big Money."""
    from ga.ga import run_ga

    base_kingdom = BASE12
    result = run_ga({
        "pop_size": 20,
        "generations": 15,
        "games_per_eval": 30,
        "tournament_size": 3,
        "elite_count": 2,
        "mutation_rate": 0.15,
        "seed": 42,
        "kingdom": base_kingdom,
        "hall_add_threshold": 0.55,
        "hall_max_size": 4,
        "csv_path": "/tmp/test_ga_hall.csv",
        "best_model_dir": "/tmp/test_ga_hall_model",
    })

    # Big Money is always in the hall, so GA should maintain competitiveness
    rng = random.Random(99)
    seeds = make_seed_list(100, rng)
    vs = evaluate_vs_opponent(result["best_strategy"], seeds, base_kingdom,
                              opponent=None)
    assert vs["win_rate"] >= 0.40, (
        f"Hall-of-fame GA should still beat BM >=40%, got {vs['win_rate']:.0%}")




# ---------------------------------------------------------------------------
# Card ID mapping tests
# ---------------------------------------------------------------------------

def test_card_id_coverage():
    """Every card in ALL_CARDS has an integer ID and vice versa."""
    for name in ALL_CARDS:
        assert name in CARD_ID, f"Card {name} missing from CARD_ID"
    for cid in range(NUM_CARDS):
        assert cid in CARD_NAME, f"Card ID {cid} missing from CARD_NAME"
        assert CARD_NAME[cid] in ALL_CARDS, f"CARD_NAME[{cid}]={CARD_NAME[cid]} not in ALL_CARDS"


def test_c_engine_available():
    """C engine (dominion.so) loads successfully."""
    assert USE_C_ENGINE, "C engine not available — dominion.so may not be built"


def test_c_engine_big_money_vs_self_balanced():
    """Big Money vs itself via C engine should win ~50%."""
    from ga.c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    seeds = make_seed_list(200, random.Random(42))
    result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)
    assert 0.30 <= result["win_rate"] <= 0.70, (
        f"BM vs itself should be ~50%, got {result['win_rate']:.0%}")


def test_c_engine_win_loss_tie_sum():
    """C engine results: win + tie + loss = 1.0."""
    from ga.c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    seeds = make_seed_list(50, random.Random(42))
    result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)
    total = result["win_rate"] + result["tie_rate"] + result["loss_rate"]
    assert abs(total - 1.0) < 1e-9, f"Win+tie+loss = {total}"


def test_c_engine_stronger_strategy_wins():
    """C engine: Big Money should beat a do-nothing strategy."""
    from ga.c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    do_nothing = _make_strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        mid_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        late_buy_priority=["PASS"] + BUYABLE_CARDS[:],
    )
    seeds = make_seed_list(50, random.Random(42))
    result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=do_nothing)
    assert result["win_rate"] >= 0.90, (
        f"BM should crush do-nothing, got {result['win_rate']:.0%}")
    assert result["mean_vp_margin"] > 0


def test_c_engine_random_strategies_complete():
    """C engine handles 20 random strategies without crashing."""
    from ga.c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    seeds = make_seed_list(20, random.Random(42))
    for i in range(20):
        strat = random_strategy(random.Random(i * 7))
        result = evaluate_vs_opponent_c(strat, seeds, KINGDOM_CARDS, opponent=bm)
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["mean_turns"] > 0


def test_c_engine_consistent_with_python():
    """C and Python engines produce statistically similar results."""
    bm = big_money_strategy()
    seeds = make_seed_list(200, random.Random(99))

    py_result = evaluate_vs_opponent(bm, seeds, KINGDOM_CARDS, opponent=bm,
                                     need_deck=True)

    from ga.c_bridge import evaluate_vs_opponent_c
    c_result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)

    # Both should be roughly balanced (different RNGs, so not exact)
    assert abs(py_result["win_rate"] - c_result["win_rate"]) < 0.15, (
        f"Win rates differ too much: Python={py_result['win_rate']:.2f} "
        f"C={c_result['win_rate']:.2f}")
    assert abs(py_result["mean_turns"] - c_result["mean_turns"]) < 5, (
        f"Turn counts differ too much: Python={py_result['mean_turns']:.1f} "
        f"C={c_result['mean_turns']:.1f}")


def test_c_engine_matches_python_with_attacks():
    """Lopsided matchups using the highest card ids (Witch, Moat) agree between engines.

    A BM mirror is always ~50% in both engines, so it can't catch a broken card.
    This did: the C supply array was 20 long while Witch/Moat have ids 20/21,
    so their piles were overwritten and C bought ~25% fewer Witches.
    """
    from core.strategy import _single_card_big_money, witch_big_money_strategy
    from ga.c_bridge import evaluate_vs_opponent_c

    seeds = make_seed_list(300, random.Random(5))
    cases = [
        (["Witch"], witch_big_money_strategy(["Witch"]), big_money_strategy(["Witch"])),
        (["Witch", "Moat"], _single_card_big_money("Moat", 2, ["Witch", "Moat"]),
         witch_big_money_strategy(["Witch", "Moat"])),
    ]
    for kingdom, strat, opp in cases:
        py = evaluate_vs_opponent(strat, seeds, kingdom, opponent=opp, need_deck=True)
        c = evaluate_vs_opponent_c(strat, seeds, kingdom, opponent=opp)
        assert abs(py["win_rate"] - c["win_rate"]) < 0.07, (
            f"{kingdom}: Python {py['win_rate']:.2f} vs C {c['win_rate']:.2f}")
        assert abs(py["mean_turns"] - c["mean_turns"]) < 1.0, (
            f"{kingdom}: Python {py['mean_turns']:.1f} turns vs C {c['mean_turns']:.1f}")


# ---------------------------------------------------------------------------
# Attack cards: Witch, Militia, Moat, Curse
# ---------------------------------------------------------------------------

def test_2p_game_with_attacks():
    """Full 2-player game with attack cards in kingdom completes."""
    kingdom = ["Village", "Smithy", "Market", "Laboratory", "Festival",
               "Chapel", "Throne Room", "Militia", "Witch", "Moat"]
    s1 = big_money_strategy(kingdom)
    s2 = engine_strategy(kingdom)
    result = play_game_2p(s1, s2, rng_seed=42, kingdom=kingdom)
    assert result["turns"] > 0
    assert result["turns"] <= 40


def test_new_cards_in_card_ids():
    """New cards have correct IDs and are in all lookup tables."""
    from core.cards import CARD_ID, CARD_NAME
    assert CARD_ID["Curse"] == 18
    assert CARD_ID["Militia"] == 19
    assert CARD_ID["Witch"] == 20
    assert CARD_ID["Moat"] == 21
    assert CARD_NAME[18] == "Curse"
    assert CARD_NAME[19] == "Militia"
    assert CARD_NAME[20] == "Witch"
    assert CARD_NAME[21] == "Moat"


def test_militia_in_buy_priority():
    """Militia appears in random strategy buy priorities."""
    rng = random.Random(42)
    kingdom = KINGDOM_CARDS
    strat = random_strategy(rng, kingdom)
    all_buyable = strat.early_buy_priority + strat.mid_buy_priority + strat.late_buy_priority
    assert "Militia" in all_buyable


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS: {name}")
    print("\nAll smoke tests passed.")

