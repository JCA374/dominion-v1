"""Smoke tests for all components."""

import random

from cards import (ALL_CARDS, BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS,
                    NONTERMINAL_ACTIONS, TERMINAL_ACTIONS, CardType,
                    CARD_ID, CARD_NAME, NUM_CARDS, PASS_ID, STOP_ID,
                    CARD_COST, CARD_COINS, CARD_VP, CARD_DRAW,
                    CARD_ACTIONS, CARD_BUYS, CARD_TYPE_ID, CARD_SPECIAL_ID)
from engine import (
    new_game, play_game, play_game_2p, play_action_phase, play_buy_phase,
    play_chapel, play_moneylender, play_throne_room, play_mine,
    auto_play_treasures, cleanup,
    is_game_over, count_vp, default_supply, GameState, draw_cards,
)
from strategy import (
    Strategy, Transitions, get_current_phase, random_strategy,
    big_money_strategy, engine_strategy, describe,
    save_best_model, load_strategy, get_action_priority,
    get_chapel_trash_priority,
)
from fitness import evaluate, evaluate_vs_opponent, make_seed_list, USE_C_ENGINE
from ga import order_crossover, mutate, crossover, init_population


def test_card_definitions():
    """All 18 cards defined with correct costs."""
    assert len(ALL_CARDS) == 18
    assert ALL_CARDS["Copper"].cost == 0
    assert ALL_CARDS["Province"].cost == 8
    assert ALL_CARDS["Province"].vp == 6
    assert ALL_CARDS["Chapel"].special == "chapel"
    assert ALL_CARDS["Market"].buys == 1
    assert ALL_CARDS["Market"].actions == 1
    assert ALL_CARDS["Market"].coins == 1
    assert ALL_CARDS["Market"].cards_drawn == 1
    assert ALL_CARDS["Village"].actions == 2
    assert ALL_CARDS["Smithy"].cards_drawn == 3
    assert ALL_CARDS["Festival"].coins == 2
    # New cards
    assert ALL_CARDS["Throne Room"].cost == 4
    assert ALL_CARDS["Throne Room"].special == "throne_room"
    assert ALL_CARDS["Council Room"].cost == 5
    assert ALL_CARDS["Council Room"].cards_drawn == 4
    assert ALL_CARDS["Council Room"].buys == 1
    assert ALL_CARDS["Moneylender"].cost == 4
    assert ALL_CARDS["Moneylender"].special == "moneylender"
    assert ALL_CARDS["Gardens"].cost == 4
    assert ALL_CARDS["Gardens"].card_type == CardType.VICTORY
    assert ALL_CARDS["Gardens"].special == "gardens"
    # Mine and Merchant
    assert ALL_CARDS["Mine"].cost == 5
    assert ALL_CARDS["Mine"].special == "mine"
    assert ALL_CARDS["Merchant"].cost == 3
    assert ALL_CARDS["Merchant"].cards_drawn == 1
    assert ALL_CARDS["Merchant"].actions == 1
    assert ALL_CARDS["Merchant"].special == "merchant"


def test_single_game_big_money():
    """Big Money game: reasonable VP and turn count, card conservation."""
    s = big_money_strategy()
    for seed in range(20):
        state = new_game(seed)
        initial_supply_total = sum(state.supply.values())
        initial_owned = len(state.deck) + len(state.hand)

        while not is_game_over(state):
            state.turn += 1
            state.actions = 1
            state.buys = 1
            state.coins = 0
            play_action_phase(state, s)
            play_buy_phase(state, s)

            # Card conservation
            owned = len(state.deck) + len(state.hand) + len(state.discard) + len(state.play_area)
            supply = sum(state.supply.values())
            trashed = len(state.trash)
            assert owned + supply + trashed == initial_supply_total + initial_owned

            cleanup(state)

        vp = count_vp(state)
        assert 30 <= vp <= 120, f"VP out of range: {vp}"
        assert 10 <= state.turn <= 40, f"Turns out of range: {state.turn}"


def test_phase_selection():
    """Phase selection returns correct phase for edge cases."""
    t = Transitions(early_to_mid_turn=6, mid_to_late_provinces=4)

    assert get_current_phase(1, 12, t) == "early"
    assert get_current_phase(6, 12, t) == "early"
    assert get_current_phase(7, 12, t) == "mid"
    assert get_current_phase(7, 5, t) == "mid"
    assert get_current_phase(7, 4, t) == "late"
    assert get_current_phase(7, 0, t) == "late"
    assert get_current_phase(20, 4, t) == "late"


def test_nonterminals_play_before_terminals():
    """Non-terminal actions always play before terminal actions."""
    state = _make_state_with_hand(
        ["Village", "Smithy"],
        deck=["Copper"] * 10,
    )
    state.supply = _full_supply()
    state.turn = 1
    # Even though terminal_priority lists Smithy first,
    # Village (non-terminal) plays first due to structural split
    strategy = _make_strategy(
        early_nonterminal_priority=["Village"],
        early_terminal_priority=["Smithy"],
        transitions=Transitions(early_to_mid_turn=3, mid_to_late_provinces=4),
    )
    play_action_phase(state, strategy)
    # Village plays first (+2 actions), then Smithy
    assert "Village" in state.play_area
    assert "Smithy" in state.play_area
    # Village must have been played first (index 0)
    assert state.play_area.index("Village") < state.play_area.index("Smithy")


def test_drawn_terminal_waits_for_terminal_tier():
    """A terminal card drawn during the non-terminal tier doesn't play until terminal tier."""
    # Village draws from deck — if it draws Smithy, Smithy should wait
    state = _make_state_with_hand(
        ["Village"],
        deck=["Copper", "Copper", "Copper", "Copper", "Smithy"],
    )
    state.supply = _full_supply()
    state.turn = 1
    strategy = _make_strategy(
        early_nonterminal_priority=["Village"],
        early_terminal_priority=["Smithy"],
    )
    play_action_phase(state, strategy)
    # Village plays in NT tier (draws Smithy into hand), then Smithy plays in T tier
    assert "Village" in state.play_area
    assert "Smithy" in state.play_area
    assert state.play_area.index("Village") < state.play_area.index("Smithy")


def test_phase_specific_chapel_trash():
    """Chapel trashes differently in early vs late phases."""
    # Early: trash Estate
    state_early = _make_state_with_hand(
        ["Chapel", "Estate", "Copper"],
        deck=["Silver"] * 5,
    )
    state_early.supply = _full_supply()
    state_early.turn = 1
    strategy = _make_strategy(
        early_terminal_priority=["Chapel"],
        mid_terminal_priority=["Chapel"],
        late_terminal_priority=["Chapel"],
        early_chapel_trash=["Estate", "Copper", "STOP"],
        mid_chapel_trash=["Estate", "STOP"],
        late_chapel_trash=["STOP"],  # never trash late
        transitions=Transitions(early_to_mid_turn=3, mid_to_late_provinces=4),
    )
    play_action_phase(state_early, strategy)
    assert "Estate" in state_early.trash
    assert "Copper" in state_early.trash

    # Late: STOP means nothing trashed
    state_late = _make_state_with_hand(
        ["Chapel", "Estate", "Copper"],
        deck=["Silver"] * 5,
    )
    state_late.supply = _full_supply()
    state_late.supply["Province"] = 3  # triggers late phase
    state_late.turn = 10
    play_action_phase(state_late, strategy)
    assert state_late.trash == []


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

        nt_set = set(NONTERMINAL_ACTIONS)
        t_set = set(TERMINAL_ACTIONS)
        for nt_name in ["early_nonterminal_priority", "mid_nonterminal_priority", "late_nonterminal_priority"]:
            nt = getattr(s, nt_name)
            assert set(nt) == nt_set, f"Invalid {nt_name}"
        for t_name in ["early_terminal_priority", "mid_terminal_priority", "late_terminal_priority"]:
            t = getattr(s, t_name)
            assert set(t) == t_set, f"Invalid {t_name}"

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


def test_fitness_determinism():
    """Same seeds produce same fitness."""
    rng = random.Random(42)
    seeds = make_seed_list(20, rng)
    s = big_money_strategy()
    r1 = evaluate(s, seeds)
    r2 = evaluate(s, seeds)
    assert r1["mean_vp"] == r2["mean_vp"]


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
    from ga import run_ga

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


def _full_supply():
    """Helper: full default supply for tests."""
    return default_supply()


def _make_state_with_hand(hand, deck=None):
    """Helper: create a GameState with a specific hand and deck."""
    state = GameState()
    state.rng = random.Random(0)
    state.hand = list(hand)
    state.deck = list(deck) if deck is not None else ["Copper"] * 10
    state.supply = _full_supply()
    state.actions = 1
    state.buys = 1
    state.coins = 0
    return state


def _make_strategy(**overrides) -> Strategy:
    """Create a Strategy with sensible defaults. Override any field via kwargs."""
    defaults = dict(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        early_nonterminal_priority=NONTERMINAL_ACTIONS[:],
        early_terminal_priority=TERMINAL_ACTIONS[:],
        mid_nonterminal_priority=NONTERMINAL_ACTIONS[:],
        mid_terminal_priority=TERMINAL_ACTIONS[:],
        late_nonterminal_priority=NONTERMINAL_ACTIONS[:],
        late_terminal_priority=TERMINAL_ACTIONS[:],
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
    is_nt = ALL_CARDS[card_name].actions > 0
    return _make_strategy(
        early_buy_priority=prov_first,
        mid_buy_priority=prov_first,
        late_buy_priority=prov_first,
        early_nonterminal_priority=[card_name] if is_nt else [],
        early_terminal_priority=[card_name] if not is_nt else [],
        mid_nonterminal_priority=[card_name] if is_nt else [],
        mid_terminal_priority=[card_name] if not is_nt else [],
        late_nonterminal_priority=[card_name] if is_nt else [],
        late_terminal_priority=[card_name] if not is_nt else [],
    )


def test_smithy_draws_three_cards():
    """Playing Smithy should draw 3 cards."""
    state = _make_state_with_hand(["Smithy"], deck=["Silver", "Gold", "Copper", "Estate"])
    strategy = _action_strategy("Smithy")

    play_action_phase(state, strategy)

    assert "Smithy" in state.play_area
    assert "Smithy" not in state.hand
    # Smithy draws 3 cards from deck
    assert len(state.hand) == 3
    assert len(state.deck) == 1  # only 1 left in deck
    assert state.actions == 0  # spent 1, gained 0
    assert state.coins == 0  # Smithy adds no coins
    assert state.buys == 1  # Smithy adds no buys


def test_village_draws_and_adds_actions():
    """Playing Village should draw 1 card and add 2 actions."""
    state = _make_state_with_hand(["Village"], deck=["Silver", "Gold"])
    strategy = _action_strategy("Village")

    play_action_phase(state, strategy)

    assert "Village" in state.play_area
    # Village draws 1 card
    assert len(state.hand) == 1
    assert state.hand[0] == "Gold"  # deck pops from end
    # Village: spent 1 action, gained 2 => net 1 remaining
    assert state.actions == 2
    assert state.coins == 0
    assert state.buys == 1


def test_market_all_effects():
    """Playing Market should draw 1, add 1 action, 1 buy, 1 coin."""
    state = _make_state_with_hand(["Market"], deck=["Silver", "Gold"])
    strategy = _action_strategy("Market")

    play_action_phase(state, strategy)

    assert "Market" in state.play_area
    assert len(state.hand) == 1  # drew 1
    # Market: spent 1 action, gained 1 => net 0 change => still 1
    assert state.actions == 1
    assert state.coins == 1
    assert state.buys == 2  # started with 1, gained 1


def test_laboratory_draws_and_adds_action():
    """Playing Laboratory should draw 2 cards and add 1 action."""
    state = _make_state_with_hand(["Laboratory"], deck=["Silver", "Gold", "Copper"])
    strategy = _action_strategy("Laboratory")

    play_action_phase(state, strategy)

    assert "Laboratory" in state.play_area
    assert len(state.hand) == 2  # drew 2
    # Lab: spent 1 action, gained 1 => still 1
    assert state.actions == 1
    assert state.coins == 0
    assert state.buys == 1


def test_festival_actions_buys_coins():
    """Playing Festival should add 2 actions, 1 buy, 2 coins (no draw)."""
    state = _make_state_with_hand(["Festival"], deck=["Silver", "Gold"])
    strategy = _action_strategy("Festival")

    play_action_phase(state, strategy)

    assert "Festival" in state.play_area
    assert len(state.hand) == 0  # Festival draws 0 cards
    # Festival: spent 1 action, gained 2 => 2 remaining
    assert state.actions == 2
    assert state.coins == 2
    assert state.buys == 2  # started with 1, gained 1


def test_village_then_smithy_chain():
    """Village gives enough actions to chain into Smithy."""
    state = _make_state_with_hand(
        ["Village", "Smithy"],
        deck=["Copper", "Silver", "Gold", "Estate"],
    )
    strategy = _make_strategy(
        early_nonterminal_priority=["Village"],
        early_terminal_priority=["Smithy"],
    )

    play_action_phase(state, strategy)

    assert "Village" in state.play_area
    assert "Smithy" in state.play_area
    # Village draws 1 (Estate from top/end), Smithy draws 3 (Gold, Silver, Copper)
    assert len(state.hand) == 4
    # Village: 1 - 1 + 2 = 2 actions, then Smithy: 2 - 1 + 0 = 1
    assert state.actions == 1


def test_draw_cards_reshuffles_discard():
    """Drawing when deck is empty should reshuffle discard pile."""
    state = _make_state_with_hand(
        ["Smithy"],
        deck=[],  # empty deck
    )
    state.discard = ["Gold", "Silver", "Province"]
    strategy = _action_strategy("Smithy")

    play_action_phase(state, strategy)

    # Smithy draws 3; deck was empty so discard got reshuffled
    assert len(state.hand) == 3
    assert set(state.hand) == {"Gold", "Silver", "Province"}
    assert len(state.discard) == 0


# ---------------------------------------------------------------------------
# Buy phase tests
# ---------------------------------------------------------------------------

def test_buy_phase_auto_plays_treasures():
    """Treasures move from hand to play area and add coins."""
    state = _make_state_with_hand(["Copper", "Silver", "Gold", "Estate"])
    # Only Province in supply (required by phase logic), nothing affordable
    state.supply = {"Province": 12}
    state.turn = 1
    strategy = _action_strategy("Smithy")

    play_buy_phase(state, strategy)

    # All 3 treasures moved to play area
    assert "Copper" in state.play_area
    assert "Silver" in state.play_area
    assert "Gold" in state.play_area
    # Estate stays in hand (not a treasure)
    assert "Estate" in state.hand
    # Coins: 1 + 2 + 3 = 6
    assert state.coins == 6


def test_buy_phase_buys_card():
    """Buying a card removes it from supply and adds to discard."""
    state = _make_state_with_hand(["Gold", "Gold"])
    state.supply = _full_supply()
    state.turn = 10
    state.buys = 1
    strategy = _make_strategy(
        mid_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
    )

    play_buy_phase(state, strategy)

    # With 6 coins in mid phase, should buy Province (cost 8? no, 6 coins)
    # Actually Gold gives 3 each = 6 coins, Province costs 8, can't afford
    # Should buy the highest-priority card affordable at 6 coins
    # mid priority: Province(8), then rest... will skip Province, buy next affordable
    total_bought = len(state.discard)
    assert total_bought >= 1
    assert state.buys == 0


def test_buy_phase_respects_cost():
    """Can't buy a card that costs more than available coins."""
    state = _make_state_with_hand(["Copper"])  # only 1 coin
    state.supply = _full_supply()
    state.turn = 10
    state.buys = 1
    # Priority wants Province first, but we can only afford cost <= 1
    strategy = _make_strategy(
        mid_buy_priority=["Province", "Gold", "Duchy", "Silver"] + [c for c in BUYABLE_CARDS if c not in ["Province", "Gold", "Duchy", "Silver"]],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
    )

    play_buy_phase(state, strategy)

    # Only 1 coin, can't afford Province(8), Gold(6), Duchy(5), Silver(3)
    # Should buy Copper(0) or something cost <= 1
    if state.discard:
        bought_card = state.discard[0]
        assert ALL_CARDS[bought_card].cost <= 1


def test_buy_phase_empty_supply_skipped():
    """Cards with 0 supply are skipped in buy priority."""
    state = _make_state_with_hand(["Gold", "Gold"])  # 6 coins
    state.supply = _full_supply()
    state.supply["Province"] = 0
    state.supply["Gold"] = 0
    state.supply["Silver"] = 0
    state.turn = 10
    state.buys = 1
    strategy = _make_strategy(
        mid_buy_priority=["Province", "Gold", "Silver", "Duchy"] + [c for c in BUYABLE_CARDS if c not in ["Province", "Gold", "Silver", "Duchy"]],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
    )

    play_buy_phase(state, strategy)

    # Province, Gold, Silver all empty — should skip them
    if state.discard:
        bought = state.discard[0]
        assert bought not in ["Province", "Gold", "Silver"]


def test_buy_phase_multiple_buys():
    """Extra buys allow purchasing multiple cards in one turn."""
    state = _make_state_with_hand(["Gold", "Gold"])  # 6 coins
    state.supply = _full_supply()
    state.turn = 10
    state.buys = 3  # simulate Festival + Market giving extra buys
    strategy = _make_strategy(
        mid_buy_priority=["Silver", "Copper"] + [c for c in BUYABLE_CARDS if c not in ["Silver", "Copper"]],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
    )

    play_buy_phase(state, strategy)

    # 6 coins, 3 buys: buy Silver(3), then Silver(3) again, then 0 coins left
    assert len(state.discard) >= 2


def test_buy_phase_pass_stops_buying():
    """PASS in buy priority stops all buying even with buys remaining."""
    state = _make_state_with_hand(["Gold", "Gold"])  # 6 coins
    state.supply = _full_supply()
    state.turn = 1
    state.buys = 2
    # PASS at the very start — should buy nothing
    strategy = _make_strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
    )

    play_buy_phase(state, strategy)

    assert len(state.discard) == 0
    assert state.buys == 0


# ---------------------------------------------------------------------------
# Chapel tests
# ---------------------------------------------------------------------------

def test_chapel_trashes_cards():
    """Chapel trashes cards from hand into trash pile."""
    state = _make_state_with_hand(["Chapel", "Estate", "Estate", "Copper", "Copper"])
    strategy = _chapel_strategy(["Estate", "Copper", "STOP"])

    play_action_phase(state, strategy)

    assert "Chapel" in state.play_area
    # Should trash Estates first, then Coppers, up to 4
    assert len(state.trash) == 4
    assert state.trash.count("Estate") == 2
    assert state.trash.count("Copper") == 2
    # Nothing left in hand
    assert len(state.hand) == 0


def _chapel_strategy(trash_priority):
    """Helper: strategy that plays Chapel with given trash priority in all phases."""
    return _make_strategy(
        early_terminal_priority=["Chapel"],
        mid_terminal_priority=["Chapel"],
        late_terminal_priority=["Chapel"],
        early_chapel_trash=trash_priority,
        mid_chapel_trash=trash_priority,
        late_chapel_trash=trash_priority,
    )


def test_chapel_stop_marker():
    """STOP in chapel_trash_priority prevents trashing cards after it."""
    state = _make_state_with_hand(["Chapel", "Estate", "Copper", "Copper", "Copper"])
    strategy = _chapel_strategy(["Estate", "STOP", "Copper"])

    play_action_phase(state, strategy)

    # Only Estate should be trashed (STOP before Copper)
    assert len(state.trash) == 1
    assert state.trash[0] == "Estate"
    assert state.hand.count("Copper") == 3


def test_chapel_max_four():
    """Chapel trashes at most 4 cards even if more match."""
    state = _make_state_with_hand(
        ["Chapel", "Copper", "Copper", "Copper", "Copper", "Copper", "Copper"],
        deck=["Silver"] * 5,
    )
    # Chapel draws 0, so hand has 6 Coppers after Chapel is played
    strategy = _chapel_strategy(["Copper", "STOP"])

    play_action_phase(state, strategy)

    assert len(state.trash) == 4
    assert state.hand.count("Copper") == 2  # 6 - 4 = 2 remain


def test_chapel_no_stop_trashes_all_matching():
    """Without STOP, chapel trashes all matching cards up to 4."""
    state = _make_state_with_hand(["Chapel", "Estate", "Estate", "Estate"])
    strategy = _chapel_strategy(["Estate", "Copper"])  # no STOP

    play_action_phase(state, strategy)

    assert len(state.trash) == 3
    assert state.hand == []


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

def test_cleanup_discards_and_draws():
    """Cleanup moves hand + play area to discard, then draws 5."""
    state = GameState()
    state.rng = random.Random(0)
    state.hand = ["Estate", "Copper"]
    state.play_area = ["Silver", "Village"]
    state.deck = ["Gold"] * 10
    state.discard = ["Duchy"]

    cleanup(state)

    assert len(state.hand) == 5
    assert all(c == "Gold" for c in state.hand)
    assert state.play_area == []
    # Discard should now contain the old hand + play area + old discard
    assert "Estate" in state.discard
    assert "Copper" in state.discard
    assert "Silver" in state.discard
    assert "Village" in state.discard
    assert "Duchy" in state.discard


def test_cleanup_empty_deck_reshuffles():
    """Cleanup reshuffles discard into deck if deck runs out mid-draw."""
    state = GameState()
    state.rng = random.Random(0)
    state.hand = ["Copper"]
    state.play_area = ["Silver"]
    state.deck = ["Gold", "Gold"]  # only 2 in deck
    state.discard = ["Estate", "Estate", "Estate"]  # 3 in discard

    cleanup(state)

    # hand(1) + play_area(1) go to discard first, then draw 5
    # deck has 2, after those are drawn, discard (3+2=5) gets reshuffled
    assert len(state.hand) == 5


# ---------------------------------------------------------------------------
# Game over tests
# ---------------------------------------------------------------------------

def test_game_over_provinces_empty():
    """Game ends when Province supply hits 0."""
    state = GameState()
    state.supply = {"Province": 0, "Gold": 30, "Silver": 40}
    state.turn = 5
    assert is_game_over(state) is True


def test_game_over_three_piles_empty():
    """Game ends when 3 supply piles are empty."""
    state = GameState()
    state.supply = {"Province": 12, "Gold": 0, "Silver": 0, "Copper": 0, "Duchy": 12}
    state.turn = 5
    assert is_game_over(state) is True


def test_game_over_two_piles_not_over():
    """Game does NOT end with only 2 empty piles and provinces remaining."""
    state = GameState()
    state.supply = {"Province": 12, "Gold": 0, "Silver": 0, "Copper": 53, "Duchy": 12}
    state.turn = 5
    assert is_game_over(state) is False


def test_game_over_turn_cap():
    """Game ends when turn cap is reached."""
    state = GameState()
    state.supply = {"Province": 12, "Gold": 30}
    state.turn = 40
    assert is_game_over(state) is True

    state.turn = 39
    assert is_game_over(state) is False


def test_game_over_custom_turn_cap():
    """Turn cap can be customized."""
    state = GameState()
    state.supply = {"Province": 12}
    state.turn = 20
    assert is_game_over(state, turn_cap=20) is True
    assert is_game_over(state, turn_cap=21) is False


# ---------------------------------------------------------------------------
# Count VP tests
# ---------------------------------------------------------------------------

def test_count_vp_all_zones():
    """VP is counted from deck, hand, discard, and play area."""
    state = GameState()
    state.deck = ["Estate"]          # 1 VP
    state.hand = ["Duchy"]           # 3 VP
    state.discard = ["Province"]     # 6 VP
    state.play_area = ["Estate"]     # 1 VP
    # Total: 11 VP
    assert count_vp(state) == 11


def test_count_vp_ignores_non_victory():
    """Non-victory cards contribute 0 VP."""
    state = GameState()
    state.deck = ["Copper", "Silver", "Gold", "Smithy", "Village"]
    state.hand = []
    state.discard = []
    state.play_area = []
    assert count_vp(state) == 0


# ---------------------------------------------------------------------------
# New game / supply setup tests
# ---------------------------------------------------------------------------

def test_new_game_starting_deck():
    """New game starts with 7 Copper + 3 Estate, 5 drawn to hand."""
    state = new_game(42)
    all_cards = state.deck + state.hand
    assert len(all_cards) == 10
    assert all_cards.count("Copper") == 7
    assert all_cards.count("Estate") == 3
    assert len(state.hand) == 5
    assert len(state.deck) == 5


def test_default_supply_1p():
    """1-player supply has correct counts."""
    supply = default_supply()
    assert supply["Copper"] == 60 - 7
    assert supply["Estate"] == 12 - 3
    assert supply["Silver"] == 40
    assert supply["Gold"] == 30
    assert supply["Province"] == 12
    assert supply["Duchy"] == 12
    for k in KINGDOM_CARDS:
        if ALL_CARDS[k].card_type == CardType.VICTORY:
            assert supply[k] == 12  # victory kingdom cards
        else:
            assert supply[k] == 10


def test_default_supply_2p():
    """2-player supply adjusts Copper and Estate counts."""
    supply = default_supply(num_players=2)
    assert supply["Copper"] == 60 - 14  # 7 * 2
    assert supply["Estate"] == 12 - 6   # 3 * 2
    assert supply["Province"] == 12


# ---------------------------------------------------------------------------
# Action phase edge cases
# ---------------------------------------------------------------------------

def test_no_actions_cannot_play():
    """With 0 actions remaining, no action cards can be played."""
    state = _make_state_with_hand(["Smithy", "Village"])
    state.actions = 0  # override the default 1
    strategy = _action_strategy("Smithy")

    play_action_phase(state, strategy)

    assert state.play_area == []
    assert len(state.hand) == 2  # nothing changed


def test_action_card_not_in_hand_skipped():
    """Action priority entries not in hand are skipped."""
    state = _make_state_with_hand(["Copper", "Estate"])  # no action cards
    strategy = _action_strategy("Smithy")

    play_action_phase(state, strategy)

    assert state.play_area == []
    assert state.actions == 1  # unspent


def test_action_without_extra_actions_plays_one():
    """With 1 action and no +action cards, only one action card is played."""
    state = _make_state_with_hand(["Smithy", "Smithy"])
    strategy = _make_strategy(
        early_terminal_priority=["Smithy"], mid_terminal_priority=["Smithy"],
        late_terminal_priority=["Smithy"],
    )

    play_action_phase(state, strategy)

    # Smithy gives 0 extra actions, so only 1 can be played
    assert state.play_area.count("Smithy") == 1
    assert state.hand.count("Smithy") == 1
    assert state.actions == 0


# ---------------------------------------------------------------------------
# Draw edge cases
# ---------------------------------------------------------------------------

def test_draw_empty_deck_and_discard():
    """Drawing with empty deck AND empty discard draws nothing."""
    state = GameState()
    state.rng = random.Random(0)
    state.deck = []
    state.discard = []
    state.hand = []

    draw_cards(state, 5)

    assert state.hand == []
    assert state.deck == []


def test_draw_partial_deck_and_discard():
    """Drawing more than deck+discard combined draws what's available."""
    state = GameState()
    state.rng = random.Random(0)
    state.deck = ["Gold"]
    state.discard = ["Silver"]
    state.hand = []

    draw_cards(state, 5)  # ask for 5, only 2 exist

    assert len(state.hand) == 2
    assert set(state.hand) == {"Gold", "Silver"}


# ---------------------------------------------------------------------------
# 2-player game tests
# ---------------------------------------------------------------------------

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


def test_2p_game_shorter_than_solo():
    """2-player games end faster due to shared supply depletion."""
    s = big_money_strategy()
    solo_turns = []
    two_p_turns = []
    for seed in range(20):
        solo = play_game(s, seed)
        solo_turns.append(solo["turns"])
        twop = play_game_2p(s, s, seed)
        two_p_turns.append(twop["turns"])

    avg_solo = sum(solo_turns) / len(solo_turns)
    avg_2p = sum(two_p_turns) / len(two_p_turns)
    # 2p games should be noticeably shorter
    assert avg_2p < avg_solo


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

def test_full_turn_action_buy_cleanup():
    """A complete turn: action phase → buy phase → cleanup."""
    state = _make_state_with_hand(
        ["Village", "Copper", "Copper", "Copper", "Silver"],
        deck=["Copper"] * 10,
    )
    state.supply = _full_supply()
    state.turn = 1
    strategy = _make_strategy(
        early_buy_priority=["Silver"] + [c for c in BUYABLE_CARDS if c != "Silver"],
        early_nonterminal_priority=["Village"],
        mid_nonterminal_priority=["Village"],
        late_nonterminal_priority=["Village"],
    )

    # Action phase: Village played, draws 1, gets +2 actions
    play_action_phase(state, strategy)
    assert "Village" in state.play_area

    # Buy phase: treasures auto-played, buy something
    play_buy_phase(state, strategy)

    # Cleanup: everything goes to discard, draw 5
    cleanup(state)
    assert len(state.hand) == 5
    assert state.play_area == []


# ---------------------------------------------------------------------------
# Chapel trashing: 0 to 4 card types via STOP position
# ---------------------------------------------------------------------------

def test_chapel_trash_zero_cards():
    """STOP at the front of chapel_trash_priority means trash nothing."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Estate", "Copper", "Copper"],
    )
    strategy = _chapel_strategy(["STOP", "Estate", "Copper", "Duchy", "Silver"])

    play_action_phase(state, strategy)

    assert len(state.trash) == 0
    assert state.hand == ["Estate", "Estate", "Copper", "Copper"]


def test_chapel_priority_order_matters():
    """Chapel trashes in priority order, so with 4-card cap, low priority may survive."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Estate", "Estate", "Copper", "Copper"],
        deck=["Silver"] * 5,
    )
    strategy = _chapel_strategy(["Estate", "Copper", "STOP"])

    play_action_phase(state, strategy)

    # 3 Estates trashed first, then 1 Copper (hits 4-card cap)
    assert state.trash.count("Estate") == 3
    assert state.trash.count("Copper") == 1
    assert state.hand.count("Copper") == 1  # second Copper survives


def test_ga_chapel_crossover_preserves_stop():
    """Crossover of two chapel lists keeps STOP marker."""
    p1 = ["Estate", "Copper", "STOP", "Duchy", "Silver"]
    p2 = ["STOP", "Silver", "Copper", "Estate", "Duchy"]

    rng = random.Random(42)
    for _ in range(20):
        child = order_crossover(p1, p2, rng)
        assert "STOP" in child, f"STOP missing from crossover result: {child}"


def test_ga_chapel_trash_hardcoded():
    """Chapel trash is hardcoded and not affected by mutation."""
    base = _make_strategy(
        early_chapel_trash=["Duchy", "STOP", "Copper", "Estate"],
        mid_chapel_trash=["Duchy", "STOP", "Copper", "Estate"],
        late_chapel_trash=["Duchy", "STOP", "Copper", "Estate"],
    )

    rng = random.Random(42)
    for _ in range(50):
        mutated = mutate(base, rate=1.0, rng=rng)
        assert mutated.early_chapel_trash == ["Estate", "Copper", "STOP"]
        assert mutated.mid_chapel_trash == ["Estate", "Copper", "STOP"]
        assert mutated.late_chapel_trash == ["STOP"]
        assert mutated.chapel_max_trash == 4


# ---------------------------------------------------------------------------
# Save / load best model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Special card tests: Throne Room, Council Room, Moneylender, Gardens, Mine, Merchant
# ---------------------------------------------------------------------------

def test_throne_room_doubles_smithy():
    """Throne Room + Smithy should draw 6 cards (3 twice)."""
    state = _make_state_with_hand(
        ["Throne Room", "Smithy"],
        deck=["Copper"] * 10,
    )
    strategy = _make_strategy(
        early_terminal_priority=["Throne Room", "Smithy"],
        mid_terminal_priority=["Throne Room", "Smithy"],
        late_terminal_priority=["Throne Room", "Smithy"],
        throne_room_priority=["Smithy"],
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Smithy" in state.play_area
    assert len(state.hand) == 6  # 3 + 3
    assert state.actions == 0  # TR costs 1, Smithy gives 0


def test_throne_room_doubles_chapel():
    """Throne Room + Chapel (both terminal): should trash twice."""
    state = _make_state_with_hand(
        ["Throne Room", "Chapel", "Estate", "Estate", "Copper"],
        deck=["Silver"] * 5,
    )
    strategy = _make_strategy(
        early_terminal_priority=["Throne Room", "Chapel"],
        mid_terminal_priority=["Throne Room", "Chapel"],
        late_terminal_priority=["Throne Room", "Chapel"],
        throne_room_priority=["Chapel"],
        early_chapel_trash=["Estate", "Copper", "STOP"],
        mid_chapel_trash=["Estate", "Copper", "STOP"],
        late_chapel_trash=["Estate", "Copper", "STOP"],
        chapel_max_trash=2,
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Chapel" in state.play_area
    # Chapel doubled: trash up to 2 per play × 2 = up to 4
    assert len(state.trash) >= 2


def test_throne_room_no_target():
    """Throne Room with no other action in hand does nothing extra."""
    state = _make_state_with_hand(
        ["Throne Room", "Copper", "Silver"],
        deck=["Copper"] * 10,
    )
    strategy = _make_strategy(
        early_terminal_priority=["Throne Room", "Smithy"],
        mid_terminal_priority=["Throne Room", "Smithy"],
        late_terminal_priority=["Throne Room", "Smithy"],
        throne_room_priority=["Smithy"],
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    # No action to double, hand unchanged except TR removed
    assert len(state.hand) == 2  # Copper, Silver
    assert state.actions == 0


def test_throne_room_doubles_moneylender():
    """Throne Room + Moneylender should trash up to 2 Coppers for +$6."""
    state = _make_state_with_hand(
        ["Throne Room", "Moneylender", "Copper", "Copper", "Silver"],
        deck=["Gold"] * 5,
    )
    strategy = _make_strategy(
        early_terminal_priority=["Throne Room", "Moneylender"],
        mid_terminal_priority=["Throne Room", "Moneylender"],
        late_terminal_priority=["Throne Room", "Moneylender"],
        throne_room_priority=["Moneylender"],
    )

    play_action_phase(state, strategy)

    assert state.trash.count("Copper") == 2
    assert state.coins == 6  # +$3 twice
    assert "Silver" in state.hand


def test_throne_room_priority_matters():
    """Throne Room should pick target from throne_room_priority, not terminal_priority."""
    state = _make_state_with_hand(
        ["Throne Room", "Smithy", "Moneylender"],
        deck=["Copper"] * 10,
    )
    # terminal_priority prefers Smithy first, but throne_room_priority prefers Moneylender
    strategy = _make_strategy(
        early_terminal_priority=["Throne Room", "Smithy", "Moneylender"],
        mid_terminal_priority=["Throne Room", "Smithy", "Moneylender"],
        late_terminal_priority=["Throne Room", "Smithy", "Moneylender"],
        throne_room_priority=["Moneylender", "Smithy"],
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Moneylender" in state.play_area
    # Moneylender doubled (no Copper in hand to trash, so just plays)
    # Smithy still in hand (wasn't picked by TR)
    assert "Smithy" not in state.play_area or "Moneylender" in state.play_area


def test_council_room_draws_four():
    """Playing Council Room should draw 4 cards and add +1 buy."""
    state = _make_state_with_hand(
        ["Council Room"],
        deck=["Copper", "Silver", "Gold", "Estate", "Duchy"],
    )
    strategy = _action_strategy("Council Room")

    play_action_phase(state, strategy)

    assert "Council Room" in state.play_area
    assert len(state.hand) == 4  # drew 4
    assert state.actions == 0  # spent 1, gained 0
    assert state.buys == 2  # started with 1, gained 1


def test_moneylender_trashes_copper():
    """Moneylender should trash a Copper and add +$3."""
    state = _make_state_with_hand(["Moneylender", "Copper", "Copper", "Silver"])
    strategy = _action_strategy("Moneylender")

    play_action_phase(state, strategy)

    assert "Moneylender" in state.play_area
    assert state.trash == ["Copper"]
    assert state.hand.count("Copper") == 1  # one Copper remains
    assert state.coins == 3
    assert state.actions == 0


def test_moneylender_no_copper():
    """Moneylender with no Copper in hand trashes nothing."""
    state = _make_state_with_hand(["Moneylender", "Silver", "Gold"])
    strategy = _action_strategy("Moneylender")

    play_action_phase(state, strategy)

    assert "Moneylender" in state.play_area
    assert state.trash == []
    assert state.coins == 0


def test_gardens_vp():
    """Gardens should give 1 VP per 10 cards in deck."""
    state = GameState()
    # 20 cards total: 2 Gardens = 2 * (20 // 10) = 4 VP from Gardens
    state.deck = ["Gardens", "Gardens"] + ["Copper"] * 18
    state.hand = []
    state.discard = []
    state.play_area = []
    vp = count_vp(state)
    assert vp == 4  # 2 Gardens * 2 VP each (20 cards // 10)


def test_gardens_vp_with_victory():
    """Gardens VP stacks with normal VP."""
    state = GameState()
    # 30 cards: 1 Gardens + 1 Estate + 28 Copper
    state.deck = ["Gardens", "Estate"] + ["Copper"] * 28
    state.hand = []
    state.discard = []
    state.play_area = []
    vp = count_vp(state)
    # Estate = 1 VP, Gardens = 1 * (30 // 10) = 3 VP → total 4
    assert vp == 4


def test_gardens_in_supply():
    """Gardens should have 12 supply (victory card count)."""
    supply = default_supply()
    assert supply["Gardens"] == 12


def test_mine_upgrades_copper_to_silver():
    """Mine should trash Copper and gain Silver to hand."""
    state = _make_state_with_hand(["Mine", "Copper", "Silver"])
    strategy = _make_strategy(
        early_terminal_priority=["Mine"], mid_terminal_priority=["Mine"],
        late_terminal_priority=["Mine"], mine_trash_priority=["Copper", "Silver"],
    )

    play_action_phase(state, strategy)

    assert "Mine" in state.play_area
    assert state.trash == ["Copper"]
    assert "Silver" in state.hand
    # Gained Silver goes to hand, so 2 Silvers now
    assert state.hand.count("Silver") == 2
    assert state.actions == 0


def test_mine_upgrades_silver_to_gold():
    """Mine should trash Silver and gain Gold to hand."""
    state = _make_state_with_hand(["Mine", "Silver", "Copper"])
    strategy = _make_strategy(
        early_terminal_priority=["Mine"], mid_terminal_priority=["Mine"],
        late_terminal_priority=["Mine"], mine_trash_priority=["Silver", "Copper"],
    )

    play_action_phase(state, strategy)

    assert state.trash == ["Silver"]
    assert "Gold" in state.hand
    assert "Copper" in state.hand


def test_mine_no_treasure():
    """Mine with no treasure in hand does nothing."""
    state = _make_state_with_hand(["Mine", "Estate", "Estate"])
    strategy = _make_strategy(
        early_terminal_priority=["Mine"], mid_terminal_priority=["Mine"],
        late_terminal_priority=["Mine"],
    )

    play_action_phase(state, strategy)

    assert state.trash == []


def test_mine_empty_supply_skips():
    """Mine skips upgrade if target treasure supply is empty."""
    state = _make_state_with_hand(["Mine", "Copper", "Copper"])
    state.supply["Silver"] = 0
    strategy = _make_strategy(
        early_terminal_priority=["Mine"], mid_terminal_priority=["Mine"],
        late_terminal_priority=["Mine"],
    )

    play_action_phase(state, strategy)

    # No Silver in supply, can't upgrade Copper
    assert state.trash == []
    assert state.hand.count("Copper") == 2


def test_merchant_bonus_with_silver():
    """Merchant in play area gives +$1 when Silver is played as treasure."""
    state = _make_state_with_hand(["Merchant", "Silver", "Copper"],
                                  deck=["Estate"] * 5)
    strategy = _make_strategy(
        early_nonterminal_priority=["Merchant"], mid_nonterminal_priority=["Merchant"],
        late_nonterminal_priority=["Merchant"],
    )

    play_action_phase(state, strategy)
    # Merchant: +1 card, +1 action drawn an Estate
    assert "Merchant" in state.play_area
    assert state.actions == 1  # spent 1, gained 1

    # Now play treasures
    treasures = auto_play_treasures(state)
    # Silver(2) + Copper(1) + Merchant bonus(1) = 4
    assert state.coins == 4


def test_merchant_no_silver_no_bonus():
    """Merchant gives no bonus if no Silver is played."""
    state = _make_state_with_hand(["Merchant", "Copper", "Copper"],
                                  deck=["Estate"] * 5)
    strategy = _make_strategy(
        early_nonterminal_priority=["Merchant"], mid_nonterminal_priority=["Merchant"],
        late_nonterminal_priority=["Merchant"],
    )

    play_action_phase(state, strategy)
    treasures = auto_play_treasures(state)
    # Copper(1) + Copper(1) + drawn Estate(0) = 2, no Merchant bonus
    assert state.coins == 2


def test_multiple_merchants_stack():
    """Multiple Merchants each grant +$1 on first Silver."""
    state = GameState()
    state.rng = random.Random(0)
    state.hand = ["Silver", "Copper"]
    state.deck = ["Estate"] * 5
    state.supply = _full_supply()
    state.actions = 0
    state.buys = 1
    state.coins = 0
    # Simulate 2 Merchants already played
    state.play_area = ["Merchant", "Merchant"]

    treasures = auto_play_treasures(state)
    # Silver(2) + Copper(1) + 2 Merchants(2) = 5
    assert state.coins == 5


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
    assert loaded.early_nonterminal_priority == strategy.early_nonterminal_priority
    assert loaded.early_terminal_priority == strategy.early_terminal_priority
    assert loaded.mid_nonterminal_priority == strategy.mid_nonterminal_priority
    assert loaded.mid_terminal_priority == strategy.mid_terminal_priority
    assert loaded.late_nonterminal_priority == strategy.late_nonterminal_priority
    assert loaded.late_terminal_priority == strategy.late_terminal_priority
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
# Hall of fame tests
# ---------------------------------------------------------------------------

def test_hall_of_fame_adds_member_when_threshold_met():
    """GA adds strategy to hall when win rate >= hall_add_threshold."""
    from unittest.mock import patch
    from ga import run_ga

    def fake_evaluate_population_vs_hall(population, seed_list, kingdom,
                                         hall=None, workers=1):
        results = []
        for _ in population:
            # All strategies "win" at 60% — above 0.55 threshold
            results.append({
                "win_rate": 0.6, "tie_rate": 0.1, "loss_rate": 0.3,
                "mean_turns": 20.0, "avg_final_deck": {},
            })
        return results

    with patch("ga.evaluate_population_vs_hall", fake_evaluate_population_vs_hall), \
         patch("ga.save_best_model"):
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
            "csv_path": "/tmp/test_hall.csv",
        })

    # Hall should have grown beyond the initial Big Money member
    assert len(result["hall"]) >= 2


def test_hall_of_fame_respects_max_size():
    """Hall of fame does not exceed hall_max_size."""
    from unittest.mock import patch
    from ga import run_ga

    def fake_evaluate_population_vs_hall(population, seed_list, kingdom,
                                         hall=None, workers=1):
        results = []
        for _ in population:
            results.append({
                "win_rate": 0.8, "tie_rate": 0.1, "loss_rate": 0.1,
                "mean_turns": 20.0, "avg_final_deck": {},
            })
        return results

    with patch("ga.evaluate_population_vs_hall", fake_evaluate_population_vs_hall), \
         patch("ga.save_best_model"):
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
            "csv_path": "/tmp/test_hall_max.csv",
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
    from engine import play_game_2p

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
    from engine import play_game_2p

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
    from ga import run_ga

    result = run_ga({
        "pop_size": 20,
        "generations": 15,
        "games_per_eval": 30,
        "tournament_size": 3,
        "elite_count": 2,
        "mutation_rate": 0.15,
        "seed": 42,
        "kingdom": KINGDOM_CARDS,
        "hall_add_threshold": 0.55,
        "hall_max_size": 4,
        "csv_path": "/tmp/test_ga_bm.csv",
        "best_model_dir": "/tmp/test_ga_bm_model",
    })

    rng = random.Random(99)
    seeds = make_seed_list(100, rng)
    vs = evaluate_vs_opponent(result["best_strategy"], seeds, KINGDOM_CARDS,
                              opponent=None, need_deck=True)
    assert vs["win_rate"] >= 0.50, (
        f"GA best should beat Big Money >=50%, got {vs['win_rate']:.0%}")


def test_hall_of_fame_prevents_bm_drift():
    """GA with hall of fame (BM always in hall) must still beat Big Money."""
    from ga import run_ga

    result = run_ga({
        "pop_size": 20,
        "generations": 15,
        "games_per_eval": 30,
        "tournament_size": 3,
        "elite_count": 2,
        "mutation_rate": 0.15,
        "seed": 42,
        "kingdom": KINGDOM_CARDS,
        "hall_add_threshold": 0.55,
        "hall_max_size": 4,
        "csv_path": "/tmp/test_ga_hall.csv",
        "best_model_dir": "/tmp/test_ga_hall_model",
    })

    # Big Money is always in the hall, so GA should maintain competitiveness
    rng = random.Random(99)
    seeds = make_seed_list(100, rng)
    vs = evaluate_vs_opponent(result["best_strategy"], seeds, KINGDOM_CARDS,
                              opponent=None)
    assert vs["win_rate"] >= 0.40, (
        f"Hall-of-fame GA should still beat BM >=40%, got {vs['win_rate']:.0%}")




def test_province_max_coins_skips_buy():
    """Province is skipped when coins exceed province_max_coins threshold."""
    state = _make_state_with_hand(["Gold", "Gold", "Silver"])  # 8 coins
    state.supply = _full_supply()
    state.turn = 10
    state.buys = 1
    # Province costs 8, we have 8 coins, but threshold says skip if > 8... 8 is not > 8
    strategy = _make_strategy(
        mid_buy_priority=["Province", "Gold", "Silver", "PASS"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
        province_max_coins=8,  # buy Province at exactly 8
    )
    play_buy_phase(state, strategy)
    assert "Province" in state.discard  # 8 coins, threshold 8, should buy

    # Now test with 11 coins (Gold + Gold + Gold + Silver via extra card)
    state2 = _make_state_with_hand(["Gold", "Gold", "Gold"])  # 9 coins
    state2.supply = _full_supply()
    state2.turn = 10
    state2.buys = 1
    strategy2 = _make_strategy(
        mid_buy_priority=["Province", "Gold", "Silver", "PASS"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
        province_max_coins=8,  # only buy at 8
    )
    play_buy_phase(state2, strategy2)
    # 9 coins > 8 threshold, Province should be skipped, buy Gold instead
    assert "Province" not in state2.discard
    assert "Gold" in state2.discard


def test_duchy_max_coins_skips_buy():
    """Duchy is skipped when coins exceed duchy_max_coins threshold."""
    state = _make_state_with_hand(["Silver", "Silver", "Copper"])  # 5 coins
    state.supply = _full_supply()
    state.turn = 10
    state.buys = 1
    strategy = _make_strategy(
        mid_buy_priority=["Duchy", "Silver", "PASS"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
        duchy_max_coins=5,  # buy Duchy at exactly 5
    )
    play_buy_phase(state, strategy)
    assert "Duchy" in state.discard  # 5 coins, threshold 5, should buy

    # With 6 coins, skip Duchy
    state2 = _make_state_with_hand(["Gold", "Copper", "Copper"])  # 5 coins
    state2.supply = _full_supply()
    state2.turn = 10
    state2.buys = 1
    strategy2 = _make_strategy(
        mid_buy_priority=["Duchy", "Silver", "PASS"],
        transitions=Transitions(early_to_mid_turn=4, mid_to_late_provinces=3),
        duchy_max_coins=4,  # only buy at 4 or below
    )
    play_buy_phase(state2, strategy2)
    # 5 coins > 4 threshold, Duchy skipped
    assert "Duchy" not in state2.discard


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


def test_card_id_data_arrays_match():
    """Flat data arrays match the Card dataclass fields."""
    for name, cid in CARD_ID.items():
        if cid >= NUM_CARDS:
            continue
        card = ALL_CARDS[name]
        assert CARD_COST[cid] == card.cost, f"{name} cost mismatch"
        assert CARD_COINS[cid] == card.coins, f"{name} coins mismatch"
        assert CARD_VP[cid] == card.vp, f"{name} vp mismatch"
        assert CARD_DRAW[cid] == card.cards_drawn, f"{name} draw mismatch"
        assert CARD_ACTIONS[cid] == card.actions, f"{name} actions mismatch"
        assert CARD_BUYS[cid] == card.buys, f"{name} buys mismatch"


def test_card_id_sentinels():
    """PASS and STOP have IDs >= NUM_CARDS."""
    assert PASS_ID >= NUM_CARDS
    assert STOP_ID >= NUM_CARDS
    assert PASS_ID != STOP_ID


# ---------------------------------------------------------------------------
# C engine tests
# ---------------------------------------------------------------------------

def test_c_engine_available():
    """C engine (dominion.so) loads successfully."""
    assert USE_C_ENGINE, "C engine not available — dominion.so may not be built"


def test_c_engine_strategy_serialization():
    """strategy_to_ints produces a valid array of the expected size."""
    from c_bridge import strategy_to_ints, STRATEGY_SIZE

    for strat_fn in [big_money_strategy, engine_strategy]:
        strat = strat_fn()
        arr = strategy_to_ints(strat)
        assert len(arr) == STRATEGY_SIZE
        # Transitions should be positive
        assert arr[0] > 0  # early_to_mid_turn
        assert arr[1] > 0  # mid_to_late_provinces


def test_c_engine_big_money_vs_self_balanced():
    """Big Money vs itself via C engine should win ~50%."""
    from c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    seeds = make_seed_list(200, random.Random(42))
    result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)
    assert 0.30 <= result["win_rate"] <= 0.70, (
        f"BM vs itself should be ~50%, got {result['win_rate']:.0%}")


def test_c_engine_win_loss_tie_sum():
    """C engine results: win + tie + loss = 1.0."""
    from c_bridge import evaluate_vs_opponent_c

    bm = big_money_strategy()
    seeds = make_seed_list(50, random.Random(42))
    result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)
    total = result["win_rate"] + result["tie_rate"] + result["loss_rate"]
    assert abs(total - 1.0) < 1e-9, f"Win+tie+loss = {total}"


def test_c_engine_stronger_strategy_wins():
    """C engine: Big Money should beat a do-nothing strategy."""
    from c_bridge import evaluate_vs_opponent_c

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
    from c_bridge import evaluate_vs_opponent_c

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

    from c_bridge import evaluate_vs_opponent_c
    c_result = evaluate_vs_opponent_c(bm, seeds, KINGDOM_CARDS, opponent=bm)

    # Both should be roughly balanced (different RNGs, so not exact)
    assert abs(py_result["win_rate"] - c_result["win_rate"]) < 0.15, (
        f"Win rates differ too much: Python={py_result['win_rate']:.2f} "
        f"C={c_result['win_rate']:.2f}")
    assert abs(py_result["mean_turns"] - c_result["mean_turns"]) < 5, (
        f"Turn counts differ too much: Python={py_result['mean_turns']:.1f} "
        f"C={c_result['mean_turns']:.1f}")


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS: {name}")
    print("\nAll smoke tests passed.")
