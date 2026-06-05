"""Smoke tests for all components."""

import random

from cards import ALL_CARDS, BUYABLE_CARDS, ACTION_CARDS, KINGDOM_CARDS, CardType
from engine import (
    new_game, play_game, play_game_2p, play_action_phase, play_buy_phase,
    play_chapel, play_moneylender, play_throne_room, play_mine,
    auto_play_treasures, cleanup,
    is_game_over, count_vp, default_supply, GameState, draw_cards,
)
from strategy import (
    Strategy, Transitions, get_current_phase, random_strategy,
    big_money_strategy, engine_strategy, describe,
    save_best_model, load_strategy,
)
from fitness import evaluate, make_seed_list
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

        non_stop = [c for c in s.action_priority if c != "STOP"]
        assert set(non_stop) == action_set

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


def _action_strategy(card_name):
    """Helper: strategy that plays a single action card."""
    return Strategy(
        early_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        mid_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        late_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        action_priority=[card_name],
        chapel_trash_priority=["Estate", "Copper", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        mid_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        late_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        action_priority=["Village", "Smithy"],
        chapel_trash_priority=["Estate", "Copper", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=["Province"] + [c for c in BUYABLE_CARDS if c != "Province"],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=["Province", "Gold", "Duchy", "Silver"] + [c for c in BUYABLE_CARDS if c not in ["Province", "Gold", "Duchy", "Silver"]],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=["Province", "Gold", "Silver", "Duchy"] + [c for c in BUYABLE_CARDS if c not in ["Province", "Gold", "Silver", "Duchy"]],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=["Silver", "Copper"] + [c for c in BUYABLE_CARDS if c not in ["Silver", "Copper"]],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
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
    strategy = Strategy(
        early_buy_priority=["PASS"] + BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=3),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert "Chapel" in state.play_area
    # Should trash Estates first, then Coppers, up to 4
    assert len(state.trash) == 4
    assert state.trash.count("Estate") == 2
    assert state.trash.count("Copper") == 2
    # Nothing left in hand
    assert len(state.hand) == 0


def test_chapel_stop_marker():
    """STOP in chapel_trash_priority prevents trashing cards after it."""
    state = _make_state_with_hand(["Chapel", "Estate", "Copper", "Copper", "Copper"])
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "STOP", "Copper"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Copper", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert len(state.trash) == 4
    assert state.hand.count("Copper") == 2  # 6 - 4 = 2 remain


def test_chapel_no_stop_trashes_all_matching():
    """Without STOP, chapel trashes all matching cards up to 4."""
    state = _make_state_with_hand(["Chapel", "Estate", "Estate", "Estate"])
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper"],  # no STOP
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Smithy"],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=["Silver"] + [c for c in BUYABLE_CARDS if c != "Silver"],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Village"],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["STOP", "Estate", "Copper", "Duchy", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert len(state.trash) == 0
    assert state.hand == ["Estate", "Estate", "Copper", "Copper"]


def test_chapel_trash_one_type():
    """STOP after first entry trashes only that card type."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Estate", "Copper", "Copper"],
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "STOP", "Copper", "Duchy", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert state.trash == ["Estate", "Estate"]
    assert state.hand == ["Copper", "Copper"]


def test_chapel_trash_two_types():
    """STOP after two entries trashes both types."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Copper", "Copper", "Silver"],
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper", "STOP", "Duchy", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert len(state.trash) == 3  # 1 Estate + 2 Copper
    assert state.hand == ["Silver"]


def test_chapel_trash_three_types():
    """STOP after three entries trashes all three types (up to 4 cards)."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Copper", "Silver", "Duchy"],
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper", "Silver", "STOP", "Duchy"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert len(state.trash) == 3  # 1 of each
    assert state.hand == ["Duchy"]


def test_chapel_trash_four_types():
    """STOP at the end (or absent) trashes all listed types up to 4."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Copper", "Silver", "Duchy"],
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper", "Silver", "Duchy", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert len(state.trash) == 4
    assert state.hand == []


def test_chapel_priority_order_matters():
    """Chapel trashes in priority order, so with 4-card cap, low priority may survive."""
    state = _make_state_with_hand(
        ["Chapel", "Estate", "Estate", "Estate", "Copper", "Copper"],
        deck=["Silver"] * 5,
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Chapel"],
        chapel_trash_priority=["Estate", "Copper", "STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

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


def test_ga_chapel_mutation_moves_stop():
    """Mutation can swap STOP to different positions, changing trash behavior."""
    base = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=ACTION_CARDS[:],
        chapel_trash_priority=["Estate", "Copper", "STOP", "Duchy", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    rng = random.Random(42)
    stop_positions = set()
    for _ in range(200):
        mutated = mutate(base, rate=1.0, rng=rng)  # high rate to force swaps
        pos = mutated.chapel_trash_priority.index("STOP")
        stop_positions.add(pos)

    # With enough mutations, STOP should appear in multiple positions
    assert len(stop_positions) >= 3, f"STOP only found at positions {stop_positions}"


# ---------------------------------------------------------------------------
# Save / load best model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# New card tests: Woodcutter, Council Room, Moneylender, Gardens
# ---------------------------------------------------------------------------

def test_throne_room_doubles_smithy():
    """Throne Room + Smithy should draw 6 cards (3 twice)."""
    state = _make_state_with_hand(
        ["Throne Room", "Smithy"],
        deck=["Copper"] * 10,
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Throne Room", "Smithy"],
        chapel_trash_priority=["STOP"],
        throne_room_priority=["Smithy"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Smithy" in state.play_area
    assert len(state.hand) == 6  # 3 + 3
    assert state.actions == 0  # TR costs 1, Smithy gives 0


def test_throne_room_doubles_village():
    """Throne Room + Village: draw 2 cards, gain 4 actions."""
    state = _make_state_with_hand(
        ["Throne Room", "Village"],
        deck=["Copper"] * 10,
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Throne Room", "Village"],
        chapel_trash_priority=["STOP"],
        throne_room_priority=["Village"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Village" in state.play_area
    assert len(state.hand) == 2  # 1 + 1
    # TR: 1 - 1 + 0 = 0, then Village doubled: 0 + 2 + 2 = 4
    assert state.actions == 4


def test_throne_room_no_target():
    """Throne Room with no other action in hand does nothing extra."""
    state = _make_state_with_hand(
        ["Throne Room", "Copper", "Silver"],
        deck=["Copper"] * 10,
    )
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Throne Room", "Smithy"],
        chapel_trash_priority=["STOP"],
        throne_room_priority=["Smithy"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Throne Room", "Moneylender"],
        chapel_trash_priority=["STOP"],
        throne_room_priority=["Moneylender"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert state.trash.count("Copper") == 2
    assert state.coins == 6  # +$3 twice
    assert "Silver" in state.hand


def test_throne_room_priority_matters():
    """Throne Room should pick target from throne_room_priority, not action_priority."""
    state = _make_state_with_hand(
        ["Throne Room", "Village", "Smithy"],
        deck=["Copper"] * 10,
    )
    # action_priority prefers Village, but throne_room_priority prefers Smithy
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Throne Room", "Village", "Smithy"],
        chapel_trash_priority=["STOP"],
        throne_room_priority=["Smithy", "Village"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert "Throne Room" in state.play_area
    assert "Smithy" in state.play_area
    # Smithy doubled = 6 cards drawn; Village still in hand
    assert "Village" in state.hand
    assert len(state.hand) == 7  # Village + 6 drawn


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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Mine"],
        chapel_trash_priority=["STOP"],
        mine_trash_priority=["Copper", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Mine"],
        chapel_trash_priority=["STOP"],
        mine_trash_priority=["Silver", "Copper"],  # prefer Silver upgrade
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert state.trash == ["Silver"]
    assert "Gold" in state.hand
    assert "Copper" in state.hand


def test_mine_no_treasure():
    """Mine with no treasure in hand does nothing."""
    state = _make_state_with_hand(["Mine", "Estate", "Estate"])
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Mine"],
        chapel_trash_priority=["STOP"],
        mine_trash_priority=["Copper", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    assert state.trash == []


def test_mine_empty_supply_skips():
    """Mine skips upgrade if target treasure supply is empty."""
    state = _make_state_with_hand(["Mine", "Copper", "Copper"])
    state.supply["Silver"] = 0
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Mine"],
        chapel_trash_priority=["STOP"],
        mine_trash_priority=["Copper", "Silver"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
    )

    play_action_phase(state, strategy)

    # No Silver in supply, can't upgrade Copper
    assert state.trash == []
    assert state.hand.count("Copper") == 2


def test_merchant_bonus_with_silver():
    """Merchant in play area gives +$1 when Silver is played as treasure."""
    state = _make_state_with_hand(["Merchant", "Silver", "Copper"],
                                  deck=["Estate"] * 5)
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Merchant"],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    strategy = Strategy(
        early_buy_priority=BUYABLE_CARDS[:],
        mid_buy_priority=BUYABLE_CARDS[:],
        late_buy_priority=BUYABLE_CARDS[:],
        action_priority=["Merchant"],
        chapel_trash_priority=["STOP"],
        transitions=Transitions(early_to_mid_turn=6, mid_to_late_provinces=4),
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
    assert loaded.action_priority == strategy.action_priority
    assert loaded.chapel_trash_priority == strategy.chapel_trash_priority
    assert loaded.transitions.early_to_mid_turn == strategy.transitions.early_to_mid_turn
    assert loaded.transitions.mid_to_late_provinces == strategy.transitions.mid_to_late_provinces

    # Summary text contains key info
    with open(os.path.join(out, "summary.txt")) as f:
        text = f.read()
    assert "BEST TACTIC SUMMARY" in text
    assert "42%" in text


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS: {name}")
    print("\nAll smoke tests passed.")
