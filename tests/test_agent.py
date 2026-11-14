"""What the genome decides (core/agent.py), on hand-made positions. Rules are tested in engine/."""
import core  # noqa: F401
from core.agent import GAAgent
from core.strategy import Transitions, big_money_strategy, engine_strategy
from dominion_engine import Game
from dominion_engine import cards as K
from dominion_engine import game as G

KINGDOM = ["Village", "Smithy", "Market", "Laboratory", "Festival", "Chapel", "Throne Room",
           "Mine", "Militia", "Remodel"]


def ids(*names):
    return [K.ID[n] for n in names]


def position(hand=(), owned=(), turn=1, coins=0):
    g = Game(KINGDOM, seed=1)
    p = g.players[0]
    p.hand, p.deck, p.discard = ids(*hand), ids(*owned), []
    p.turns = turn - 1
    g.coins = coins
    return g


def ask(agent, g, kind, options, may_pass=True, subject=-1):
    ans = agent.decide(g, G.Decision(kind, 0, tuple(sorted(options)), may_pass, subject))
    return K.NAME.get(ans, ans) if ans >= 0 else None


def buyable(g):
    return [c for c in range(K.NUM_CARDS) if g.supply[c] > 0 and K.CARDS[c].cost <= g.coins]


def strategy(**kw):
    s = big_money_strategy(KINGDOM)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_nonterminals_before_terminals():
    a = GAAgent(strategy(nonterminal_priority=["Village"], terminal_priority=["Smithy"]))
    g = position(hand=["Smithy", "Village"])
    assert ask(a, g, G.PLAY_ACTION, ids("Smithy", "Village")) == "Village"


def test_buy_follows_phase_list_and_pass_stops():
    s = strategy(early_buy_priority=["Smithy", "PASS", "Silver"])
    g = position(coins=4)
    assert ask(GAAgent(s), g, G.BUY, buyable(g)) == "Smithy"
    g = position(coins=3)
    assert ask(GAAgent(s), g, G.BUY, buyable(g)) is None


def test_buy_target_limits_copies():
    s = strategy(early_buy_priority=["Smithy", "Silver", "PASS"], buy_targets={"Smithy": 1},
                 terminal_slack=5)
    g = position(coins=4, owned=["Smithy"])
    assert ask(GAAgent(s), g, G.BUY, buyable(g)) == "Silver"


def test_terminal_density_rule():
    s = strategy(early_buy_priority=["Smithy", "Silver", "PASS"], terminal_slack=1)
    a = GAAgent(s)
    g = position(coins=4, owned=["Smithy"])
    assert ask(a, g, G.BUY, buyable(g)) == "Silver"               # 1 terminal >= 0 + slack 1
    g = position(coins=4, owned=["Smithy", "Village"])
    assert ask(a, g, G.BUY, buyable(g)) == "Smithy"               # Village adds capacity
    s.early_buy_priority = ["Throne Room", "PASS"]
    g = position(coins=4, owned=["Smithy", "Smithy"])
    assert ask(GAAgent(s), g, G.BUY, buyable(g)) == "Throne Room"  # never blocked


def test_chapel_order_stop_and_max():
    s = engine_strategy(KINGDOM)
    s.early_chapel_trash = ["Estate", "Copper", "STOP"]
    s.chapel_max_trash = 3
    a = GAAgent(s)
    g = position(hand=["Estate", "Copper", "Silver"])
    assert ask(a, g, G.CHAPEL_TRASH, ids("Copper", "Estate", "Silver"), subject=0) == "Estate"
    assert ask(a, g, G.CHAPEL_TRASH, ids("Silver"), subject=1) is None
    assert ask(a, g, G.CHAPEL_TRASH, ids("Copper"), subject=3) is None   # max reached


def test_phase_specific_chapel_list():
    s = engine_strategy(KINGDOM)
    s.transitions = Transitions(early_to_mid_turn=2, mid_to_late_provinces=4, mid_to_late_turn=20)
    s.early_chapel_trash, s.mid_chapel_trash = ["Copper", "STOP"], ["STOP", "Copper"]
    a = GAAgent(s)
    assert ask(a, position(turn=1), G.CHAPEL_TRASH, ids("Copper"), subject=0) == "Copper"
    assert ask(a, position(turn=5), G.CHAPEL_TRASH, ids("Copper"), subject=0) is None


def test_throne_room_priority():
    a = GAAgent(strategy(throne_room_priority=["Market", "Smithy"]))
    assert ask(a, position(), G.THRONE, ids("Smithy", "Market")) == "Market"


def test_mine_upgrades_by_priority():
    a = GAAgent(strategy(mine_trash_priority=["Silver", "Copper"]))
    g = position(hand=["Copper", "Silver"])
    assert ask(a, g, G.MINE_TRASH, ids("Copper", "Silver")) == "Silver"
    assert ask(a, g, G.MINE_GAIN, ids("Copper", "Silver", "Gold"), False, K.SILVER) == "Gold"


def test_militia_discard_junk_then_duplicate_actions():
    a = GAAgent(engine_strategy(KINGDOM))
    g = position(hand=["Gold", "Estate", "Smithy", "Smithy", "Village"])
    assert ask(a, g, G.MILITIA_DISCARD, ids("Gold", "Estate", "Smithy", "Village"), False) == "Estate"
    g = position(hand=["Gold", "Smithy", "Smithy", "Village", "Silver"])
    assert ask(a, g, G.MILITIA_DISCARD, ids("Gold", "Smithy", "Village", "Silver"), False) == "Smithy"


def test_remodel_gold_into_province_late():
    s = engine_strategy(KINGDOM)
    s.transitions = Transitions(early_to_mid_turn=2, mid_to_late_provinces=8, mid_to_late_turn=3)
    a = GAAgent(s)
    g = position(hand=["Gold", "Estate"], turn=10)
    assert ask(a, g, G.REMODEL_TRASH, ids("Gold", "Estate"), False) == "Gold"
    gains = [c for c in range(K.NUM_CARDS) if g.supply[c] > 0 and K.CARDS[c].cost <= 8]
    assert ask(a, g, G.REMODEL_GAIN, gains, False, K.GOLD) == "Province"


def test_old_models_never_play_cards_they_never_learned():
    s = big_money_strategy(["Smithy"])           # lists know only Smithy
    a = GAAgent(s)
    assert ask(a, position(hand=["Library"]), G.PLAY_ACTION, ids("Library")) is None
