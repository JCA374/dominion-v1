"""Protocol 2: a player asking the per-card requests gets exactly the choices the genome makes in
the engine (the games it was trained and benchmarked on).

Engine games are played with GA agents; at each card decision of seat 0 the observation a player
would send is built from the game, the policy is asked, and its answer must equal the agent's.
"""
import random
from collections import Counter

from core.agent import GAAgent
from core.cards import KINGDOM_CARDS
from core.strategy import load_strategy, random_strategy
from dominion_engine import cards as K
from dominion_engine import game as G
from dominion_engine.game import Game
from ga.specialist import complete
from policy.ga_policy import GAPolicy
from policy.protocol import NO_REQUEST_YET, observation

NEW = ["Artisan", "Cellar", "Harbinger", "Library", "Poacher", "Remodel", "Sentry", "Vassal",
       "Workshop", "Mine", "Moneylender", "Bandit", "Bureaucrat"]


def obs_of(g: Game) -> dict:
    p = g.players[0]
    return observation(turn=g.turn_number(0), provinces_left=g.provinces_left,
                       kingdom=[K.NAME[c] for c in g.kingdom],
                       empty_piles=[K.NAME[c] for c in range(K.NUM_CARDS) if g.supply[c] == 0
                                    and (c in g.kingdom or c < 7)],
                       owned=dict(Counter(K.NAME[c] for c in p.all_cards())),
                       hand=[K.NAME[c] for c in p.hand], actions=g.actions, buys=g.buys, coins=g.coins)


class Spy:
    """Seat 0: plays as the GA agent and checks the policy against every card decision."""

    def __init__(self, strategy):
        self.agent, self.policy = GAAgent(strategy), GAPolicy({"name": "t"}, strategy)
        self.expect = []           # answers the policy gave for the rest of a card's sequence
        self.checked = Counter()
        self.poaching = 0          # Poacher discards still to come

    def want(self, d, got_name):
        assert self.expect, f"no policy answer for {d}"
        exp = self.expect.pop(0)
        assert exp == got_name, (str(d), exp, got_name)

    def decide(self, g, d):
        ans = self.agent.decide(g, d)
        if d.player != 0:
            return ans
        name = None if ans < 0 else K.NAME.get(ans)
        o, k = obs_of(g), d.kind
        pol = self.policy
        self.checked[G.KIND_NAME[k]] += 1
        if k == G.CELLAR_DISCARD:
            if d.subject == 0:
                out = pol.cellar(o)
                self.expect = out + ([None] if len(out) < len(o["hand"]) else [])
            self.want(d, name)
        elif k == G.POACHER_DISCARD:
            if not self.poaching:
                self.expect = pol.poacher(o, d.subject)
                self.poaching = min(d.subject, len(o["hand"]))
            self.poaching -= 1
            self.want(d, name)
        elif k == G.REMODEL_TRASH:
            r = pol.remodel(o)
            self.expect = [r["trash"], r["gain"]]
            self.want(d, name)
        elif k in (G.REMODEL_GAIN, G.MINE_GAIN, G.ARTISAN_TOPDECK):
            self.want(d, name)
        elif k == G.MINE_TRASH:
            r = pol.mine(o)
            self.expect = [r["trash"]] + ([r["gain"]] if r["trash"] else [])
            self.want(d, name)
        elif k == G.WORKSHOP_GAIN:
            assert pol.workshop(o) == name
        elif k == G.ARTISAN_GAIN:
            r = pol.artisan(o)
            self.expect = [r["gain"], r["topdeck"]]
            self.want(d, name)
        elif k == G.MONEYLENDER_TRASH:
            assert pol.moneylender(o) == (ans == K.COPPER)
        elif k == G.HARBINGER_TOPDECK:
            assert pol.harbinger(o, [K.NAME[c] for c in g.players[0].discard]) == name
        elif k == G.VASSAL_PLAY:
            assert pol.vassal(o, K.NAME[d.subject]) == (ans >= 0)
        elif k == G.LIBRARY_SKIP:
            assert pol.library(o, K.NAME[d.subject]) == (ans >= 0)
        elif k == G.SENTRY_CARD:
            r = pol.sentry(o, [K.NAME[d.subject]])
            placed = {G.SENTRY_TRASH: "trash", G.SENTRY_DISCARD: "discard", G.SENTRY_KEEP: "keep"}[ans]
            assert r[placed] == [K.NAME[d.subject]], (str(d), r)
        elif k == G.SENTRY_TOP:
            r = pol.sentry(o, [K.NAME[c] for c in d.options])
            assert r["keep"][0] == name, (str(d), r)
        elif k == G.BANDIT_TRASH:
            assert pol.bandit(o, [K.NAME[c] for c in d.options]) == name
        elif k == G.BUREAUCRAT_TOPDECK:
            assert pol.bureaucrat(o) == name
        return ans


def test_no_card_is_blocked_any_more():
    assert NO_REQUEST_YET == ()


def test_policy_answers_equal_the_genome_in_engine_games():
    rnd = random.Random(5)
    total = Counter()
    for n in range(300):
        kingdom = rnd.sample(NEW, 7) + rnd.sample([c for c in KINGDOM_CARDS if c not in NEW], 3)
        s0 = complete(random_strategy(rnd, kingdom))
        # buy the new cards early, so their choices come up
        s0.early_buy_priority = [c for c in kingdom if c in NEW] + s0.early_buy_priority
        s0.nonterminal_priority = [c for c in kingdom if c in NEW] + s0.nonterminal_priority
        s0.terminal_priority = [c for c in kingdom if c in NEW] + s0.terminal_priority
        s1 = complete(random_strategy(rnd, kingdom))
        s1.early_buy_priority = ["Bandit", "Bureaucrat", "Militia"] + s1.early_buy_priority
        spy = Spy(s0)
        G.play(kingdom, [spy, GAAgent(s1)], rnd.getrandbits(31))
        total += spy.checked
    for kind in ("cellar_discard", "poacher_discard", "remodel_trash", "remodel_gain", "workshop_gain",
                 "artisan_gain", "artisan_topdeck", "mine_trash", "mine_gain", "moneylender_trash",
                 "harbinger_topdeck", "vassal_play", "library_skip", "sentry_card", "sentry_top",
                 "bandit_trash", "bureaucrat_topdeck"):
        assert total[kind] > 0, f"{kind} never came up: {total}"


def test_released_portfolio_members_answer_like_their_genomes():
    import os
    from policy.registry import MODELS_DIR
    folder = os.path.join(MODELS_DIR, "portfolio-v1", "members")
    rnd = random.Random(9)
    for f in sorted(os.listdir(folder))[:8]:
        s = load_strategy(os.path.join(folder, f))
        for _ in range(4):
            kingdom = rnd.sample(NEW, 6) + rnd.sample([c for c in KINGDOM_CARDS if c not in NEW], 4)
            G.play(kingdom, [Spy(s), GAAgent(s)], rnd.getrandbits(31))
