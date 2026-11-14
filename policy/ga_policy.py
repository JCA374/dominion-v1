"""Policy for the GA's evolved strategy.json (kind "ga-strategy").

Each protocol question is asked as the engine's own decision, answered by the same GAAgent
that played in training, on a stand-in game built from the observation. Cards the strategy
never learned are never bought.
"""
from __future__ import annotations

from collections import Counter

import core  # noqa: F401
from core.agent import GAAgent
from core.strategy import Strategy, get_current_phase, load_strategy
from dominion_engine import Game
from dominion_engine import cards as K
from dominion_engine import game as G
from policy.base import Policy
from policy.protocol import NO_REQUEST_YET

_NO_BUY = {K.ID[c] for c in NO_REQUEST_YET}


def _distinct(cards) -> tuple[int, ...]:
    return tuple(sorted(set(cards)))


class GAPolicy(Policy):
    kind = "ga-strategy"

    def __init__(self, manifest: dict, strategy: Strategy | None = None):
        super().__init__(manifest)
        self.strategy = strategy or load_strategy(manifest["path"])
        self.agent = GAAgent(self.strategy)

    def phase(self, obs: dict) -> str:
        return get_current_phase(obs["turn"], obs["provinces_left"], self.strategy.transitions)

    def _game(self, obs: dict) -> Game:
        """A game in which seat 0 is the observing player."""
        g = Game([c for c in obs["kingdom"] if c in K.ID], seed=1)
        for c in obs["empty_piles"]:
            if c in K.ID:
                g.supply[K.ID[c]] = 0
        g.supply[K.PROVINCE] = obs["provinces_left"]
        p = g.players[0]
        hand = [K.ID[c] for c in obs["hand"] if c in K.ID]
        rest = Counter({K.ID[c]: n for c, n in obs["owned"].items() if c in K.ID})
        rest.subtract(Counter(hand))
        p.hand, p.deck, p.discard, p.play = hand, list(rest.elements()), [], []
        p.turns = obs["turn"] - 1
        g.actions, g.buys, g.coins = obs["actions"], obs["buys"], obs["coins"]
        return g

    def _ask(self, g: Game, kind: int, options, may_pass: bool, subject: int = -1) -> int:
        return self.agent.decide(g, G.Decision(kind, 0, tuple(options), may_pass, subject))

    # -- action phase --------------------------------------------------------

    def play_action(self, obs: dict) -> str | None:
        """The engine's choice, skipping cards that would do nothing now."""
        if obs["actions"] < 1:
            return None
        g = self._game(obs)
        hand = g.players[0].hand
        opts = [c for c in _distinct(hand) if K.CARDS[c].is_action and not self._useless(c, g)]
        if not opts:
            return None
        c = self._ask(g, G.PLAY_ACTION, opts, True)
        return None if c < 0 else K.NAME[c]

    def _useless(self, card: int, g: Game) -> bool:
        if card == K.MONEYLENDER:
            return K.COPPER not in g.players[0].hand
        if card == K.THRONE_ROOM:
            return self._throne(g) < 0
        return False

    def throne_target(self, obs: dict) -> str | None:
        t = self._throne(self._game(obs))
        return None if t < 0 else K.NAME[t]

    def _throne(self, g: Game) -> int:
        rest = list(g.players[0].hand)
        if K.THRONE_ROOM in rest:
            rest.remove(K.THRONE_ROOM)
        opts = [c for c in _distinct(rest) if K.CARDS[c].is_action]
        return self._ask(g, G.THRONE, opts, True) if opts else -1

    def trash(self, obs: dict) -> list[str]:
        g = self._game(obs)
        hand, out = g.players[0].hand, []
        while len(out) < 4 and hand:
            c = self._ask(g, G.CHAPEL_TRASH, _distinct(hand), True, len(out))
            if c < 0:
                break
            hand.remove(c)
            out.append(K.NAME[c])
        return out

    def discard(self, obs: dict, keep: int) -> list[str]:
        """Discard down to `keep` cards (the engine's Militia choice)."""
        extra = [c for c in obs["hand"] if c not in K.ID]      # unknown cards go first
        n = len(obs["hand"]) - keep
        if n <= 0:
            return []
        if len(extra) >= n:
            return extra[:n]
        g = self._game(obs)
        hand, out = g.players[0].hand, list(extra)
        while len(out) < n:
            c = self._ask(g, G.MILITIA_DISCARD, _distinct(hand), False, n - len(out))
            hand.remove(c)
            out.append(K.NAME[c])
        return out

    # -- buy phase -----------------------------------------------------------

    def buy(self, obs: dict) -> list[str]:
        g = self._game(obs)
        out = []
        while g.buys > 0:
            opts = [c for c in range(K.NUM_CARDS)
                    if g.supply[c] > 0 and K.CARDS[c].cost <= g.coins and c not in _NO_BUY]
            c = self._ask(g, G.BUY, opts, True) if opts else -1
            if c < 0:
                break
            g.supply[c] -= 1
            g.players[0].discard.append(c)
            g.coins -= K.CARDS[c].cost
            g.buys -= 1
            out.append(K.NAME[c])
        return out
