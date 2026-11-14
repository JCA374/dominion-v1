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

    # -- the cards' own choices (protocol 2) -----------------------------------
    # Each is the engine's decision sequence for that card (engine/dominion_engine/game.py
    # Game._play), asked of the same GAAgent, on the stand-in game.

    def _names(self, ids):
        return [K.NAME[c] for c in ids]

    def _gain_choice(self, g: Game, kind: int, max_cost: int, subject: int = -1,
                     treasure: bool = False) -> int:
        opts = [c for c in g._gainable(max_cost) if not treasure or K.CARDS[c].is_treasure]
        return self._ask(g, kind, opts, False, subject) if opts else -1

    def cellar(self, obs: dict) -> list[str]:
        g = self._game(obs)
        hand, out = g.players[0].hand, []
        while hand:
            c = self._ask(g, G.CELLAR_DISCARD, _distinct(hand), True, len(out))
            if c < 0:
                break
            hand.remove(c)
            g.players[0].discard.append(c)
            out.append(c)
        return self._names(out)

    def poacher(self, obs: dict, n: int) -> list[str]:
        g = self._game(obs)
        hand, out = g.players[0].hand, []
        for j in range(min(n, len(hand))):
            c = self._ask(g, G.POACHER_DISCARD, _distinct(hand), False, n - j)
            hand.remove(c)
            out.append(c)
        return self._names(out)

    def remodel(self, obs: dict) -> dict:
        g = self._game(obs)
        hand = g.players[0].hand
        if not hand:
            return {"trash": None, "gain": None}
        t = self._ask(g, G.REMODEL_TRASH, _distinct(hand), False)
        hand.remove(t)
        gain = self._gain_choice(g, G.REMODEL_GAIN, K.CARDS[t].cost + 2, t)
        return {"trash": K.NAME[t], "gain": K.NAME[gain] if gain >= 0 else None}

    def workshop(self, obs: dict) -> str | None:
        gain = self._gain_choice(self._game(obs), G.WORKSHOP_GAIN, 4)
        return K.NAME[gain] if gain >= 0 else None

    def artisan(self, obs: dict) -> dict:
        g = self._game(obs)
        gain = self._gain_choice(g, G.ARTISAN_GAIN, 5)
        hand = g.players[0].hand
        if gain >= 0:
            g.supply[gain] -= 1
            hand.append(gain)
        top = self._ask(g, G.ARTISAN_TOPDECK, _distinct(hand), False) if hand else -1
        return {"gain": K.NAME[gain] if gain >= 0 else None, "topdeck": K.NAME[top] if top >= 0 else None}

    def mine(self, obs: dict) -> dict:
        g = self._game(obs)
        hand = g.players[0].hand
        ts = _distinct([c for c in hand if K.CARDS[c].is_treasure])
        t = self._ask(g, G.MINE_TRASH, ts, True) if ts else -1
        if t < 0:
            return {"trash": None, "gain": None}
        hand.remove(t)
        gain = self._gain_choice(g, G.MINE_GAIN, K.CARDS[t].cost + 3, t, treasure=True)
        return {"trash": K.NAME[t], "gain": K.NAME[gain] if gain >= 0 else None}

    def moneylender(self, obs: dict) -> bool:
        g = self._game(obs)
        if K.COPPER not in g.players[0].hand:
            return False
        return self._ask(g, G.MONEYLENDER_TRASH, (K.COPPER,), True) == K.COPPER

    def harbinger(self, obs: dict, discard_pile: list[str]) -> str | None:
        g = self._game(obs)
        pile = [K.ID[c] for c in discard_pile if c in K.ID]
        if not pile:
            return None
        g.players[0].discard = pile
        c = self._ask(g, G.HARBINGER_TOPDECK, _distinct(pile), True)
        return K.NAME[c] if c >= 0 else None

    def vassal(self, obs: dict, card: str) -> bool:
        c = K.ID.get(card, -1)
        if c < 0 or not K.CARDS[c].is_action:
            return False
        return self._ask(self._game(obs), G.VASSAL_PLAY, (c,), True, c) == c

    def library(self, obs: dict, card: str) -> bool:
        """True: set the drawn action card aside (skip it)."""
        c = K.ID.get(card, -1)
        if c < 0 or not K.CARDS[c].is_action:
            return False
        return self._ask(self._game(obs), G.LIBRARY_SKIP, (c,), True, c) == c

    def sentry(self, obs: dict, cards: list[str]) -> dict:
        g = self._game(obs)
        out = {"trash": [], "discard": [], "keep": []}
        kept = []
        for name in cards:
            c = K.ID.get(name, -1)
            if c < 0:
                out["keep"].append(name)
                continue
            a = self._ask(g, G.SENTRY_CARD, (G.SENTRY_TRASH, G.SENTRY_DISCARD, G.SENTRY_KEEP), False, c)
            if a == G.SENTRY_TRASH:
                out["trash"].append(name)
            elif a == G.SENTRY_DISCARD:
                out["discard"].append(name)
            else:
                kept.append(c)
        if len(kept) == 2 and kept[0] != kept[1]:
            top = self._ask(g, G.SENTRY_TOP, _distinct(kept), False)
            kept.remove(top)
            kept.append(top)
        out["keep"] += self._names(reversed(kept))          # top of the deck first
        return out

    def bandit(self, obs: dict, cards: list[str]) -> str | None:
        """Attacked by Bandit: which of the revealed treasures (not Copper) is trashed."""
        cand = _distinct([K.ID[c] for c in cards if c in K.ID and K.CARDS[K.ID[c]].is_treasure
                          and K.ID[c] != K.COPPER])
        if not cand:
            return None
        if len(cand) == 1:
            return K.NAME[cand[0]]
        return K.NAME[self._ask(self._game(obs), G.BANDIT_TRASH, cand, False)]

    def bureaucrat(self, obs: dict) -> str | None:
        """Attacked by Bureaucrat: the victory card put back on the deck."""
        g = self._game(obs)
        vs = _distinct([c for c in g.players[0].hand if K.CARDS[c].is_victory])
        return K.NAME[self._ask(g, G.BUREAUCRAT_TOPDECK, vs, False)] if vs else None

