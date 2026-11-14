"""The evolved genome as an engine agent: answers every engine decision from a Strategy.

The genome decides what to buy and play, Chapel/Throne Room/Mine targets and when to switch
phases. Every other choice (attacks, and cards added for Steam parity: Cellar, Harbinger,
Vassal, Workshop, Bureaucrat, Poacher, Remodel, Library, Sentry, Artisan, Bandit) is a fixed
heuristic built on the same priority lists.

c/ga_agent.c is the C twin (training runs there); tests/test_agent_parity.py checks they play
identical games. Change both together.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import core  # noqa: F401
from dominion_engine import cards as K
from dominion_engine import game as G

from core.strategy import get_current_phase

if TYPE_CHECKING:
    from core.strategy import Strategy

PASS, STOP = -2, -3   # list markers (C: GA_PASS, GA_STOP)
_PHASE = {"early": 0, "mid": 1, "late": 2}


def _ids(names: list[str]) -> list[int]:
    out = []
    for n in names:
        if n == "PASS":
            out.append(PASS)
        elif n == "STOP":
            out.append(STOP)
        elif n in K.ID:
            out.append(K.ID[n])
    return out


def is_terminal(c: int) -> bool:
    """Terminal for the density rule: an action without +Actions, except Throne Room."""
    card = K.CARDS[c]
    return card.is_action and card.actions == 0 and c != K.THRONE_ROOM


class GAAgent:
    def __init__(self, strategy: Strategy):
        s = self.strategy = strategy
        self.buy = [_ids(s.early_buy_priority), _ids(s.mid_buy_priority), _ids(s.late_buy_priority)]
        self.chapel = [_ids(s.early_chapel_trash), _ids(s.mid_chapel_trash), _ids(s.late_chapel_trash)]
        self.nt = _ids(s.nonterminal_priority)
        self.t = _ids(s.terminal_priority)
        self.throne = _ids(s.throne_room_priority)
        self.mine = _ids(s.mine_trash_priority)
        self.targets = {K.ID[c]: n for c, n in s.buy_targets.items() if c in K.ID}
        self.actions_played = 0

    # -- helpers ------------------------------------------------------------------------

    def phase(self, game: G.Game, player: int) -> int:
        return _PHASE[get_current_phase(game.turn_number(player), game.provinces_left,
                                        self.strategy.transitions)]

    def keep_rank(self, c: int) -> int:
        """How much an action is worth keeping: non-terminals above terminals, list order."""
        if c in self.nt:
            return 60 - self.nt.index(c)
        if c in self.t:
            return 30 - self.t.index(c)
        return 0

    def discard_rank(self, c: int, hand: list[int]) -> int:
        """Lower = discard first: Curse, Estate, Copper, Duchy, Gardens, actions, Silver, Gold, Province."""
        fixed = {K.CURSE: 0, K.ESTATE: 10, K.COPPER: 20, K.DUCHY: 30, K.GARDENS: 35,
                 K.SILVER: 800, K.GOLD: 850, K.PROVINCE: 900}
        if c in fixed:
            return fixed[c]
        if K.CARDS[c].is_action:
            return (500 if hand.count(c) > 1 else 700) - self.keep_rank(c)
        return 400

    def worst(self, options, hand: list[int]) -> int:
        return min(options, key=lambda c: self.discard_rank(c, hand))

    def allowed(self, c: int, owned: list[int]) -> bool:
        """Buy targets and the terminal-density rule."""
        if c in self.targets and owned[c] >= self.targets[c]:
            return False
        if is_terminal(c):
            terminals = sum(owned[x] for x in range(K.NUM_CARDS) if is_terminal(x))
            capacity = sum(owned[x] * max(0, K.CARDS[x].actions - 1)
                           for x in range(K.NUM_CARDS) if K.CARDS[x].is_action)
            if terminals >= capacity + self.strategy.terminal_slack:
                return False
        return True

    def owned(self, game: G.Game, player: int) -> list[int]:
        counts = [0] * K.NUM_CARDS
        for c in game.players[player].all_cards():
            counts[c] += 1
        return counts

    def gain(self, game: G.Game, d: G.Decision) -> int:
        """A free gain: first allowed card of the buy list (PASS ignored), else the priciest non-Curse."""
        owned = self.owned(game, d.player)
        for c in self.buy[self.phase(game, d.player)]:
            if c >= 0 and c in d.options and self.allowed(c, owned):
                return c
        best = d.options[0]
        for c in d.options:
            if c != K.CURSE and (best == K.CURSE or K.CARDS[c].cost > K.CARDS[best].cost):
                best = c
        return best

    # -- the decisions ------------------------------------------------------------------

    def decide(self, game: G.Game, d: G.Decision) -> int:
        k, opts, me = d.kind, d.options, d.player
        hand = game.players[me].hand

        if k == G.PLAY_ACTION:
            for c in self.nt + self.t:
                if c in opts:
                    self.actions_played += 1
                    return c
            return -1
        if k == G.BUY:
            owned = self.owned(game, me)
            for c in self.buy[self.phase(game, me)]:
                if c == PASS:
                    return -1
                if c in opts and self.allowed(c, owned):
                    return c
            return -1
        if k == G.THRONE:
            return next((c for c in self.throne if c in opts), -1)
        if k == G.CHAPEL_TRASH:
            if d.subject >= min(self.strategy.chapel_max_trash, 4):
                return -1
            for c in self.chapel[self.phase(game, me)]:
                if c == STOP:
                    return -1
                if c in opts:
                    return c
            return -1
        if k == G.MONEYLENDER_TRASH:
            return K.COPPER
        if k == G.MINE_TRASH:
            for t in self.mine:
                if t in opts and any(game.supply[g] > 0 and
                                     K.CARDS[t].cost < K.CARDS[g].cost <= K.CARDS[t].cost + 3
                                     for g in (K.GOLD, K.SILVER, K.COPPER)):
                    return t
            return -1
        if k == G.MINE_GAIN:
            for g in (K.GOLD, K.SILVER, K.COPPER):
                if g in opts and K.CARDS[g].cost > K.CARDS[d.subject].cost:
                    return g
            return opts[-1]
        if k in (G.MILITIA_DISCARD, G.POACHER_DISCARD, G.BUREAUCRAT_TOPDECK):
            return self.worst(opts, hand)
        if k == G.BANDIT_TRASH:
            return opts[0]                      # Silver before Gold
        if k == G.CELLAR_DISCARD:
            return next((c for c in opts if not K.CARDS[c].is_action and not K.CARDS[c].is_treasure), -1)
        if k == G.HARBINGER_TOPDECK:
            for c in [K.GOLD] + self.nt + self.t + [K.SILVER]:
                if c in opts:
                    return c
            return -1
        if k == G.VASSAL_PLAY:
            return d.subject if (d.subject in self.nt or d.subject in self.t) else -1
        if k == G.LIBRARY_SKIP:
            return d.subject if game.actions == 0 else -1
        if k == G.SENTRY_CARD:
            trash = self.chapel[self.phase(game, me)]
            trash = trash[:trash.index(STOP)] if STOP in trash else trash
            if d.subject in trash:
                return G.SENTRY_TRASH
            card = K.CARDS[d.subject]
            if not card.is_action and not card.is_treasure:
                return G.SENTRY_DISCARD
            return G.SENTRY_KEEP
        if k == G.SENTRY_TOP:
            a, b = opts
            return b if self.discard_rank(b, []) > self.discard_rank(a, []) else a
        if k == G.REMODEL_TRASH:
            late = self.phase(game, me) == 2
            if late and K.GOLD in opts and game.supply[K.PROVINCE] > 0:
                return K.GOLD
            if K.CURSE in opts:
                return K.CURSE
            if K.ESTATE in opts and not late:
                return K.ESTATE
            return self.worst(opts, hand)
        if k in (G.REMODEL_GAIN, G.WORKSHOP_GAIN, G.ARTISAN_GAIN):
            return self.gain(game, d)
        if k == G.ARTISAN_TOPDECK:
            for c in self.nt + self.t:
                if c in opts:
                    return c
            for c in opts:
                if K.CARDS[c].is_action:
                    return c
            return K.SILVER if K.SILVER in opts else self.worst(opts, hand)
        raise ValueError(f"unknown decision {d}")
