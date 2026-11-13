"""Policy for the GA's evolved strategy.json (kind "ga-strategy").

Decisions reuse the training engine (play_buy_phase, militia_discard, the
priority lists) on a stand-in game state, so a real game is played exactly as
in training. Cards the strategy never learned are never bought.
"""
from __future__ import annotations

from core.cards import ALL_CARDS, CardType
from core.engine import GameState, militia_discard, play_buy_phase
from core.strategy import Strategy, get_chapel_trash_priority, get_current_phase, load_strategy
from policy.base import Policy
from policy.cards import BASE


class GAPolicy(Policy):
    kind = "ga-strategy"

    def __init__(self, manifest: dict, strategy: Strategy | None = None):
        super().__init__(manifest)
        self.strategy = strategy or load_strategy(manifest["path"])

    def phase(self, obs: dict) -> str:
        return get_current_phase(obs["turn"], obs["provinces_left"], self.strategy.transitions)

    # -- action phase --------------------------------------------------------

    def play_action(self, obs: dict) -> str | None:
        """Non-terminals first, then terminals; skip cards that would do nothing."""
        hand = obs["hand"]
        if obs["actions"] < 1:
            return None
        for prio in (self.strategy.nonterminal_priority, self.strategy.terminal_priority):
            for card in prio:
                if card in hand and not self._useless(card, hand):
                    return card
        return None

    def _useless(self, card: str, hand: list[str]) -> bool:
        if card == "Moneylender":
            return "Copper" not in hand
        if card == "Throne Room":
            return self._throne_target(hand) is None
        return False

    def throne_target(self, obs: dict) -> str | None:
        return self._throne_target(obs["hand"])

    def _throne_target(self, hand: list[str]) -> str | None:
        rest = list(hand)
        if "Throne Room" in rest:
            rest.remove("Throne Room")
        for card in self.strategy.throne_room_priority:
            if card in rest and ALL_CARDS[card].card_type == CardType.ACTION:
                return card
        return None

    def trash(self, obs: dict) -> list[str]:
        trash, rest = [], list(obs["hand"])
        for card in get_chapel_trash_priority(self.strategy, self.phase(obs)):
            if card == "STOP":
                break
            while card in rest and len(trash) < min(self.strategy.chapel_max_trash, 4):
                rest.remove(card)
                trash.append(card)
        return trash

    def discard(self, obs: dict, keep: int) -> list[str]:
        """Discard down to `keep` cards (same order as the training engine)."""
        hand = obs["hand"]
        n = len(hand) - keep
        if n <= 0:
            return []
        extra = [c for c in hand if c not in ALL_CARDS]  # unknown cards go first
        if len(extra) >= n:
            return extra[:n]
        # militia_discard empties a hand down to 3; pad so it stops at `keep`
        st = GameState(hand=[c for c in hand if c in ALL_CARDS] + ["Province"] * (3 - keep))
        militia_discard(st, self.strategy)
        return (extra + st.discard)[:n]

    # -- buy phase -----------------------------------------------------------

    def buy(self, obs: dict) -> list[str]:
        known = [c for c in obs["kingdom"] if c in ALL_CARDS]
        supply = {c: 10 for c in BASE + known}
        supply["Province"] = obs["provinces_left"]
        for c in obs["empty_piles"]:
            supply[c] = 0
        deck = [c for c, n in obs["owned"].items() for _ in range(n)]
        st = GameState(supply=supply, coins=obs["coins"], buys=obs["buys"],
                       turn=obs["turn"], deck=deck)
        play_buy_phase(st, self.strategy)
        return list(st.discard)
