"""Pygame-based graphical Dominion — play against evolved AI models.

Uses the same engine primitives as play.py (no duplicated game logic).
Separated from GA code; only imports engine, cards, strategy, and play helpers.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from enum import Enum, auto

import pygame

from cards import ALL_CARDS, CardType, KINGDOM_CARDS
from engine import (
    GameState, _new_player, default_supply, resolve_action,
    apply_action_effects, auto_play_treasures, buy_card, trash_card,
    play_moneylender, cleanup, is_game_over, count_vp,
    play_action_phase, play_buy_phase,
)
from play import discover_models, select_opponent
from strategy import Strategy, load_strategy, big_money_strategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 1280, 800
LEFT_W = 500          # your cards panel
RIGHT_W = 500         # supply panel
CENTER_W = SCREEN_W - LEFT_W - RIGHT_W  # info strip in between

CARD_W, CARD_H = 88, 125
CARD_W_SM = 65        # small cards in play area
CARD_H_SM = 92
CARD_GAP = 6

FPS = 30
AI_PAUSE_FRAMES = int(1.5 * FPS)

# Colors
BG_COLOR = (30, 30, 40)
PANEL_BG_LEFT = (35, 38, 50)
PANEL_BG_RIGHT = (38, 35, 50)
CENTER_BG = (25, 25, 35)
CARD_COLORS = {
    CardType.TREASURE: ((255, 220, 100), (180, 150, 40)),
    CardType.VICTORY: ((120, 200, 120), (50, 130, 50)),
    CardType.ACTION: ((200, 200, 215), (100, 100, 120)),
}
HIGHLIGHT_BORDER = (255, 255, 100)
DIM_OVERLAY = (0, 0, 0, 120)
DONE_BTN_COLOR = (70, 130, 70)
DONE_BTN_HOVER = (90, 160, 90)
TEXT_COLOR = (230, 230, 230)
TEXT_DIM = (150, 150, 170)
TEXT_DARK = (20, 20, 20)
MSG_COLOR = (200, 200, 220)
PHASE_COLORS = {
    "action": (100, 180, 255),
    "buy": (255, 200, 80),
    "chapel": (200, 100, 100),
    "throne": (180, 100, 220),
    "mine": (220, 160, 60),
    "ai": (150, 150, 170),
    "over": (255, 100, 100),
}
LABEL_COLOR = (160, 160, 180)

SUPPLY_ROWS = [
    ["Copper", "Silver", "Gold"],
    ["Estate", "Duchy", "Province"],
]


class GamePhase(Enum):
    HUMAN_ACTION = auto()
    HUMAN_CHAPEL = auto()
    HUMAN_THRONE_ROOM = auto()
    HUMAN_MINE = auto()
    HUMAN_BUY = auto()
    AI_TURN = auto()
    GAME_OVER = auto()


# ---------------------------------------------------------------------------
# Card drawing
# ---------------------------------------------------------------------------

def draw_card(surf: pygame.Surface, name: str, x: int, y: int,
              w: int, h: int, font: pygame.font.Font, font_sm: pygame.font.Font,
              highlighted: bool = False, dimmed: bool = False,
              count: int | None = None, hover: bool = False) -> pygame.Rect:
    """Draw a single card rectangle. Returns its rect for hit testing."""
    card = ALL_CARDS[name]
    bg, border = CARD_COLORS[card.card_type]
    rect = pygame.Rect(x, y, w, h)

    # Card body
    pygame.draw.rect(surf, bg, rect, border_radius=6)
    border_color = HIGHLIGHT_BORDER if (highlighted or hover) else border
    border_width = 3 if highlighted else (2 if hover else 1)
    pygame.draw.rect(surf, border_color, rect, border_width, border_radius=6)

    # Card name (wrap if needed)
    padding = 4
    name_y = y + 22
    words = name.split()
    if len(words) > 1 and w < 95:
        for i, word in enumerate(words):
            txt = font_sm.render(word, True, TEXT_DARK)
            surf.blit(txt, (x + padding, name_y + i * 14))
    else:
        txt = font_sm.render(name, True, TEXT_DARK)
        surf.blit(txt, (x + padding, name_y))

    # Cost badge (top-left circle)
    badge_r = 11
    badge_cx = x + badge_r + 3
    badge_cy = y + badge_r + 3
    pygame.draw.circle(surf, (60, 60, 80), (badge_cx, badge_cy), badge_r)
    cost_txt = font_sm.render(str(card.cost), True, (255, 255, 255))
    surf.blit(cost_txt, cost_txt.get_rect(center=(badge_cx, badge_cy)))

    # Card info (bottom area)
    info_y = y + h - 40
    info_lines = []
    if card.coins:
        info_lines.append(f"+${card.coins}")
    if card.cards_drawn:
        info_lines.append(f"+{card.cards_drawn} card{'s' if card.cards_drawn > 1 else ''}")
    if card.actions:
        info_lines.append(f"+{card.actions} act")
    if card.buys:
        info_lines.append(f"+{card.buys} buy")
    if card.vp and card.card_type == CardType.VICTORY:
        info_lines.append(f"{card.vp} VP")
    if card.special and not info_lines:
        info_lines.append(card.special)

    for i, line in enumerate(info_lines[:3]):
        txt = font_sm.render(line, True, TEXT_DARK)
        surf.blit(txt, (x + padding, info_y + i * 13))

    # Supply count (bottom-right)
    if count is not None:
        count_txt = font.render(str(count), True, TEXT_DARK)
        surf.blit(count_txt, (x + w - count_txt.get_width() - 5,
                              y + h - count_txt.get_height() - 4))

    # Dim overlay
    if dimmed:
        dim_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        dim_surf.fill(DIM_OVERLAY)
        surf.blit(dim_surf, (x, y))

    return rect


# ---------------------------------------------------------------------------
# GUI class
# ---------------------------------------------------------------------------

class DominionGUI:
    def __init__(self, opponent: Strategy,
                 kingdom: list[str] | None = None,
                 seed: int | None = None):
        if kingdom is None:
            kingdom = KINGDOM_CARDS
        if seed is None:
            seed = random.randint(0, 2**31)

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Dominion")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("dejavusansmono", 16)
        self.font_sm = pygame.font.SysFont("dejavusansmono", 12)
        self.font_lg = pygame.font.SysFont("dejavusansmono", 22, bold=True)
        self.font_title = pygame.font.SysFont("dejavusansmono", 28, bold=True)

        # Game state
        self.kingdom = kingdom
        self.opponent = opponent
        rng = random.Random(seed)
        self.supply = default_supply(kingdom, num_players=2)
        self.human = _new_player(random.Random(rng.randint(0, 2**31)))
        self.human.supply = self.supply
        self.ai = _new_player(random.Random(rng.randint(0, 2**31)))
        self.ai.supply = self.supply

        self.phase = GamePhase.HUMAN_ACTION
        self.round_num = 0
        self.messages: deque[str] = deque(maxlen=8)
        self.ai_pause_timer = 0

        # Sub-phase state
        self.chapel_trashed = 0
        self.throne_room_target: str | None = None
        self.throne_room_plays_left = 0

        # Build supply rows (treasures, victories, then kingdom in rows of 5)
        self.supply_rows = list(SUPPLY_ROWS)
        k = list(kingdom)
        while k:
            self.supply_rows.append(k[:5])
            k = k[5:]

        # Hit test areas (rebuilt each frame)
        self.supply_rects: list[tuple[pygame.Rect, str]] = []
        self.hand_rects: list[tuple[pygame.Rect, str, int]] = []
        self.done_rect: pygame.Rect | None = None

        self.mouse_pos = (0, 0)

        # Start first turn
        self._start_human_turn()

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def _start_human_turn(self):
        if is_game_over(self.human, turn_cap=40):
            self.phase = GamePhase.GAME_OVER
            return
        self.round_num += 1
        self.human.turn = self.round_num
        self.human.actions = 1
        self.human.buys = 1
        self.human.coins = 0
        self.phase = GamePhase.HUMAN_ACTION
        self.messages.append(f"--- Turn {self.round_num} ---")
        self._auto_advance_action()

    def _auto_advance_action(self):
        """If no action cards in hand, skip straight to buy phase."""
        if self.phase != GamePhase.HUMAN_ACTION:
            return
        action_cards = [c for c in self.human.hand
                        if ALL_CARDS[c].card_type == CardType.ACTION]
        if not action_cards or self.human.actions <= 0:
            self._transition_to_buy()

    def _transition_to_buy(self):
        treasures = auto_play_treasures(self.human)
        if treasures:
            counts: dict[str, int] = {}
            for t in treasures:
                counts[t] = counts.get(t, 0) + 1
            parts = [f"{n}x{c}" for c, n in sorted(counts.items())]
            self.messages.append(f"Treasures: {', '.join(parts)} = ${self.human.coins}")
        self.phase = GamePhase.HUMAN_BUY

    def _start_ai_turn(self):
        if is_game_over(self.ai, turn_cap=40):
            self.phase = GamePhase.GAME_OVER
            return
        self.ai.turn = self.round_num
        self.ai.actions = 1
        self.ai.buys = 1
        self.ai.coins = 0

        supply_before = dict(self.supply)

        play_action_phase(self.ai, self.opponent)
        play_buy_phase(self.ai, self.opponent)

        # Report what AI did
        played = [c for c in self.ai.play_area
                  if ALL_CARDS[c].card_type == CardType.ACTION]
        bought = []
        for name in self.supply:
            diff = supply_before[name] - self.supply[name]
            for _ in range(diff):
                bought.append(name)

        parts = []
        if played:
            parts.append(f"played {', '.join(played)}")
        if bought:
            parts.append(f"bought {', '.join(bought)}")
        self.messages.append(f"AI: {'; '.join(parts) if parts else 'did nothing'}")

        cleanup(self.ai)

        self.phase = GamePhase.AI_TURN
        self.ai_pause_timer = AI_PAUSE_FRAMES

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------

    def _handle_action_click(self, card_name: str):
        if ALL_CARDS[card_name].card_type != CardType.ACTION:
            return
        if card_name not in self.human.hand:
            return
        if self.human.actions <= 0:
            return

        newly_drawn = resolve_action(self.human, card_name)
        msg = f"Played {card_name}"
        if newly_drawn:
            msg += f", drew {', '.join(newly_drawn)}"
        self.messages.append(msg)

        special = ALL_CARDS[card_name].special
        if special == "chapel":
            self.chapel_trashed = 0
            self.phase = GamePhase.HUMAN_CHAPEL
        elif special == "moneylender":
            if play_moneylender(self.human):
                self.messages.append("Trashed Copper, +$3")
            else:
                self.messages.append("(no Copper to trash)")
            self._auto_advance_action()
        elif special == "throne_room":
            actions_in_hand = [c for c in self.human.hand
                               if ALL_CARDS[c].card_type == CardType.ACTION]
            if actions_in_hand:
                self.phase = GamePhase.HUMAN_THRONE_ROOM
                self.throne_room_target = None
                self.throne_room_plays_left = 2
            else:
                self.messages.append("(no action to double)")
                self._auto_advance_action()
        elif special == "mine":
            upgradeable = [c for c in self.human.hand
                           if ALL_CARDS[c].card_type == CardType.TREASURE
                           and c != "Gold"]
            if upgradeable:
                self.phase = GamePhase.HUMAN_MINE
            else:
                self.messages.append("(no treasure to upgrade)")
                self._auto_advance_action()
        else:
            self._auto_advance_action()

    def _handle_chapel_click(self, card_name: str):
        if card_name not in self.human.hand:
            return
        if self.chapel_trashed >= 4:
            return
        trash_card(self.human, card_name)
        self.chapel_trashed += 1
        self.messages.append(f"Trashed {card_name} ({self.chapel_trashed}/4)")
        if self.chapel_trashed >= 4 or not self.human.hand:
            self._finish_chapel()

    def _finish_chapel(self):
        if self.throne_room_plays_left > 0:
            self._throne_room_next_play()
        else:
            self.phase = GamePhase.HUMAN_ACTION
            self._auto_advance_action()

    def _handle_throne_room_click(self, card_name: str):
        if ALL_CARDS[card_name].card_type != CardType.ACTION:
            return
        if card_name not in self.human.hand:
            return

        self.throne_room_target = card_name
        self.human.hand.remove(card_name)
        self.human.play_area.append(card_name)
        self.messages.append(f"Throne Room -> {card_name} (x2)")

        self._throne_room_next_play()

    def _throne_room_next_play(self):
        target = self.throne_room_target
        if self.throne_room_plays_left <= 0 or target is None:
            self.throne_room_target = None
            self.phase = GamePhase.HUMAN_ACTION
            self._auto_advance_action()
            return

        newly_drawn = apply_action_effects(self.human, target)
        play_num = 3 - self.throne_room_plays_left
        msg = f"  [{play_num}/2] {target}"
        if newly_drawn:
            msg += f", drew {', '.join(newly_drawn)}"
        self.messages.append(msg)
        self.throne_room_plays_left -= 1

        special = ALL_CARDS[target].special
        if special == "chapel":
            self.chapel_trashed = 0
            self.phase = GamePhase.HUMAN_CHAPEL
        elif special == "moneylender":
            if play_moneylender(self.human):
                self.messages.append("Trashed Copper, +$3")
            else:
                self.messages.append("(no Copper to trash)")
            self._throne_room_next_play()
        elif special == "mine":
            upgradeable = [c for c in self.human.hand
                           if ALL_CARDS[c].card_type == CardType.TREASURE
                           and c != "Gold"]
            if upgradeable:
                self.phase = GamePhase.HUMAN_MINE
            else:
                self.messages.append("(no treasure to upgrade)")
                self._throne_room_next_play()
        else:
            self._throne_room_next_play()

    def _handle_mine_click(self, card_name: str):
        if ALL_CARDS[card_name].card_type != CardType.TREASURE:
            return
        if card_name == "Gold":
            return
        if card_name not in self.human.hand:
            return

        trashed_cost = ALL_CARDS[card_name].cost
        max_gain = trashed_cost + 3
        for gain_name in ["Gold", "Silver"]:
            gc = ALL_CARDS[gain_name].cost
            if gc <= max_gain and gc > trashed_cost and self.supply.get(gain_name, 0) > 0:
                trash_card(self.human, card_name)
                self.supply[gain_name] -= 1
                self.human.hand.append(gain_name)
                self.messages.append(f"Mine: {card_name} -> {gain_name}")
                break
        else:
            self.messages.append("(no valid upgrade)")

        if self.throne_room_plays_left > 0:
            self._throne_room_next_play()
        else:
            self.phase = GamePhase.HUMAN_ACTION
            self._auto_advance_action()

    def _handle_buy_click(self, card_name: str):
        if self.human.buys <= 0:
            return
        if card_name not in self.supply or self.supply[card_name] <= 0:
            return
        if ALL_CARDS[card_name].cost > self.human.coins:
            return
        buy_card(self.human, card_name)
        self.messages.append(f"Bought {card_name}")
        if self.human.buys <= 0:
            self._finish_buy()

    def _handle_done(self):
        if self.phase == GamePhase.HUMAN_ACTION:
            self._transition_to_buy()
        elif self.phase == GamePhase.HUMAN_BUY:
            self._finish_buy()
        elif self.phase == GamePhase.HUMAN_CHAPEL:
            self._finish_chapel()
        elif self.phase == GamePhase.HUMAN_MINE:
            if self.throne_room_plays_left > 0:
                self._throne_room_next_play()
            else:
                self.phase = GamePhase.HUMAN_ACTION
                self._auto_advance_action()

    def _finish_buy(self):
        cleanup(self.human)
        self._start_ai_turn()

    # ------------------------------------------------------------------
    # Rendering — left/right layout
    # ------------------------------------------------------------------

    def _render(self):
        self.screen.fill(BG_COLOR)
        self.supply_rects.clear()
        self.hand_rects.clear()

        self._render_left_panel()
        self._render_right_panel()
        self._render_center()

        pygame.display.flip()

    # --- Left panel: your hand + play area ---

    def _render_left_panel(self):
        panel = pygame.Rect(0, 0, LEFT_W, SCREEN_H)
        pygame.draw.rect(self.screen, PANEL_BG_LEFT, panel)

        pad = 15
        y = 10

        # Section label
        txt = self.font_lg.render("YOUR CARDS", True, LABEL_COLOR)
        self.screen.blit(txt, (pad, y))
        y += 32

        # Play area — cards played this turn
        played = self.human.play_area
        if played:
            txt = self.font_sm.render("In play:", True, TEXT_DIM)
            self.screen.blit(txt, (pad, y))
            y += 18
            x = pad
            for name in played:
                if x + CARD_W_SM > LEFT_W - pad:
                    x = pad
                    y += CARD_H_SM + 4
                draw_card(self.screen, name, x, y,
                          CARD_W_SM, CARD_H_SM, self.font_sm, self.font_sm)
                x += CARD_W_SM + 4
            y += CARD_H_SM + 12
        else:
            y += 10

        # Separator
        pygame.draw.line(self.screen, (60, 60, 80), (pad, y), (LEFT_W - pad, y))
        y += 12

        # Hand label
        hand_label = "Your hand:"
        if self.phase == GamePhase.HUMAN_ACTION:
            hand_label = "Click an action to play:"
        elif self.phase == GamePhase.HUMAN_CHAPEL:
            hand_label = f"Click to trash ({self.chapel_trashed}/4):"
        elif self.phase == GamePhase.HUMAN_THRONE_ROOM:
            hand_label = "Pick action to double:"
        elif self.phase == GamePhase.HUMAN_MINE:
            hand_label = "Pick treasure to upgrade:"
        elif self.phase == GamePhase.HUMAN_BUY:
            hand_label = "Your hand (buy from supply ->)"

        txt = self.font_sm.render(hand_label, True, LABEL_COLOR)
        self.screen.blit(txt, (pad, y))
        y += 20

        # Hand cards in a grid
        hand = self.human.hand
        if not hand:
            txt = self.font_sm.render("(empty)", True, TEXT_DIM)
            self.screen.blit(txt, (pad, y))
            return

        # Sort hand for display
        sorted_indices = sorted(range(len(hand)),
                                key=lambda i: (ALL_CARDS[hand[i]].card_type.value, hand[i]))

        cols = max(1, (LEFT_W - 2 * pad + CARD_GAP) // (CARD_W + CARD_GAP))
        x = pad
        col = 0
        for orig_i in sorted_indices:
            name = hand[orig_i]

            # Clickability per phase
            clickable = False
            if self.phase == GamePhase.HUMAN_ACTION:
                clickable = (ALL_CARDS[name].card_type == CardType.ACTION
                             and self.human.actions > 0)
            elif self.phase == GamePhase.HUMAN_CHAPEL:
                clickable = True
            elif self.phase == GamePhase.HUMAN_THRONE_ROOM:
                clickable = ALL_CARDS[name].card_type == CardType.ACTION
            elif self.phase == GamePhase.HUMAN_MINE:
                clickable = (ALL_CARDS[name].card_type == CardType.TREASURE
                             and name != "Gold")

            dimmed = not clickable and self.phase in (
                GamePhase.HUMAN_ACTION, GamePhase.HUMAN_CHAPEL,
                GamePhase.HUMAN_THRONE_ROOM, GamePhase.HUMAN_MINE,
            )

            r = pygame.Rect(x, y, CARD_W, CARD_H)
            hover = clickable and r.collidepoint(self.mouse_pos)

            rect = draw_card(self.screen, name, x, y, CARD_W, CARD_H,
                             self.font, self.font_sm,
                             highlighted=False, dimmed=dimmed, hover=hover)
            self.hand_rects.append((rect, name, orig_i))

            col += 1
            if col >= cols:
                col = 0
                x = pad
                y += CARD_H + CARD_GAP
            else:
                x += CARD_W + CARD_GAP

    # --- Right panel: supply ---

    def _render_right_panel(self):
        rx = SCREEN_W - RIGHT_W
        panel = pygame.Rect(rx, 0, RIGHT_W, SCREEN_H)
        pygame.draw.rect(self.screen, PANEL_BG_RIGHT, panel)

        pad = 15
        y = 10

        # Section label
        txt = self.font_lg.render("SUPPLY", True, LABEL_COLOR)
        self.screen.blit(txt, (rx + pad, y))
        y += 32

        for row in self.supply_rows:
            # Center each row within the right panel
            row_cards = [n for n in row if n in self.supply]
            if not row_cards:
                continue
            total_w = len(row_cards) * (CARD_W + CARD_GAP) - CARD_GAP
            x_start = rx + (RIGHT_W - total_w) // 2

            for i, name in enumerate(row_cards):
                x = x_start + i * (CARD_W + CARD_GAP)
                count = self.supply[name]

                clickable = (self.phase == GamePhase.HUMAN_BUY
                             and count > 0
                             and ALL_CARDS[name].cost <= self.human.coins
                             and self.human.buys > 0)
                dimmed = (self.phase == GamePhase.HUMAN_BUY and not clickable)

                hover = False
                if clickable:
                    r = pygame.Rect(x, y, CARD_W, CARD_H)
                    hover = r.collidepoint(self.mouse_pos)

                rect = draw_card(self.screen, name, x, y, CARD_W, CARD_H,
                                 self.font, self.font_sm,
                                 highlighted=False, dimmed=dimmed,
                                 count=count, hover=hover)
                self.supply_rects.append((rect, name))

            y += CARD_H + CARD_GAP

    # --- Center strip: info, messages, done button ---

    def _render_center(self):
        cx = LEFT_W
        panel = pygame.Rect(cx, 0, CENTER_W, SCREEN_H)
        pygame.draw.rect(self.screen, CENTER_BG, panel)
        # Vertical dividers
        pygame.draw.line(self.screen, (55, 55, 70), (cx, 0), (cx, SCREEN_H), 2)
        pygame.draw.line(self.screen, (55, 55, 70),
                         (cx + CENTER_W, 0), (cx + CENTER_W, SCREEN_H), 2)

        pad = 12
        y = 12

        # Title
        txt = self.font_lg.render("DOMINION", True, TEXT_COLOR)
        self.screen.blit(txt, (cx + pad, y))
        y += 30

        # Turn
        txt = self.font.render(f"Turn {self.round_num}", True, TEXT_COLOR)
        self.screen.blit(txt, (cx + pad, y))
        y += 24

        # Phase indicator
        phase_names = {
            GamePhase.HUMAN_ACTION: ("ACTION", "action"),
            GamePhase.HUMAN_BUY: ("BUY", "buy"),
            GamePhase.HUMAN_CHAPEL: ("CHAPEL", "chapel"),
            GamePhase.HUMAN_THRONE_ROOM: ("THRONE", "throne"),
            GamePhase.HUMAN_MINE: ("MINE", "mine"),
            GamePhase.AI_TURN: ("AI...", "ai"),
            GamePhase.GAME_OVER: ("OVER", "over"),
        }
        phase_label, phase_key = phase_names[self.phase]
        color = PHASE_COLORS.get(phase_key, TEXT_COLOR)
        txt = self.font_lg.render(phase_label, True, color)
        self.screen.blit(txt, (cx + pad, y))
        y += 30

        # Separator
        pygame.draw.line(self.screen, (60, 60, 80),
                         (cx + 8, y), (cx + CENTER_W - 8, y))
        y += 12

        # Resources
        for label, val in [("Actions", self.human.actions),
                           ("Buys", self.human.buys),
                           ("Coins", self.human.coins)]:
            txt = self.font.render(f"{label}: {val}", True, TEXT_COLOR)
            self.screen.blit(txt, (cx + pad, y))
            y += 22

        y += 8
        total = (len(self.human.deck) + len(self.human.hand)
                 + len(self.human.discard) + len(self.human.play_area))
        for label, val in [("Deck", len(self.human.deck)),
                           ("Discard", len(self.human.discard)),
                           ("Total", total)]:
            txt = self.font_sm.render(f"{label}: {val}", True, TEXT_DIM)
            self.screen.blit(txt, (cx + pad, y))
            y += 16

        y += 12

        # Done button
        if self.phase in (GamePhase.HUMAN_ACTION, GamePhase.HUMAN_BUY,
                          GamePhase.HUMAN_CHAPEL, GamePhase.HUMAN_MINE):
            btn_w = CENTER_W - 24
            btn_h = 36
            btn_rect = pygame.Rect(cx + 12, y, btn_w, btn_h)
            hover = btn_rect.collidepoint(self.mouse_pos)
            btn_color = DONE_BTN_HOVER if hover else DONE_BTN_COLOR
            pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)

            label = "Done"
            if self.phase == GamePhase.HUMAN_ACTION:
                label = "End Actions"
            elif self.phase == GamePhase.HUMAN_BUY:
                label = "End Buys"
            elif self.phase == GamePhase.HUMAN_CHAPEL:
                label = "Done Trash"
            elif self.phase == GamePhase.HUMAN_MINE:
                label = "Skip Mine"

            txt = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=btn_rect.center))
            self.done_rect = btn_rect
        else:
            self.done_rect = None

        y += 48

        # Separator
        pygame.draw.line(self.screen, (60, 60, 80),
                         (cx + 8, y), (cx + CENTER_W - 8, y))
        y += 10

        # Message log
        txt = self.font_sm.render("Log:", True, TEXT_DIM)
        self.screen.blit(txt, (cx + pad, y))
        y += 16
        for msg in list(self.messages)[-8:]:
            # Wrap long messages
            if len(msg) > 28:
                msg = msg[:27] + ".."
            txt = self.font_sm.render(msg, True, MSG_COLOR)
            self.screen.blit(txt, (cx + pad, y))
            y += 15

        # Game over display
        if self.phase == GamePhase.GAME_OVER:
            y = SCREEN_H - 160
            human_vp = count_vp(self.human)
            ai_vp = count_vp(self.ai)

            pygame.draw.line(self.screen, (60, 60, 80),
                             (cx + 8, y), (cx + CENTER_W - 8, y))
            y += 12

            txt = self.font.render(f"You: {human_vp} VP", True, (100, 255, 100))
            self.screen.blit(txt, (cx + pad, y))
            y += 22
            txt = self.font.render(f"AI:  {ai_vp} VP", True, (255, 130, 130))
            self.screen.blit(txt, (cx + pad, y))
            y += 28

            if human_vp > ai_vp:
                result, color = "YOU WIN!", (100, 255, 100)
            elif ai_vp > human_vp:
                result, color = "AI WINS", (255, 130, 130)
            else:
                result, color = "TIE", (200, 200, 100)
            txt = self.font_lg.render(result, True, color)
            self.screen.blit(txt, (cx + pad, y))
            y += 30

            txt = self.font_sm.render("R=restart", True, TEXT_DIM)
            self.screen.blit(txt, (cx + pad, y))
            y += 14
            txt = self.font_sm.render("Q=quit", True, TEXT_DIM)
            self.screen.blit(txt, (cx + pad, y))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r and self.phase == GamePhase.GAME_OVER:
                        self.__init__(self.opponent, self.kingdom)
                        continue

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            # AI turn timer
            if self.phase == GamePhase.AI_TURN:
                self.ai_pause_timer -= 1
                if self.ai_pause_timer <= 0:
                    self._start_human_turn()

            self._render()
            self.clock.tick(FPS)

        pygame.quit()

    def _handle_click(self, pos: tuple[int, int]):
        # Done button
        if self.done_rect and self.done_rect.collidepoint(pos):
            self._handle_done()
            return

        # Hand clicks (left panel)
        if self.phase in (GamePhase.HUMAN_ACTION, GamePhase.HUMAN_CHAPEL,
                          GamePhase.HUMAN_THRONE_ROOM, GamePhase.HUMAN_MINE):
            for rect, name, idx in reversed(self.hand_rects):
                if rect.collidepoint(pos):
                    if self.phase == GamePhase.HUMAN_ACTION:
                        self._handle_action_click(name)
                    elif self.phase == GamePhase.HUMAN_CHAPEL:
                        self._handle_chapel_click(name)
                    elif self.phase == GamePhase.HUMAN_THRONE_ROOM:
                        self._handle_throne_room_click(name)
                    elif self.phase == GamePhase.HUMAN_MINE:
                        self._handle_mine_click(name)
                    return

        # Supply clicks (right panel)
        if self.phase == GamePhase.HUMAN_BUY:
            for rect, name in self.supply_rects:
                if rect.collidepoint(pos):
                    self._handle_buy_click(name)
                    return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  DOMINION — Graphical Mode")
    print("=" * 50)

    models = discover_models()
    if not models:
        print("\nNo saved models found. Using Big Money as opponent.\n")
        opponent = big_money_strategy()
    else:
        opponent = select_opponent(models)

    gui = DominionGUI(opponent)
    gui.run()


if __name__ == "__main__":
    main()
