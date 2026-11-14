"""Pygame play mode: you against an evolved model, on the shared engine (engine/).

The engine runs as a generator and stops at every decision. The model answers its own at once
(with a short pause so you can follow); yours are shown as buttons in the middle panel, and
the matching cards in your hand or the supply light up and can be clicked too.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, deque

import pygame

import core  # noqa: F401
from core.agent import GAAgent
from core.strategy import Strategy, big_money_strategy
from dominion_engine import Game
from dominion_engine import cards as K
from dominion_engine import game as G
from play.terminal import PROMPTS, discover_models, model_kingdom, select_opponent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 1280, 800
LEFT_W, RIGHT_W = 500, 500
CENTER_X = LEFT_W
CENTER_W = SCREEN_W - LEFT_W - RIGHT_W

CARD_W, CARD_H = 88, 125
CARD_W_SM, CARD_H_SM = 65, 92
CARD_GAP = 6
FPS = 30
AI_PAUSE_FRAMES = int(0.6 * FPS)

BG_COLOR = (30, 30, 40)
PANEL_BG_LEFT = (35, 38, 50)
PANEL_BG_RIGHT = (38, 35, 50)
CENTER_BG = (25, 25, 35)
COLORS = {"treasure": ((255, 220, 100), (180, 150, 40)),
          "victory": ((120, 200, 120), (50, 130, 50)),
          "curse": ((190, 140, 200), (110, 60, 130)),
          "action": ((200, 200, 215), (100, 100, 120))}
HIGHLIGHT_BORDER = (255, 255, 100)
DIM_OVERLAY = (0, 0, 0, 120)
BTN_COLOR, BTN_HOVER = (70, 100, 140), (90, 130, 180)
DONE_BTN_COLOR, DONE_BTN_HOVER = (70, 130, 70), (90, 160, 90)
TEXT_COLOR, TEXT_DIM, TEXT_DARK = (230, 230, 230), (150, 150, 170), (20, 20, 20)
LABEL_COLOR = (160, 160, 180)

HAND_KINDS = {G.PLAY_ACTION, G.THRONE, G.CHAPEL_TRASH, G.MONEYLENDER_TRASH, G.MINE_TRASH,
              G.REMODEL_TRASH, G.ARTISAN_TOPDECK, G.CELLAR_DISCARD, G.POACHER_DISCARD,
              G.MILITIA_DISCARD, G.BUREAUCRAT_TOPDECK}
SUPPLY_KINDS = {G.BUY, G.MINE_GAIN, G.REMODEL_GAIN, G.WORKSHOP_GAIN, G.ARTISAN_GAIN}


def _style(c: int) -> str:
    card = K.CARDS[c]
    if card.is_action:
        return "action"
    if card.is_treasure:
        return "treasure"
    return "victory" if card.is_victory else "curse"


def draw_card(surf, c: int, x: int, y: int, w: int, h: int, font, font_sm,
              highlighted=False, dimmed=False, count=None, hover=False) -> pygame.Rect:
    card = K.CARDS[c]
    bg, border = COLORS[_style(c)]
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surf, bg, rect, border_radius=6)
    pygame.draw.rect(surf, HIGHLIGHT_BORDER if (highlighted or hover) else border, rect,
                     3 if highlighted else (2 if hover else 1), border_radius=6)
    words = card.name.split()
    for i, word in enumerate(words if len(words) > 1 and w < 95 else [card.name]):
        surf.blit(font_sm.render(word, True, TEXT_DARK), (x + 4, y + 22 + i * 14))
    pygame.draw.circle(surf, (60, 60, 80), (x + 14, y + 14), 11)
    cost = font_sm.render(str(card.cost), True, (255, 255, 255))
    surf.blit(cost, cost.get_rect(center=(x + 14, y + 14)))
    info = []
    if card.coins:
        info.append(f"+${card.coins}")
    if card.cards:
        info.append(f"+{card.cards} card{'s' if card.cards > 1 else ''}")
    if card.actions:
        info.append(f"+{card.actions} act")
    if card.buys:
        info.append(f"+{card.buys} buy")
    if card.vp:
        info.append(f"{card.vp} VP")
    for i, line in enumerate(info[:3]):
        surf.blit(font_sm.render(line, True, TEXT_DARK), (x + 4, y + h - 40 + i * 13))
    if count is not None:
        t = font.render(str(count), True, TEXT_DARK)
        surf.blit(t, (x + w - t.get_width() - 5, y + h - t.get_height() - 4))
    if dimmed:
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill(DIM_OVERLAY)
        surf.blit(dim, (x, y))
    return rect


class DominionGUI:
    def __init__(self, opponent: Strategy, kingdom: list[str] | None = None, seed: int | None = None):
        kingdom = kingdom or model_kingdom(opponent)
        seed = random.randint(0, 2**31) if seed is None else seed
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Dominion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusansmono", 16)
        self.font_sm = pygame.font.SysFont("dejavusansmono", 12)
        self.font_lg = pygame.font.SysFont("dejavusansmono", 22, bold=True)

        self.game = Game(kingdom, seed)
        self.ai = GAAgent(opponent)
        self.supply_rows = [[K.COPPER, K.SILVER, K.GOLD, K.CURSE], [K.ESTATE, K.DUCHY, K.PROVINCE]]
        ks = sorted(self.game.kingdom, key=lambda c: (K.CARDS[c].cost, K.NAME[c]))
        self.supply_rows += [ks[:5], ks[5:]]
        self.messages: deque[str] = deque(maxlen=14)
        self.gen = self.game.run()
        self.pending: G.Decision | None = None
        self.result = None
        self.ai_timer = 0
        self.mouse = (0, 0)
        self.targets: list[tuple[pygame.Rect, int]] = []
        self._advance(None)

    # -- driving the engine --------------------------------------------------

    def _advance(self, answer):
        """Send `answer` (None = start) and run until the human must decide or the AI pauses."""
        try:
            d = next(self.gen) if answer is None else self.gen.send(answer)
            while d.player == 1:
                ans = self.ai.decide(self.game, d)
                self._log_ai(d, ans)
                d = self.gen.send(ans)
                if d.player == 1 and d.kind in (G.PLAY_ACTION, G.BUY):
                    self.pending, self.ai_timer = d, AI_PAUSE_FRAMES
                    return
            self.pending = d
        except StopIteration as stop:
            self.pending, self.result = None, stop.value

    def _log_ai(self, d, ans):
        if ans >= 0 and d.kind == G.PLAY_ACTION:
            self.messages.append(f"AI plays {K.NAME[ans]}")
        elif ans >= 0 and d.kind == G.BUY:
            self.messages.append(f"AI buys {K.NAME[ans]}")
        elif d.kind == G.SENTRY_CARD:
            self.messages.append(f"AI Sentry: {['trash', 'discard', 'keep'][ans]} {K.NAME[d.subject]}")
        elif ans >= 0 and d.kind != G.PLAY_ACTION:
            self.messages.append(f"AI {PROMPTS[d.kind].split(':')[0]}: {K.NAME[ans]}")

    def _answer(self, ans: int):
        d = self.pending
        if d.kind == G.PLAY_ACTION and ans >= 0:
            self.messages.append(f"You play {K.NAME[ans]}")
        elif d.kind == G.BUY and ans >= 0:
            self.messages.append(f"You buy {K.NAME[ans]}")
        self._advance(ans)

    def _tick(self):
        if self.pending and self.pending.player == 1:
            self.ai_timer -= 1
            if self.ai_timer <= 0:
                d = self.pending
                ans = self.ai.decide(self.game, d)
                self._log_ai(d, ans)
                self._advance(ans)

    # -- drawing -------------------------------------------------------------

    def _human_options(self) -> set[int]:
        d = self.pending
        return set(d.options) if d and d.player == 0 and d.kind != G.SENTRY_CARD else set()

    def _draw(self):
        s, g = self.screen, self.game
        s.fill(BG_COLOR)
        pygame.draw.rect(s, PANEL_BG_LEFT, (0, 0, LEFT_W, SCREEN_H))
        pygame.draw.rect(s, CENTER_BG, (CENTER_X, 0, CENTER_W, SCREEN_H))
        pygame.draw.rect(s, PANEL_BG_RIGHT, (SCREEN_W - RIGHT_W, 0, RIGHT_W, SCREEN_H))
        self.targets = []
        opts = self._human_options()
        d = self.pending
        me, ai = g.players

        # left: you
        s.blit(self.font_lg.render("You", True, TEXT_COLOR), (12, 10))
        s.blit(self.font.render(f"actions {g.actions if g.current == 0 else '-'}  buys "
                                f"{g.buys if g.current == 0 else '-'}  ${g.coins if g.current == 0 else '-'}",
                                True, TEXT_DIM), (12, 40))
        s.blit(self.font_sm.render("Hand", True, LABEL_COLOR), (12, 70))
        hand_kind = d is not None and d.player == 0 and d.kind in HAND_KINDS
        for i, c in enumerate(sorted(me.hand)):
            x, y = 12 + (i % 5) * (CARD_W + CARD_GAP), 90 + (i // 5) * (CARD_H + CARD_GAP)
            hit = hand_kind and c in opts
            r = draw_card(s, c, x, y, CARD_W, CARD_H, self.font, self.font_sm, highlighted=hit,
                          dimmed=hand_kind and not hit,
                          hover=hit and pygame.Rect(x, y, CARD_W, CARD_H).collidepoint(self.mouse))
            if hit:
                self.targets.append((r, c))
        play_y = 90 + 2 * (CARD_H + CARD_GAP) + 10
        s.blit(self.font_sm.render("In play", True, LABEL_COLOR), (12, play_y))
        shown = g.players[g.current].play
        for i, c in enumerate(shown[:14]):
            draw_card(s, c, 12 + (i % 7) * (CARD_W_SM + 4), play_y + 18 + (i // 7) * (CARD_H_SM + 4),
                      CARD_W_SM, CARD_H_SM, self.font, self.font_sm)
        info_y = SCREEN_H - 70
        s.blit(self.font.render(f"Deck {len(me.deck)}  Discard {len(me.discard)}  "
                                f"Turn {me.turns + 1}", True, TEXT_DIM), (12, info_y))
        s.blit(self.font.render(f"AI: hand {len(ai.hand)}  cards {len(ai.all_cards())}",
                                True, TEXT_DIM), (12, info_y + 24))

        # right: supply
        x0 = SCREEN_W - RIGHT_W + 10
        s.blit(self.font_lg.render(f"Supply  ({g.provinces_left} Provinces)", True, TEXT_COLOR), (x0, 10))
        supply_kind = d is not None and d.player == 0 and d.kind in SUPPLY_KINDS
        for row_i, row in enumerate(self.supply_rows):
            for col_i, c in enumerate(row):
                x, y = x0 + col_i * (CARD_W + CARD_GAP), 45 + row_i * (CARD_H + CARD_GAP)
                hit = supply_kind and c in opts
                r = draw_card(s, c, x, y, CARD_W, CARD_H, self.font, self.font_sm, highlighted=hit,
                              dimmed=g.supply[c] <= 0 or (supply_kind and not hit), count=g.supply[c],
                              hover=hit and pygame.Rect(x, y, CARD_W, CARD_H).collidepoint(self.mouse))
                if hit:
                    self.targets.append((r, c))

        # center: prompt, buttons, log
        cx = CENTER_X + 10
        y = 12
        if self.result:
            r = self.result
            verdict = "Tie" if len(r.winners) == 2 else ("You win!" if r.winners == [0] else "AI wins")
            for line in [verdict, f"You {r.scores[0]} VP", f"AI  {r.scores[1]} VP", "", "Esc to quit"]:
                s.blit(self.font_lg.render(line, True, TEXT_COLOR), (cx, y))
                y += 30
        elif d is not None and d.player == 0:
            subject = f" [{K.NAME[d.subject]}]" if d.kind in (G.VASSAL_PLAY, G.LIBRARY_SKIP, G.SENTRY_CARD) else ""
            for line in self._wrap(PROMPTS[d.kind] + subject, 24):
                s.blit(self.font.render(line, True, HIGHLIGHT_BORDER), (cx, y))
                y += 20
            y += 6
            labels = (["trash", "discard", "put back"] if d.kind == G.SENTRY_CARD
                      else [K.NAME[o] for o in d.options])
            for o, label in zip(d.options, labels):
                y = self._button(cx, y, label, o, BTN_COLOR, BTN_HOVER)
            if d.may_pass:
                y = self._button(cx, y, "Done / none", -1, DONE_BTN_COLOR, DONE_BTN_HOVER)
        else:
            s.blit(self.font.render("AI is playing...", True, TEXT_DIM), (cx, y))
            y += 24
        y = max(y + 10, SCREEN_H - 14 * 18 - 10)
        for m in self.messages:
            s.blit(self.font_sm.render(m[:36], True, TEXT_DIM), (cx, y))
            y += 18
        pygame.display.flip()

    def _button(self, x, y, label, value, color, hover) -> int:
        r = pygame.Rect(x, y, CENTER_W - 20, 26)
        pygame.draw.rect(self.screen, hover if r.collidepoint(self.mouse) else color, r, border_radius=5)
        self.screen.blit(self.font.render(label, True, TEXT_COLOR), (x + 8, y + 4))
        self.targets.append((r, value))
        return y + 30

    @staticmethod
    def _wrap(text, width):
        lines, line = [], ""
        for w in text.split():
            if line and len(line) + len(w) + 1 > width:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        return lines + [line]

    # -- main loop -----------------------------------------------------------

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if ev.type == pygame.MOUSEMOTION:
                    self.mouse = ev.pos
                if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.pending
                        and self.pending.player == 0):
                    for rect, value in self.targets:
                        if rect.collidepoint(ev.pos):
                            self._answer(value)
                            break
            self._tick()
            self._draw()
            self.clock.tick(FPS)


def main():
    models = discover_models()
    opponent = select_opponent(models) if models else big_money_strategy()
    DominionGUI(opponent).run()
    sys.exit(0)


if __name__ == "__main__":
    main()
