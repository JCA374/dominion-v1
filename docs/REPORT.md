# Evolving Dominion Strategies with a Genetic Algorithm

## Technical Report

---

## 1. Overview

This project uses a genetic algorithm (GA) to evolve purchasing and play strategies for a simplified implementation of the card game Dominion. The genome encodes phase-aware priority lists that govern which cards to buy, which actions to play, and when to transition between game phases. Strategies are evaluated through head-to-head 2-player games against a hall of fame of previously successful strategies.

The goal is to discover strategies that outperform naive heuristics (Big Money) by learning to build card engines — decks where action cards synergize to generate more purchasing power and faster game endings than simply buying treasures.

---

## 2. Dominion Game Rules (Simplified)

Dominion is a deck-building card game where players start with a deck of 10 weak cards (7 Copper, 3 Estate) and progressively acquire stronger cards from a shared supply. Each turn follows:

1. **Action Phase** — play action cards from hand (start with 1 action per turn)
2. **Buy Phase** — play treasures for coins, then buy cards (start with 1 buy per turn)
3. **Cleanup** — all cards in hand and play area go to discard pile, draw 5 new cards

The game ends when the Province pile (the primary victory card) is empty or when any 3 supply piles are empty. The player with the most victory points (VP) wins.

### Card Types in This Implementation

| Card | Cost | Type | Effect |
|------|------|------|--------|
| Copper | $0 | Treasure | +$1 |
| Silver | $3 | Treasure | +$2 |
| Gold | $6 | Treasure | +$3 |
| Estate | $2 | Victory | +1 VP |
| Duchy | $5 | Victory | +3 VP |
| Province | $8 | Victory | +6 VP |
| Village | $3 | Action (NT) | +1 card, +2 actions |
| Smithy | $4 | Action (T) | +3 cards |
| Market | $5 | Action (NT) | +1 card, +1 action, +1 buy, +$1 |
| Laboratory | $5 | Action (NT) | +2 cards, +1 action |
| Festival | $5 | Action (NT) | +2 actions, +1 buy, +$2 |
| Chapel | $2 | Action (T) | Trash up to 4 cards from hand |
| Throne Room | $4 | Action (T) | Play an action card from hand twice |
| Moneylender | $4 | Action (T) | Trash a Copper from hand, +$3 |
| Mine | $5 | Action (T) | Trash a treasure, gain one costing up to $3 more (to hand) |
| Merchant | $3 | Action (NT) | +1 card, +1 action; +$1 the first time Silver is played this turn |
| Council Room | $5 | Action (T) | +4 cards, +1 buy |
| Gardens | $4 | Victory | +1 VP per 10 cards in deck |

**NT = Non-terminal** (grants +actions, allowing further plays); **T = Terminal** (uses your action without granting more).

---

## 3. Strategy Genome

Each individual in the GA population is a `Strategy` object containing the following genes:

### 3.1 Buy Priority Lists (3 genes)

Phase-specific ordered lists of all buyable cards (treasures + victories + kingdom cards) plus a `PASS` sentinel. During the buy phase, the engine scans the list top-to-bottom and buys the first affordable, available card. If `PASS` is reached, buying stops (the strategy chooses to save its buy).

```
early_buy_priority: list[str]   # e.g., ["Chapel", "Silver", "Village", "PASS", ...]
mid_buy_priority: list[str]     # e.g., ["Gold", "Smithy", "Province", "PASS", ...]
late_buy_priority: list[str]    # e.g., ["Province", "Duchy", "Gold", "PASS", ...]
```

### 3.2 Action Priority Lists (6 genes)

Two sub-lists per phase for action play order:

- **Non-terminal priority**: actions that grant +actions (Village, Market, Laboratory, Festival, Merchant)
- **Terminal priority**: actions that consume your action without replacement (Smithy, Chapel, Throne Room, Moneylender, Mine)

Non-terminals always play before terminals (structural constraint — see Section 8.2).

```
early_nonterminal_priority: list[str]
early_terminal_priority: list[str]
mid_nonterminal_priority: list[str]
mid_terminal_priority: list[str]
late_nonterminal_priority: list[str]
late_terminal_priority: list[str]
```

### 3.3 Phase Transitions (2 genes)

```
early_to_mid_turn: int          # range [2, 15] — switch from early to mid after this turn
mid_to_late_provinces: int      # range [2, 8] — switch to late when <= this many Provinces remain
```

Phase selection logic:
```python
if turn <= early_to_mid_turn:
    phase = "early"
elif provinces_remaining > mid_to_late_provinces:
    phase = "mid"
else:
    phase = "late"
```

### 3.4 Buy Targets (variable-length gene)

```
buy_targets: dict[str, int]     # card_name -> max copies to own
```

Limits how many copies of a kingdom card the strategy will accumulate. E.g., `{"Smithy": 2, "Village": 3}` means never buy a 3rd Smithy or 4th Village.

### 3.5 Auxiliary Genes

| Gene | Type | Range | Purpose |
|------|------|-------|---------|
| `throne_room_priority` | `list[str]` | all actions except TR | Which action to double with Throne Room |
| `mine_trash_priority` | `list[str]` | ["Copper", "Silver"] | Which treasure to upgrade first with Mine |
| `province_max_coins` | `int` | [8, 18] or 99 | Skip Province if coins exceed this (buy Gold instead) |
| `duchy_max_coins` | `int` | [5, 18] or 99 | Skip Duchy if coins exceed this |

### 3.6 Total Genome Size

For a 10-card kingdom with 5 non-terminals and 5 terminals:
- 3 buy priorities: ~17 elements each = 51 ordered slots
- 6 action priorities: 5 elements each = 30 ordered slots
- 2 transition integers
- 1 throne room priority (9 elements)
- 1 mine priority (2 elements)
- 10 buy target integers
- 2 coin threshold integers
- **Total: ~107 evolvable parameters**

---

## 4. Genetic Operators

### 4.1 Population Initialization

Population of 60 individuals:
- 3 seed archetypes (Big Money, Engine, Gardens) for bootstrapping
- 57 fully randomized strategies

The seed archetypes provide the GA with known-good starting points. Big Money establishes a baseline; Engine demonstrates the Chapel-thin-deck pattern; Gardens represents an alternative VP accumulation path.

### 4.2 Selection: Tournament (k=3)

For each offspring slot, 3 individuals are sampled uniformly at random. The one with the highest fitness is selected as a parent. This provides moderate selection pressure while maintaining diversity.

### 4.3 Crossover: Order Crossover (OX)

Priority lists use Order Crossover, which preserves relative ordering from both parents:

1. Select two random crossover points `i, j` in parent 1
2. Copy the segment `p1[i:j+1]` into the child at the same positions
3. Fill remaining positions from parent 2, in order, skipping elements already placed
4. Handle PASS: keep only the first occurrence

This is applied independently to all 9 buy/action priority lists plus throne room and mine priorities. Transitions and buy targets use uniform crossover (randomly pick from either parent per gene).

### 4.4 Mutation

| Mutation Type | Rate | Operation |
|---------------|------|-----------|
| List swap | `rate` (18%) | Swap two random positions in a priority list |
| Insertion move | `rate × 0.3` | Remove element, reinsert at random position |
| PASS toggle | `rate / 5` | Insert or remove PASS from buy list |
| Transition jitter | `rate` | Add delta from {-3,-2,-1,-1,0,1,1,2,3}, clamp to valid range |
| Buy target jitter | `rate` per card | +/- 1, clamp [1, 10] |
| Target add/remove | `rate / 3` | Add new limit or remove existing |
| Coin threshold jitter | `rate` | +/- 1, clamp [8,18] or [5,18] |

### 4.5 Elitism

The top 2 individuals survive unchanged into the next generation.

### 4.6 Stagnation Recovery

If the best fitness doesn't improve for 30 consecutive generations:
- Mutation rate is boosted to `min(rate × 2.5, 0.5)`
- 5 fully random strategies are injected into the population
- This continues until improvement resumes

---

## 5. Fitness Evaluation

### 5.1 Game Simulation

Each strategy is evaluated through 2-player games. Each game seed is played **twice** — once with the strategy as Player 1 and once as Player 2 — to cancel first-player advantage. With `GAMES_PER_EVAL = 80`, each strategy plays 160 games per generation.

### 5.2 Hall of Fame

Instead of evaluating against a fixed opponent, strategies compete against a **hall of fame** — a dynamic set of up to 6 previously successful strategies:

- Initialized with Big Money as the baseline opponent
- A strategy is added to the hall when it achieves > 55% win rate against the current hall
- When the hall exceeds 6 members, the oldest non-Big-Money member is evicted
- After adding a new hall member, the "best fitness" tracker resets, forcing the GA to re-optimize against the expanded opponent pool

This creates an arms race: the GA must beat increasingly diverse opponents, preventing convergence to a strategy that only exploits one opponent's weakness.

### 5.3 Fitness Function (Blended)

```
fitness = 0.4 × win_rate + 0.4 × margin_norm + 0.2 × speed_norm
```

Where:
- **win_rate** ∈ [0, 1]: fraction of games won
- **margin_norm** = clamp((mean_vp_margin + 20) / 40, 0, 1): normalized VP margin. A strategy that wins by +10 VP on average gets margin_norm ≈ 0.75.
- **speed_norm** = clamp((25 - mean_turns) / 10, 0, 1): rewards faster game completion. Big Money at ~23 turns scores 0.2; a fast engine at 17 turns scores 0.8.

**Rationale for multi-objective blending:**

Pure win rate creates a degenerate fitness landscape where any strategy that wins >50% is nearly equivalent. The margin and speed components differentiate between:
- A strategy that barely wins 55% of games by 1 VP (fitness ≈ 0.22 + 0.21 + 0.04 = 0.47)
- A strategy that wins 55% by +8 VP in 18 turns (fitness ≈ 0.22 + 0.28 + 0.14 = 0.64)

This rewards **dominant** play, which is the signature of working engine strategies: they either fire and win decisively or stall and lose, but their wins are overwhelming.

---

## 6. Engine Architecture

### 6.1 Game Loop

```
for each turn:
    draw_cards(5)
    play_action_phase(state, strategy)
    play_buy_phase(state, strategy)     # includes auto-playing treasures
    cleanup(state)
    check game_over (provinces empty OR 3 piles empty OR turn >= 40)
```

### 6.2 Action Phase (Two-Tier)

```python
def play_action_phase(state, strategy):
    phase = get_current_phase(turn, provinces_remaining, transitions)
    nonterminal_priority, terminal_priority = get_action_priorities(strategy, phase)

    # Tier 1: play all non-terminals (these generate +actions)
    _play_action_tier(state, strategy, nonterminal_priority, phase)

    # Tier 2: play terminals (these consume actions)
    _play_action_tier(state, strategy, terminal_priority, phase)
```

Each tier scans its priority list repeatedly, playing the highest-priority card found in hand (if actions > 0), then rescanning from the top. This continues until no playable cards remain in that tier.

### 6.3 Buy Phase

```python
def play_buy_phase(state, strategy):
    auto_play_treasures(state)   # all treasures from hand, with Merchant bonus
    phase = get_current_phase(...)
    buy_priority = get_buy_priority(strategy, phase)

    while buys > 0:
        for card in buy_priority:
            if card == "PASS": stop buying
            if card is affordable and in supply:
                if card exceeds buy_target: skip
                if Province and coins > province_max_coins: skip
                if Duchy and coins > duchy_max_coins: skip
                buy card, break and rescan
        if nothing bought: break
```

### 6.4 Special Card Handlers

- **Chapel**: follows phase-specific trash priority until STOP or max reached
- **Throne Room**: picks highest-priority action from hand, plays it twice (including special effects)
- **Mine**: trashes lowest-priority treasure from hand, gains the best affordable upgrade
- **Moneylender**: trashes one Copper for +$3
- **Merchant**: grants +$1 on the first Silver played each turn

---

## 7. C Engine

### 7.1 Motivation

Fitness evaluation is the primary computational bottleneck: 60 strategies × 160 games × ~6 hall opponents = ~57,600 games per generation. The pure Python engine (`engine.py`) uses string card names for all lookups, dictionary-based supply tracking, and list operations — all of which carry significant interpreter overhead.

### 7.2 Architecture

The C engine (`dominion.c`) is a complete reimplementation of the game loop optimized for batch evaluation:

- **Integer card IDs** instead of string names — all comparisons are integer equality
- **Fixed-size arrays** with counts (`deck[80]` + `deck_n`) instead of Python lists — no allocation
- **Flat strategy representation** — priority lists serialized as -1 terminated int arrays, eliminating dict/object overhead
- **xorshift64 PRNG** — faster than Python's Mersenne Twister, seeded from Python
- **Batch API** — `play_games_batch()` runs all seeds for a matchup in a single ctypes call, eliminating per-game FFI overhead

The C engine handles only the strategy-driven game loop (action phase, buy phase, cleanup). Interactive play modes (`play.py`, `gui.py`, `trace.py`) continue to use the Python engine, which provides the low-level primitives they need for user-driven decisions.

### 7.3 Integration

```
cards.py (CARD_ID, data arrays)  →  c_bridge.py (ctypes)  →  dominion.so
                                          ↑
fitness.py: if USE_C_ENGINE → evaluate_vs_opponent_c()
            else            → evaluate_vs_opponent() [Python fallback]
```

- `cards.py` defines integer card IDs and flat data arrays (cost, coins, VP, etc.)
- `c_bridge.py` loads `dominion.so` via ctypes, serializes `Strategy` objects to flat int arrays, and provides `evaluate_vs_opponent_c()` as a drop-in replacement
- `fitness.py` auto-detects the C engine and uses it for all GA evaluation; falls back to Python if `dominion.so` is unavailable
- The final evaluation in `main.py` uses the Python engine (`need_deck=True`) to capture deck composition data for the summary

### 7.4 Performance

Benchmark: 5 strategies × 500 seeds × 2 seats = 5,000 games

| Engine | Time | Per game |
|--------|------|----------|
| Python (single-threaded) | ~4.5s | ~0.9ms |
| C (single-threaded) | ~0.09s | ~18us |
| **Speedup** | **~50x** | |

The C engine is fast enough that Python multiprocessing is no longer needed for evaluation. A single-threaded C evaluation completes faster than 8-worker Python evaluation, while avoiding process spawn overhead and strategy serialization costs.

### 7.5 Parallelization (Python Fallback)

When the C engine is unavailable, the system falls back to Python's `ProcessPoolExecutor` with configurable workers (default 8):

```python
def evaluate_population_vs_hall(population, seed_list, kingdom, hall, workers):
    if workers <= 1:
        return [evaluate_vs_hall(s, ...) for s in population]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, population))
```

Each worker independently evaluates one strategy against the entire hall.

---

## 8. Simplifications and Design Decisions

### 8.1 Three-Phase System (Early/Mid/Late)

**What:** The game is divided into exactly three phases with hardcoded transition logic.

**Why:** A continuous strategy space (e.g., "at turn T with N provinces, buy X") would require the GA to optimize over a combinatorial explosion of conditions. Three phases reduce this to 3 priority lists × 2 transition parameters, making the search space tractable while still capturing the fundamental strategic shift from "build economy" (early) → "build engine" (mid) → "buy victory points" (late).

**Limitation:** Real Dominion experts make more granular decisions (e.g., "buy Duchy when exactly 4 Provinces remain but only if I'm behind"). The 3-phase model cannot express conditional mid-game pivots.

### 8.2 Structural Non-Terminal Before Terminal Split

**What:** Non-terminal actions (those granting +actions) always play before terminal actions. The GA controls the ordering *within* each tier but cannot interleave them.

**Why:** Playing Village before Smithy is **always** correct in base Dominion — Village grants +2 actions, enabling Smithy to be played without terminating the action phase. Without this constraint, the GA wastes generations rediscovering this trivially optimal ordering. By encoding it structurally, the GA only optimizes the *interesting* decisions: which non-terminals before others, and which terminals before others.

**Limitation:** Edge case with Throne Room — since it's classified as a terminal, it plays after all non-terminals. If it doubles a Village, the extra actions arrive too late to play more non-terminals. This is an acceptable trade-off: the structural guarantee that Villages always play before Smithys is more valuable than the rare Throne Room + Village combo.

### 8.3 Hardcoded Chapel Trashing

**What:** Chapel always trashes Estates first, then Coppers, then stops. This is not evolved.

**Why:** Trashing Estates and Coppers with Chapel is a solved problem in Dominion theory — it is *always* correct in the early game. The starting deck has 7 Copper ($1 each) and 3 Estate (0 VP value early, dead draws). Removing them makes the deck smaller and denser, so you draw your good cards more often.

When the GA was allowed to evolve trash order, it consistently found that *not trashing* (STOP first) was a local optimum because:
1. Trashing requires buying Chapel (cost $2, doesn't directly generate money)
2. The benefit of a thin deck only materializes 5-10 turns later
3. The GA's evaluation noise (80 games) couldn't reliably detect this delayed payoff

By hardcoding the correct trash order, we eliminate a known trap in the fitness landscape.

**Phase-specific behavior:**
- Early/Mid: trash Estate > Copper > STOP (thin the deck)
- Late: STOP (never trash — you're buying Provinces/Duchies, not optimizing draws)

### 8.4 Fixed Chapel Max Trash = 4

**What:** Chapel always trashes the maximum 4 cards per play.

**Why:** In real Dominion, Chapel can trash 0-4 cards per play (player's choice). However, in the early game, trashing the maximum is almost always correct — you want your weak cards gone as fast as possible. Allowing the GA to evolve this parameter led to convergence on `chapel_max_trash = 1`, which makes Chapel nearly useless (takes 7+ turns to remove all junk instead of 2).

### 8.5 No Conditional Buy Logic

**What:** Buy decisions are purely priority-list-based. There is no gene for "buy X only if condition Y is true."

**Why:** Conditional buy logic (e.g., "buy Duchy only if <= 5 Provinces remain") would require the GA to evolve both the condition and the action, exponentially expanding the search space. The phase system partially addresses this (late phase = buy green), and the buy targets provide simple caps. More complex conditionals are deferred to future work.

**Attempted and removed:** Buy ratios ("`buy X only if owned_X / owned_Y < ratio`") were implemented and removed because:
- References to other action cards created circular dependencies that blocked all purchases
- References to treasure cards encoded "don't buy actions without economy" — the opposite of engine play
- The feature added search space complexity without providing a useful fitness signal

### 8.6 Priority-List Action Play (No Hand Analysis)

**What:** The action phase plays cards in a fixed priority order. It does not analyze the current hand composition.

**Why:** A hand-aware strategy (e.g., "play Smithy only if I have < 7 cards") would require condition-action pairs in the genome, making crossover and mutation far more complex. The priority-list approach is simple, composable with OX crossover, and sufficient: if Chapel is higher priority than Smithy, Chapel plays first when both are in hand.

### 8.7 No Opponent Modeling

**What:** Strategies have no awareness of what the opponent is doing.

**Why:** In 2-player Dominion, opponent modeling matters primarily for attack cards and race conditions (when to start buying Provinces). Since this implementation has no attack cards and the "when to green" decision is captured by the phase transition gene, opponent-blind play is a reasonable simplification.

### 8.8 Shared Game Seeds (Paired Play)

**What:** Each game seed is played twice — as Player 1 and as Player 2 — to cancel first-player advantage.

**Why:** In Dominion, Player 1 has a measurable advantage (first to buy Province, first to end game). Without paired play, the GA might evolve strategies that appear strong but only win due to going first. Playing both sides with the same shuffle ensures wins are due to strategy quality, not turn order luck.

### 8.9 Turn Cap (40)

**What:** Games are forcibly ended at turn 40.

**Why:** Without a turn cap, degenerate strategies (e.g., never buying Provinces) would produce infinite games. The 40-turn cap ensures evaluation always terminates. In practice, well-played games end in 18-25 turns; a game reaching turn 40 indicates both strategies are dysfunctional.

### 8.10 Supply Sizes

**What:** Fixed supply sizes (Province: 12, Kingdom: 10, etc.) assuming 2 players.

**Why:** Standard Dominion supply for 2 players uses 8 Provinces. This implementation uses 12 (the 3-4 player count) to give engine strategies more time to develop before the game ends. With only 8 Provinces, Big Money can end the game before an engine has time to pay off its setup investment.

---

## 9. Current Configuration

```python
POP_SIZE        = 60
GENERATIONS     = 5000
GAMES_PER_EVAL  = 80      # × 2 (paired play) = 160 games per strategy per gen
TOURNAMENT_SIZE = 3
ELITE_COUNT     = 2
MUTATION_RATE   = 0.18
SEED            = 137
VP_MARGIN_WEIGHT = 0.4
SPEED_WEIGHT    = 0.2
HALL_MAX_SIZE   = 6
HALL_ADD_THRESHOLD = 0.55
WORKERS         = 8

# Kingdom: 10 cards selected for training
KINGDOM = ["Village", "Smithy", "Market", "Laboratory", "Festival",
           "Chapel", "Throne Room", "Mine", "Moneylender", "Merchant"]
```

### Fitness Formula

```
fitness = 0.4 × win_rate + 0.4 × margin_norm + 0.2 × speed_norm

margin_norm = clamp((mean_vp_margin + 20) / 40, 0, 1)
speed_norm  = clamp((25 - mean_turns) / 10, 0, 1)
```

| Scenario | Win Rate | VP Margin | Turns | Fitness |
|----------|----------|-----------|-------|---------|
| Big Money vs Big Money | 0.50 | 0 | 23 | 0.20 + 0.20 + 0.04 = 0.44 |
| Weak engine, close wins | 0.55 | +3 | 22 | 0.22 + 0.23 + 0.06 = 0.51 |
| Strong engine, dominant | 0.60 | +10 | 18 | 0.24 + 0.30 + 0.14 = 0.68 |
| Perfect engine | 0.70 | +15 | 16 | 0.28 + 0.35 + 0.18 = 0.81 |

---

## 10. Known Challenges and Prior Failures

### 10.1 Big Money Convergence

The GA historically converges to Big Money variants (Gold > Silver > Province) because:
1. Big Money is a strong baseline that wins ~50% against most random strategies
2. Engine strategies require 4-5 cards working in concert — random mutation rarely assembles them simultaneously
3. A half-built engine (Smithys without Villages) is *worse* than Big Money
4. The hall of fame, initially containing only Big Money, rewards strategies that beat Big Money — and Big Money beats Big Money 50%

**Mitigations applied:**
- Engine seed archetype provides a working engine template from generation 0
- Structural action split eliminates the "play Village after Smithy" failure mode
- Hardcoded Chapel trashing ensures deck thinning always works
- VP margin + speed bonuses reward the *style* of wins engines produce
- Stagnation recovery injects diversity when the population converges

### 10.2 Removed Feature: Buy Ratios

Buy ratios (`"buy Smithy only if owned_Smithy / owned_Village < 1.0"`) were implemented to express card synergy constraints. They were removed because:
- Action-to-action references created circular deadlocks
- Treasure references encoded anti-engine heuristics
- The GA consistently evolved ratios that blocked all action purchases (converging to Big Money by a different mechanism)

---

## 11. Output and Monitoring

### 11.1 CSV Log

Each generation logs: generation number, best/mean/worst fitness, phase transitions, top-3 buy priorities per phase, mean turns.

### 11.2 Best Model Snapshot

When a new best is found, the system saves:
- `best_model/strategy.json` — full genome as JSON
- `best_model/summary.txt` — human-readable strategy description
- `best_model/buy_heatmap.png` — visualization of buy priority positions across phases

### 11.3 Human-Readable Summary

The `summarize()` function generates a plain-English description of the evolved strategy, including:
- Win rate and average game length
- Phase timing boundaries
- Top buy priorities per phase with card type classification
- Action play order (non-terminals then terminals)
- Chapel trashing behavior
- Buy targets and constraints
- Average final deck composition

---

## 12. Future Directions

1. **Attack cards** (Witch, Militia) — force interactive play and reward faster cycling; requires curse supply, discard decisions, and 3-player game loop
2. **3+ player games** — expand the engine to support multiplayer with shared supply
3. **Conditional buy genes** — "buy Duchy when <= N Provinces remain"
4. **Opponent-aware fitness** — reward strategies that adapt to opponent type
5. **Larger card pools** — evolve kingdom selection alongside strategy
6. **Neural network policy** — replace priority lists with a learned value function over game state
7. **Coevolution** — evolve two populations against each other instead of a fixed hall
