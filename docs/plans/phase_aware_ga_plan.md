# Phase-Aware Genetic Algorithm for Simplified Dominion

## Goal

Build a GA that evolves **time-dependent** strategies for a simplified version of
Dominion. Each evolved strategy holds different buy priorities for different phases
of the game (early / mid / late), and the GA also evolves *when* those phases switch.

The implementation must make it **easy to watch strategies evolve** — clear
logging, readable strategy summaries, and plots showing how fitness and behaviour
change across generations.

---

## 1. Simplified Game Rules

Keep the engine small and deterministic where possible.

### Cards

| Card     | Type     | Cost | Effect                                  |
|----------|----------|------|-----------------------------------------|
| Copper   | Treasure | 0    | +1 coin                                 |
| Silver   | Treasure | 3    | +2 coins                                |
| Gold     | Treasure | 6    | +3 coins                                |
| Estate   | Victory  | 2    | 1 VP                                     |
| Duchy    | Victory  | 5    | 3 VP                                     |
| Province | Victory  | 8    | 6 VP                                     |
| Village  | Action   | 3    | +1 card, +2 actions                      |
| Smithy   | Action   | 4    | +3 cards                                 |
| Market   | Action   | 5    | +1 card, +1 action, +1 buy, +1 coin      |
| Laboratory | Action | 5    | +2 cards, +1 action                      |
| Festival | Action   | 5    | +2 actions, +1 buy, +2 coins             |
| Chapel   | Action   | 2    | Trash up to 4 cards from hand            |

(Start with this fixed 6-card kingdom: Village, Smithy, Market, Laboratory,
Festival, Chapel — plus the basic treasures/victory cards. The 8-card kingdom
selection can come later.)

### Turn structure

1. **Action phase** — play action cards while actions remain, following the
   strategy's action priority.
2. **Buy phase** — auto-play all treasures, then buy cards following the
   phase-appropriate buy priority while buys/coins remain.
3. **Cleanup** — discard hand + play area, draw 5 new cards (reshuffle discard
   into deck when empty).

### Starting deck

7 Copper + 3 Estate.

### Game end

Either Province stack empty, or any 3 supply stacks empty, or a turn cap
(e.g. 40 turns) to guarantee termination.

### Scoring (single-player optimisation)

Default: total **VP at game end**. Optionally minimise turns-to-N-Provinces for a
speed metric. Keep these two as selectable fitness functions.

---

## 2. Strategy Representation (the genome)

A strategy is split into **three phases**, each with its own buy priority, plus
genes controlling phase transitions.

```python
Strategy = {
    "phases": {
        "early": {"buy_priority": ["Chapel", "Silver", "Village", ...]},
        "mid":   {"buy_priority": ["Gold", "Smithy", "Silver", ...]},
        "late":  {"buy_priority": ["Province", "Duchy", "Gold", ...]},
    },
    "action_priority": ["Village", "Festival", "Laboratory", "Market",
                        "Smithy", "Chapel"],   # global, evolved
    "transitions": {
        "early_to_mid_turn": 6,     # gene: integer turn number
        "mid_to_late_provinces": 5, # gene: enter late when <= N Provinces left
    },
}
```

### Phase selection logic at each turn

```
if turn <= early_to_mid_turn:                  phase = "early"
elif provinces_remaining > mid_to_late_provinces: phase = "mid"
else:                                          phase = "late"
```

### Buy priority semantics

A buy priority is an ordered list of card names. During the buy phase, walk the
list; buy the first affordable card whose supply isn't empty; repeat while buys
remain. A `"PASS"` token allowed in the list lets a strategy deliberately skip
buying (useful early game).

### Genes summary

- 3 × ordered buy-priority lists (one per phase)
- 1 × ordered action-priority list
- `early_to_mid_turn` ∈ [2, 15]
- `mid_to_late_provinces` ∈ [0, 8]

---

## 3. Genetic Operators

### Initialisation

Random permutations of the card pool for each priority list; random integers
(within bounds) for transition genes. Seed a few known archetypes into the
initial population (Big Money, simple engine) so evolution has good genes to
recombine and we can watch them compete.

### Selection

Tournament selection (size 3–5). Keep elitism: top 1–2 individuals copied
unchanged each generation.

### Crossover

- **Buy/action priority lists** → Order Crossover (OX), applied independently
  per phase list. (OX preserves valid permutations.)
- **Transition genes** → blend/average of the two parents, rounded to integer,
  or random pick between the two parent values.

### Mutation

- **Priority lists** → swap two positions (rate ~0.1 per list), occasionally
  insert/remove a `"PASS"` token.
- **Transition genes** → ±1 jitter with small probability, clamped to bounds.

---

## 4. Fitness Evaluation

```
fitness(strategy) = mean(VP over M simulated games)
```

- `M` games per evaluation (start at 50; bump to 200 for final runs) to reduce
  variance from shuffles.
- Use a fixed RNG seed list per generation so all individuals face the same
  shuffles → fairer comparison, less noise.
- Provide an optional **head-to-head** fitness vs a fixed Big Money benchmark
  (win rate), since beating Big Money is the classic Dominion sanity check.

---

## 5. Observability (the important part)

This is what makes the project easy to understand and the evolution visible.

### Per-generation logging (console + CSV)

For each generation record:
- generation number
- best / mean / worst fitness
- best strategy's transition genes
- best strategy's top-3 buys per phase
- average game length of the best strategy

Write to `evolution_log.csv` for later plotting.

### Human-readable strategy printout

A `describe(strategy)` function that prints something like:

```
=== Strategy (fitness 38.4 VP) ===
EARLY (turns 1-6):   Chapel > Silver > Village > PASS
MID   (until 5 Prov): Gold > Smithy > Silver > Market
LATE  (<= 5 Prov):    Province > Duchy > Gold > Estate
Actions: Village > Festival > Lab > Market > Smithy > Chapel
```

### Plots (matplotlib, saved as PNGs)

1. **Fitness over generations** — best/mean/worst lines.
2. **Transition genes over generations** — how `early_to_mid_turn` and
   `mid_to_late_provinces` drift as the population converges.
3. **Buy-mix heatmap** — for the best strategy, fraction of each card bought per
   phase, showing how priorities differ by phase.
4. (Optional) **Single-game trace** — deck composition (treasure/action/VP count)
   turn by turn for the champion strategy.

### Reproducibility

Single `SEED` constant at top of file controls all RNG. Print it at start.

---

## 6. Suggested File Layout

```
dominion_ga/
├── cards.py          # Card definitions + effects
├── engine.py         # GameState, turn loop, game-end checks
├── strategy.py       # Strategy genome, phase selection, describe()
├── ga.py             # population, selection, crossover, mutation, run loop
├── fitness.py        # simulate M games, scoring functions
├── plotting.py       # the 4 plots above
└── main.py           # config + run + save logs/plots
```

Keep each file small and readable. Favour plain dictionaries/dataclasses over
deep class hierarchies so the genome stays inspectable.

---

## 7. Config Knobs (top of main.py)

```python
POP_SIZE        = 60
GENERATIONS     = 100
GAMES_PER_EVAL  = 50
TOURNAMENT_SIZE = 4
ELITE_COUNT     = 2
MUTATION_RATE   = 0.1
FITNESS_MODE    = "vp"      # or "vs_big_money" or "speed"
SEED            = 42
KINGDOM         = ["Village","Smithy","Market","Laboratory","Festival","Chapel"]
```

---

## 8. Build Order (for the implementer)

1. `cards.py` + `engine.py` — get a single game running with a hardcoded
   Big Money strategy; verify VP totals look sane.
2. `strategy.py` — genome + phase selection + `describe()`.
3. `fitness.py` — simulate M games; confirm Big Money scores ~stable.
4. `ga.py` — selection/crossover/mutation/run loop with CSV logging.
5. `plotting.py` — the four plots.
6. `main.py` — wire it together; run 100 generations; confirm evolved strategy
   beats the seeded Big Money baseline.

### Milestone check

Evolved best strategy should clearly **outperform Big Money** on the chosen
fitness metric, and the transition-gene plot should show the population settling
on a sensible greening point (late phase kicking in when a few Provinces remain).

---

## 9. Later Extensions (out of scope for v1)

- Expand to full 8-card kingdom + kingdom selection as part of the genome.
- 2-player co-evolution (strategies evolve against each other, not Big Money).
- Deck-size-aware heuristic genes (coins/card density triggers) layered on top
  of the phase system.
- More phases (4–5) or continuous phase weighting instead of hard cutoffs.
