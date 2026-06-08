# Dominion GA — Quick Reference

## How It Works

A genetic algorithm evolves Dominion strategies by playing thousands of games.
The GA is the core of this project — it produces `best_model/strategy.json`,
which the play modes then load and use as an AI opponent.

```
GA (train.py)  --->  best_model/strategy.json  --->  play.py / gui.py / trace.py
    |                                                     |
    |  evolves strategies using C engine                  |  loads strategy, plays
    |  (fitness, selection, crossover, mutation)           |  against it using Python engine
```

All modes share the same game logic — the GA trains on the C engine (~50x faster),
and the play modes run on the Python engine, so gameplay is identical.

---

## 1. Training — the GA (`train.py`)

### Build the C engine first (one-time)

The GA uses a fast C game engine for evaluation (~50x faster than Python).
Build it before training:

```bash
make                        # builds dominion.so
```

If `dominion.so` is missing, the system auto-builds it on first run (requires `gcc`).
Falls back to the Python engine if compilation fails.

### Run training

```bash
python train.py              # fresh training run (10000 generations)
python train.py --continue   # continue from best_model/, keeps generation numbering
```

Config is at the top of `ga/main.py` — population size, mutation rate, kingdom cards, etc.

**Outputs:**
- `best_model/strategy.json` — the evolved strategy (used by all play modes)
- `best_model/summary.txt` — human-readable description
- `best_model/buy_heatmap.png` — what the strategy buys per phase
- `evolution_log.csv` — fitness over generations
- PNG plots of fitness and transitions

## 2. Playing Against the AI

Both play modes load `best_model/strategy.json` and let you play against it.

### Terminal (`play.py`)

```bash
python play.py
```

Text-based interactive game. Pick an opponent from saved models, then make
action/buy/chapel decisions via numbered menus.

### Graphical (`gui.py`)

```bash
python gui.py               # requires: pip install pygame
```

Pygame window. Click cards in your hand to play actions, click supply piles
to buy. Press R to restart, Q to quit.

## 3. Watching the AI Play (`trace.py`)

Watch two AI strategies play a full game turn-by-turn:

```bash
python trace.py                # best model vs Big Money
python trace.py --seed 123     # specific seed for reproducibility
python trace.py --vs self      # best model vs itself
python trace.py --vs prev      # best model vs previous generation
python trace.py --vs 42        # best model vs gen 42
python trace.py --list         # list available generations
python trace.py --model path/to/strategy.json
```

## 4. Other Tools

- `python benchmark.py` — measure sequential vs parallel evaluation speed
- `python viz/plot_evolution.py` — interactive Plotly chart of buy priority evolution
- `python -m pytest tests/` — run smoke tests

---

## Kingdom Cards

12 kingdom cards available (a standard game uses 10 — configure in `ga/main.py`):

| Card | Cost | Type | Effect |
|------|------|------|--------|
| Chapel | $2 | Action | Trash up to 4 cards from hand |
| Village | $3 | Action | +1 card, +2 actions |
| Merchant | $3 | Action | +1 card, +1 action; first Silver played = +$1 |
| Smithy | $4 | Action | +3 cards |
| Throne Room | $4 | Action | Play an action from hand twice |
| Moneylender | $4 | Action | Trash Copper from hand, +$3 |
| Gardens | $4 | Victory | 1 VP per 10 cards in deck |
| Mine | $5 | Action | Trash a treasure, gain one costing up to $3 more to hand |
| Market | $5 | Action | +1 card, +1 action, +1 buy, +$1 |
| Laboratory | $5 | Action | +2 cards, +1 action |
| Festival | $5 | Action | +2 actions, +1 buy, +$2 |
| Council Room | $5 | Action | +4 cards, +1 buy |

Set `KINGDOM` in `ga/main.py` to pick which 10 to train on. `ALL_KINGDOM` lists all 12.

Woodcutter was removed (dropped from the Dominion 2nd edition base set)
and replaced with Throne Room.

Militia and Moat are excluded — no attack/reaction cards in this simplified engine.

## Project Structure

```
core/               Game foundation
  cards.py          Card definitions (18 cards) + integer ID mappings
  engine.py         Python game engine — used by interactive play modes
  strategy.py       Strategy genome, phase logic, I/O

ga/                 GA training pipeline
  main.py           GA entry point + config
  ga.py             Selection, crossover, mutation
  fitness.py        Game simulation and win-rate evaluation (auto-uses C engine)
  c_bridge.py       Python-C bridge (ctypes), strategy serialization

play/               Interactive play modes
  terminal.py       Terminal interactive play
  gui.py            Graphical interactive play (pygame)
  trace.py          AI vs AI game trace viewer

viz/                Visualization
  plotting.py       Fitness/transition/heatmap plots
  plot_evolution.py Interactive Plotly chart of buy priority evolution

c/                  C engine source
  dominion.c        C game engine — fast evaluation for GA training (~50x faster)
  Makefile          Builds dominion.so

tests/
  test_smoke.py     Smoke tests for all components
```

Root-level entry scripts (`train.py`, `play.py`, `gui.py`, `trace.py`) delegate
to the corresponding modules.
