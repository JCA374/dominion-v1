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

The GA trains on the shared engine's C twin (`engine/c/engine.c`, ~60x faster than Python)
with the genome's C agent (`c/ga_agent.c`). Build it before training:

```bash
make                        # builds dominion.so
```

If `dominion.so` is missing, the system auto-builds it on first run (requires `gcc`).
Falls back to the Python engine if compilation fails. Both play identical games
(`tests/test_agent_parity.py`).

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
- `best_model/hall/` — saved hall of fame opponents (Big Money + up to 5 evolved)
- `best_model/gen_NNN/` — snapshot of best strategy at each improvement
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
python trace.py --vs hall      # evaluate vs all hall of fame opponents
python trace.py --vs gens      # evaluate vs every 4th saved generation
python trace.py --vs hall --games 200  # more games for stable results
python trace.py --list         # list available generations
python trace.py --model path/to/strategy.json
```

The `--vs hall` and `--vs gens` modes don't trace individual turns — they run many
games and print a results table with win/tie/loss rates, average turns, VP margin,
and final deck composition per opponent.

## 4. Other Tools

- `python benchmark.py` — measure sequential vs parallel evaluation speed
- `python viz/plot_evolution.py` — interactive Plotly chart of buy priority evolution
- `python -m pytest tests/` — run the tests (the rules are tested in `engine/tests`)

---

## Kingdom Cards

All 26 cards of the base set (2nd edition) are available, with Steam Dominion's rules (8
Provinces in a 2-player game, Curse pile always in the Supply, no turn limit, ties go to the
player with fewer turns): see `engine/RULES.md`. Configure the training kingdom via `KINGDOM`
in `ga/main.py`; presets `KINGDOM1` (original 10) and `KINGDOM2` (11 cards with attacks).

The genome decides buys, action order, Chapel/Throne Room/Mine targets and phase timing. The
other choices (attacks, and Cellar, Harbinger, Vassal, Workshop, Bureaucrat, Poacher, Remodel,
Library, Sentry, Artisan, Bandit) are fixed heuristics on the same priority lists, in
`core/agent.py` (Python) and `c/ga_agent.c` (C), which must stay identical.

## Strategy Genome

The GA evolves a **4-phase strategy** with phase transitions controlled by evolvable genes:

- **Early** → **Mid**: after turn N (`early_to_mid_turn`, range 2–15)
- **Mid** → **Late**: when provinces drop to N (`mid_to_late_provinces`, range 2–8)
- **Late** → **End**: when provinces drop to N (`late_to_end_provinces`, range 1–4)

Each phase has independent genes for:
- **Buy priority**: ordered list of cards to buy (with PASS to stop buying)
- **Action priority**: separate non-terminal and terminal play order
- **Chapel trash priority**: what to trash (with STOP sentinel)

Additional evolvable parameters:
- `buy_targets`: max copies of each action card to buy
- `province_max_coins` / `duchy_max_coins`: skip Province/Duchy if coins exceed threshold
- `militia_coin_threshold`: Militia discard heuristic — high money keeps treasures, low money keeps actions
- `throne_room_priority` / `mine_trash_priority`: shared across phases

## Hall of Fame

Training uses a **hall of fame** system for robust evaluation:

1. Starts with Big Money as the only opponent
2. When the best strategy exceeds 55% win rate, it's added to the hall
3. The GA then must beat the harder hall — overall best resets
4. Max 6 members — oldest non-BM member is evicted when full
5. Hall is saved to `best_model/hall/` for post-training evaluation

## Project Structure

```
engine/             git submodule JCA374/dominion-engine: the rules (Python + C), see its README

core/               The genome and its agent
  cards.py          Card facts from the engine, in the shape the genome code uses
  agent.py          GAAgent: answers every engine decision from a Strategy
  engine.py         play_game_2p(): genomes against each other on the Python engine
  strategy.py       Strategy genome (3 phases), I/O, summaries

ga/                 GA training pipeline
  main.py           GA entry point + config (kingdom presets, hall settings)
  ga.py             Selection, crossover, mutation, hall of fame persistence
  fitness.py        Game simulation and win-rate evaluation (auto-uses C engine)
  c_bridge.py       Builds dominion.so (engine/c + c/ga_agent.c), genome serialization

play/               Interactive play modes
  terminal.py       Terminal interactive play
  gui.py            Graphical interactive play (pygame)
  trace.py          AI vs AI game trace + hall/generation evaluation

viz/                Visualization
  plotting.py       Fitness/transition/heatmap plots (4-phase)
  plot_evolution.py Interactive Plotly chart of buy priority evolution

policy/             Stable interface for players (Steam autopilot) — see docs/PROTOCOL.md
  protocol.py       Request/answer format, observation, answer checking
  ga_policy.py      Plays a GA strategy.json with the training engine's own rules
  registry.py       models/<name>/model.json, load_policy(), models/DEFAULT
  release.py        python -m policy.release best_model <name>
  check.py          python -m policy.check [name ...]  (conformance)
  bench.py          python -m policy.bench [name ...]  (strength vs fixed bots on Steam-like kingdoms)
  serve.py, client.py  the same protocol over a JSON-lines pipe

models/             Released, immutable models (model.json + strategy.json)
best_model/         Live output of the current training run

c/
  ga_agent.c        The genome as a C agent of the engine (twin of core/agent.py)

tests/
  test_ga.py        Genome, GA operators, fitness, hall of fame, C bridge
  test_agent.py     What the genome decides, on hand-made positions
  test_agent_parity.py  Python and C agents play identical games
  test_policy.py    Protocol, registry and released-model conformance

docs/               PROTOCOL.md, REPORT.md, baseline.md, plans/
```

Root-level entry scripts (`train.py`, `play.py`, `gui.py`, `trace.py`) delegate
to the corresponding modules.
