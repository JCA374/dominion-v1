# Dominion GA — Quick Reference

## Training (`main.py`)

Train a strategy using a genetic algorithm:

```bash
python main.py              # fresh training run (1000 generations)
python main.py --continue   # continue from best_model/, keeps generation numbering
```

Config is at the top of `main.py` — population size, mutation rate, kingdom cards, etc.
Outputs: `best_model/strategy.json`, `evolution_log.csv`, and PNG plots.

## Game Trace (`trace.py`)

Watch the AI play a full game turn-by-turn in the terminal:

```bash
python trace.py                # best model vs Big Money
python trace.py --seed 123     # specific seed for reproducibility
python trace.py --vs self      # best model vs a copy of itself
python trace.py --vs prev      # best model vs previous generation
python trace.py --vs 42        # best model vs gen 42
python trace.py --list          # list available generations
python trace.py --model path/to/strategy.json  # use a different model
```

Shows a chat-style layout — model actions on the left, opponent on the right.

## Interactive Play (`play.py`)

Play Dominion against the evolved AI yourself:

```bash
python play.py
```

Discovers all saved models under `best_model/` and lets you pick one to play against.
You make action/buy/chapel decisions interactively; the AI uses its evolved strategy.

## Kingdom Cards

12 kingdom cards available (a standard game uses 10 — configure in `main.py`):

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

Set `KINGDOM` in `main.py` to pick which 10 to train on. `ALL_KINGDOM` lists all 12.

Woodcutter was removed (dropped from the Dominion 2nd edition base set)
and replaced with Throne Room.

Militia and Moat are excluded — no attack/reaction cards in this simplified engine.

## Other Tools

- `python benchmark.py` — measure sequential vs parallel evaluation speed
- `python plot_evolution.py` — interactive Plotly chart of how buy priorities evolved
- `python test_smoke.py` — run smoke tests
