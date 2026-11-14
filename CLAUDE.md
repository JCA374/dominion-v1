# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is `JCA374/dominion-v1`: a genetic algorithm that evolves Dominion strategies on a simplified engine. Its released
models are played on the real Steam game by `JCA374/dominion-steam-autopilot` (checked out at `../dominion-auto`),
which has this repo as a submodule and talks to it only through `policy/`. Future work here: improve the GA, add more
models (including non-GA ones speaking the same protocol, see `docs/PROTOCOL.md`).

After releasing a model and pushing, the autopilot picks it up with `git -C dominion-v1 pull origin main` plus a
submodule bump commit in `../dominion-auto`.

## Commands

```bash
make                          # build c/ -> dominion.so (auto-built on first run if missing; falls back to Python engine)
python train.py [--continue]  # train; writes best_model/. GA config + KINGDOM presets at the top of ga/main.py
python play.py | gui.py       # play against best_model in terminal / pygame
python trace.py [--vs self|prev|hall|gens|N] [--seed N] [--games N]   # AI vs AI trace or win-rate tables
python -m pytest -q tests                      # all tests
python -m pytest -q tests/test_policy.py -k X  # one test
python -m policy.release best_model <name> --description "..." [--default]   # immutable models/<name>/
python -m policy.check [name ...]              # conformance of released models (answer format only)
python -m policy.bench [name ...] [--store] [--missing]  # strength: vs Big Money / Smithy-BM / Witch-BM on
                                               # Steam-like kingdoms with the model's key cards, random ones and the
                                               # training kingdom (~1 s per model); --store writes models/<name>/bench.json
python -m policy.keycards [--write] [name ...] # key cards (kingdom cards a model is built around), measured
python -m policy.select CARD ... | --best-kingdom   # which model the selector picks for a kingdom, and why
```

## Architecture

- Two engines with the same rules: `core/engine.py` (Python, used by play modes and `policy`) and `c/dominion.c`
  (training, via ctypes in `ga/c_bridge.py`). **A rules change must be made in both**, or trained strategies stop
  playing the way they were evaluated.
- Genome (`Strategy` in `core/strategy.py`): 3 phases (early/mid/late, `get_current_phase`) with evolvable
  transition genes; per phase a buy priority (with PASS) and Chapel trash order; shared non-terminal/terminal action
  order, terminal slack and buy targets. HOWTO.md still describes an older 4-phase genome.
- Training (`ga/`): fitness = win rate against a hall of fame (starts with Big Money; a strategy over 55% joins; max 6,
  oldest evicted), saved under `best_model/hall/`. `best_model/` changes on every run — anything played elsewhere
  must be released into `models/` first.
- Training's hall-of-fame win rate does not predict Steam results (models overfit their training kingdom);
  judge a model with `policy.bench`. `tests/test_smoke.py::test_c_engine_matches_python_with_attacks` guards
  engine parity for lopsided matchups — a Big Money mirror is ~50% in any engine and proves nothing.
- Known gaps vs Steam: both engines use 12 Provinces/Duchies in a 2-player game (Steam: 8), and know 15 of the
  26 base-set kingdom cards. `k2-gen29756` and `mixed-gen10324` are older genome formats the current loader
  misreads (they score ~0% in `policy.bench`); don't play them (they are `"retired"`, so never auto-selected).
- Models are plug-in folders (`models/` + `DOMINION_MODELS` folders). The selector (`policy/selector.py`) picks
  by exact training kingdom, then key cards dealt + stored bench score; never Big Money. After releasing a model run
  `policy.bench --store` so it has a score.
- `policy/` is the only interface outside players may use: JSON request/answer protocol (`docs/PROTOCOL.md`),
  `load_policy(name)`, `models/DEFAULT`. Bumping `PROTOCOL_VERSION` breaks the autopilot until it is updated too.
- Further detail: `HOWTO.md` (cards, genome, file layout), `docs/REPORT.md`, `docs/plans/`.
