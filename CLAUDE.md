# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is `JCA374/dominion-v1` (folder `trainers/ga-v1`): a genetic algorithm that evolves Dominion strategies. The
game is the shared engine `JCA374/dominion-engine`, a git submodule at `engine/` (Steam rules, all 26 base-set cards;
its own checkout is `../../engine`). Released models are played on the real Steam game by
`JCA374/dominion-steam-autopilot` (`../../autopilot`), which has this repo as a submodule and talks to it only
through `policy/`. Other kinds of models get their own trainer folder next to this one (`trainers/<kind>-v<N>`) on
the same engine and the same protocol (`docs/PROTOCOL.md`).

After releasing a model and pushing, the autopilot picks it up with
`git -C dominion-v1 pull origin main && git -C dominion-v1 submodule update --init` plus a submodule bump commit in
`../../autopilot`. After an engine change: `git -C engine pull origin main`, rerun the tests, `policy.bench --store`,
commit the bump.

## Commands

```bash
git submodule update --init   # fetch engine/ (or clone with --recursive)
make                          # build dominion.so = engine/c + c/ga_agent.c (auto-built when missing or stale)
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
python -m ga.specialist NAME --must Witch [--exclude-blocked]  # one portfolio specialist -> runs/NAME/
./train_portfolio.sh [NAME ...]                # all specialists in the queue (skips finished runs)
python -m policy.portfolio build NAME runs/*   # release them as one ga-portfolio model in models/NAME/
python -m policy.portfolio choose NAME CARD ...  # which member the portfolio plays on a kingdom
```

## Architecture

- The rules live in `engine/` only (Python reference + C twin, see its CLAUDE.md); never re-implement them here.
  The genome plays through an agent: `core/agent.py` (Python: play modes, `policy`) and `c/ga_agent.c` (C:
  training, built with the engine into `dominion.so` by `ga/c_bridge.py`). **The two agents must decide
  identically**: `tests/test_agent_parity.py` compares whole games decision by decision.
- Genome (`Strategy` in `core/strategy.py`): 3 phases (early/mid/late, `get_current_phase`) with evolvable
  transition genes; per phase a buy priority (with PASS) and Chapel trash order; shared non-terminal/terminal action
  order, terminal slack and buy targets. HOWTO.md still describes an older 4-phase genome.
- Training (`ga/`): fitness = win rate against a hall of fame (starts with Big Money; a strategy over 55% joins; max 6,
  oldest evicted), saved under `best_model/hall/`. `best_model/` changes on every run — anything played elsewhere
  must be released into `models/` first.
- Training's hall-of-fame win rate does not predict Steam results (models overfit their training kingdom);
  judge a model with `policy.bench`. Wins use Steam's tiebreak (equal VP: fewer turns wins).
- Models released before 2026-11-14 were trained on the first engine (12 Provinces, 15 cards, turn-40 cap; see
  `engine/RULES.md`); their `bench.json` is re-measured on Steam rules (bench_version 2). `k2-gen29756` and
  `mixed-gen10324` are older genome formats the current loader misreads (~0% in `policy.bench`); they are
  `"retired"`, so never auto-selected.
- Protocol 2 (2026-11-14): every card's own choices have a request (cellar, remodel, workshop, artisan, mine,
  moneylender, harbinger, vassal, library, sentry, poacher, bandit, bureaucrat), answered by the engine's decision
  sequence asked of the same GAAgent; `tests/test_card_requests.py` checks policy == genome in engine games.
  `NO_REQUEST_YET` is empty, so models buy every card.
- Portfolio models (kind `ga-portfolio`, `policy/portfolio.py`): specialist genomes trained by `ga/specialist.py`
  with all 26 cards on fresh random kingdoms holding their key cards, fitness = the benchmark bots. At game start the
  policy plays every member on the dealt kingdom (C engine, ~0.5 s, blocked cards masked) and the best one plays.
  Benchmarks and plans give every kingdom its own game seeds (`policy.bench.kingdom_seeds`, bench_version 3):
  shared seeds replay identical games wherever neither player buys the kingdom cards.
- Rollout search (2026-11-15): `c/ga_agent.c` search_decide plays each buy option out N times from a copy of the
  game (engine `de_resume_buy`/`de_hide`), same hidden shuffle per rollout, overrules the genome only past a margin;
  `ga.c_bridge.search_vs` (games) and `search_buy` (from a player's view). A portfolio with a manifest `"search"`
  (portfolio-v2s) uses it for buys when the observation has `opponent_owned`. Needs the opponent's cards: with a
  mirror-imagined opponent the gain vanishes. Measured dead ends: endgame score rule, money gate, bigger pools,
  4x generalist budget, league round 3 (see the site's training log).
- Models are plug-in folders (`models/` + `DOMINION_MODELS` folders). The selector (`policy/selector.py`) picks
  by exact training kingdom, then key cards dealt + stored bench score; never Big Money. After releasing a model run
  `policy.bench --store` so it has a score.
- `policy/` is the only interface outside players may use: JSON request/answer protocol (`docs/PROTOCOL.md`),
  `load_policy(name)`, `models/DEFAULT`. Bumping `PROTOCOL_VERSION` breaks the autopilot until it is updated too.
- Further detail: `HOWTO.md` (cards, genome, file layout), `docs/REPORT.md`, `docs/plans/`.
