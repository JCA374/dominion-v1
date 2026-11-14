# dominion-v1

A genetic algorithm that evolves Dominion strategies. The game is the shared
[dominion-engine](https://github.com/JCA374/dominion-engine) (Steam Dominion's rules, all 26
base-set cards; git submodule `engine/`); this repo adds the genome as an agent in Python and C.
Clone with `git clone --recursive` (or `git submodule update --init`). The evolved models are
played on the real Steam game by
[dominion-steam-autopilot](https://github.com/JCA374/dominion-steam-autopilot),
which has this repo as a submodule.

```bash
make                         # build dominion.so (engine/c + c/ga_agent.c; also auto-built)
python train.py              # train; writes best_model/ (KINGDOM in ga/main.py)
python play.py               # play against a model in the terminal (gui.py: pygame)
python trace.py              # watch two models play
python -m pytest -q tests    # tests
```

## Models and the autopilot

`best_model/` is the live output of training and changes on every run. To play a
model anywhere else, release it:

```bash
python -m policy.release best_model k4-gen52000 --description "..."   # --default to make it the default
python -m policy.check                                                # every released model
python -m policy.bench [name ...]       # score vs Big Money, Smithy-BM, Witch-BM on random Steam-like kingdoms
```

`models/<name>/` then holds an immutable `model.json` (kind, training kingdom,
date, source, win rate) and `strategy.json`. Players load models only through
`policy` (`load_policy(name)`) and speak a small JSON request/answer protocol.
Models that live outside this repo (for example a neural net in its own venv)
can use the same protocol over a pipe. See **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

| model | kingdom | notes |
|---|---|---|
| `k1-steam` (default) | KINGDOM1 | the model behind the recorded Steam wins |
| `k4-20260614` | KINGDOM4 (Witch, Militia, Moat, Moneylender) | latest run |
| `k1-gen*`, `k2-gen*`, `k2f-gen13665`, `mixed-gen10324` | see model.json | earlier snapshots |

## More

* [HOWTO.md](HOWTO.md): training options, play modes, genome, hall of fame, file layout
* [docs/REPORT.md](docs/REPORT.md): technical report
* [docs/plans/](docs/plans/): design plans (phase-aware GA, engine chains, balance groups)
