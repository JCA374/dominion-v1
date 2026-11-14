# Model ↔ player protocol (v1)

This repo trains models. Players use them; the main one is the
[Steam autopilot](https://github.com/JCA374/dominion-steam-autopilot), which has
this repo as its `dominion-v1/` submodule. The two sides meet in one place, the
`policy/` package. Players import only from `policy`, never from `core/`, `ga/`
or `best_model/`. Anything else in this repo can change without breaking the
autopilot.

```
 dominion-v1 (this repo)                         dominion-steam-autopilot
 ───────────────────────                         ────────────────────────
 ga/ trains ──> best_model/   (changes every run)
                    │
       python -m policy.release best_model k4-...
                    ▼
 models/<name>/model.json + strategy.json   (immutable)
                    │
 policy.load_policy(name) ── Policy ── JSON requests ──>  autopilot/brain.py
                                                        (screen → observation,
                                                         answer → clicks)
```

## Roles

* **The player** sees the game, keeps track of its own deck, supply and turn, and
  turns each decision into a request.
* **The model** answers. It keeps no state between requests, so everything it
  needs is in the observation.

## Messages

Each request is a JSON object, and so is each answer. The same objects are used
in-process (`policy.handle(request)`) and over a pipe (`python -m policy.serve`:
one JSON line in, one JSON line out).

Observation, sent as `"obs"` with every decision:

| field | type | meaning |
|---|---|---|
| `turn` | int | turn number, 1 = first turn |
| `provinces_left` | int | Provinces left in the supply |
| `kingdom` | [str] | the 10 kingdom cards of this game |
| `empty_piles` | [str] | supply piles known to be empty |
| `owned` | {str: int} | the player's whole deck, all zones |
| `hand` | [str] | cards in hand now |
| `actions`, `buys`, `coins` | int | this turn's counters |

| request | extra | answer |
|---|---|---|
| `{"type": "info"}` | | `{"protocol": 1, "name", "kind", "kingdom", ...}` |
| `action` | | `{"play": card or null}`: the next action to play (asked again after each one) |
| `buy` | | `{"buy": [card, ...]}`: all buys this turn, in order |
| `trash` | | `{"trash": [card, ...]}`: Chapel |
| `discard` | `"keep": n` | `{"discard": [card, ...]}`: an attack, discard down to n |
| `throne_room` | | `{"target": card or null}` |

Any answer may also include `"phase"` (a label for logs) or `"error"` (a message;
the player should then fall back or wait). Card names are the English names on
the cards. A model must cope with cards it doesn't know (Steam kingdoms vary):
it never buys them, and it discards them first.

The engine (`engine/`, Steam rules) knows all 26 base-set cards, but the requests above can
only express the choices of some of them. A model is never told to buy a card in
`policy.protocol.NO_REQUEST_YET` (Artisan, Cellar, Harbinger, Library, Poacher, Remodel,
Sentry, Vassal, Workshop), so a player never meets their prompts. Opening one up means a new
request here, a protocol version bump, and the player learning that card's prompt.

`policy.check_answer(request, answer)` checks an answer is legal: the cards are in
hand, the buys are affordable, nothing comes from an empty pile.

## Model files

`models/<name>/model.json`:

```json
{"protocol": 1, "name": "k1-steam", "kind": "ga-strategy", "file": "strategy.json",
 "kingdom": ["Chapel", "Festival", ...], "created": "2026-11-13",
 "source": "best_model @ 8444772", "description": "...", "trained_vs": "62% win ..."}
```

Optional fields (adding them never bumps the protocol):

* `key_cards`: `{"Witch": 2, "Market": 2, "Merchant": 1}`, the kingdom cards the model's game
  is built around. `policy.release` measures them for a GA model (`policy/keycards.py`); any
  other kind declares them. The model selector lets a model play any kingdom that holds all
  its key cards.
* `retired`: `true` keeps a model out of automatic selection (it can still be played by name).

`kind` decides how the model is loaded:

* `ga-strategy`: a GA `strategy.json`, played by `policy/ga_policy.py` using the
  training engine's own functions, so a real game matches training exactly.
* `subprocess`: any program that speaks the JSON-lines protocol on stdin/stdout,
  for example a neural network in its own venv, with its own dependencies:
  ```json
  {"protocol": 1, "name": "net-v1", "kind": "subprocess",
   "command": ["/home/me/nets/.venv/bin/python", "-m", "nets.serve", "ckpt.pt"],
   "cwd": "/home/me/nets", "kingdom": [...]}
  ```
  Nothing in the autopilot has to change for these.

To add an in-process kind, subclass `policy.Policy`, implement its five methods
and register the kind in `policy/registry.py:load_policy`.

## Models are plug-ins

A model is a folder with a `model.json`. Folders are found in `models/` and in every folder
listed in the `DOMINION_MODELS` environment variable (`os.pathsep`-separated), so a player
keeps its own models folder (the autopilot's is `autopilot/models/`) and adding a model
is adding a folder: no code, menu or UI changes. `list_models()` returns them all; a name in
`models/` wins over the same name elsewhere.

## Benchmark and model selector

`python -m policy.bench --store [--missing]` plays every GA model against Big Money,
Smithy-BM and Witch-BM on random Steam-like kingdoms that contain its key cards, and writes
`bench.json` next to its `model.json` (`"score"` = mean of the three; also the scores on
random kingdoms and on the training kingdom, and the measured `key_cards`). A model kind the
engine can't play (e.g. `subprocess`) can ship its own `bench.json` with a `"score"`.

`policy.select_model(kingdom)` (`policy/selector.py`, CLI `python -m policy.select CARD ...`)
picks the model for a dealt kingdom:

1. a model trained on exactly this kingdom;
2. else the best-scoring model whose key cards are all in the kingdom;
3. else the model with the largest share of its key cards dealt, best score first.

There is no Big Money fallback: the answer is always a real, non-retired model.
`rank_models(kingdom)` gives the whole ranking with a reason per model, and
`best_kingdom()` the model + training kingdom to play when the kingdom is ours to choose.

## Workflow: a new model on Steam

```bash
python train.py                                         # writes best_model/
python -m policy.release best_model k4-gen52000 \
       --description "KINGDOM4, hall diversity"         # copies + checks; add --default to make it the default
python -m policy.check k4-gen52000                      # conformance (release runs it too)
python -m policy.bench --store k4-gen52000              # numbers for the model selector
git add models/k4-gen52000 && git commit && git push

# in dominion-steam-autopilot
git -C dominion-v1 pull origin main && git add dominion-v1 && git commit -m "dominion-v1: k4-gen52000"
.venv/bin/python autopilot/pilot.py --list-models
.venv/bin/python autopilot/pilot.py --model k4-gen52000
```

Released models are never changed: give a new release a new name. `models/DEFAULT`
names the model the autopilot plays when no `--model` is given.

## Versioning

`PROTOCOL_VERSION` (in `policy/protocol.py`) goes up only when a field or request
changes meaning or is removed. Adding an optional field or a new request type
does not bump it, and players ignore fields they don't know. A model whose
`protocol` is newer than the code is refused when it loads, and a subprocess
model is checked with `info` when it starts.

Tests: `tests/test_policy.py` (every released model passes the conformance check,
GA buys match the engine, the pipe gives the same answers as in-process) and
`tests/test_select.py` (plug-in folders, key cards, bench store, selector rules).
