# Implementation Plan: Evolve Longer Action Chains (Engine Strategies)

**Repository:** dominion-v1
**Goal:** The GA currently converges on money-heavy strategies with a few terminals. This plan removes the structural obstacles preventing engine strategies (long action-card chains) from evolving, in five phases. Genome changes are allowed and expected.

**Instructions to the implementer:** Work through the phases **in order**. Each phase ends with verification steps — do not start the next phase until they pass. Phase 2 is the largest change and touches the genome, both engines (Python and C), serialisation, and tests; read its whole section before editing. Old saved models in `best_model/` will remain loadable via backward compatibility in `load_strategy`.

**Baseline reference (current state, for orientation):**
- Fitness blend in `ga/main.py`: `VP_MARGIN_WEIGHT=0.5`, `SPEED_WEIGHT=0.2`, `ENGINE_WEIGHT=0.05` → win rate only carries weight 0.25, and the speed bonus actively punishes engine setup turns.
- `Strategy.action_priority` is a single merged list; `play_action_phase` greedily plays the highest-priority action in hand regardless of remaining actions, so a terminal ranked above a Village kills the chain.
- `random_strategy` caps every kingdom card via `buy_targets = randint(1, 5)`; mutation jitters ±1 — too slow to reach 6+ Villages.
- Training kingdom is `KINGDOM2` (contains Militia + Witch, both strongly anti-engine).

---

## Phase 0 — Baseline and guardrails

1. Run the existing test suite and make sure it is green before any change:
   ```bash
   python -m pytest tests/ -q
   ```
2. Build the C engine (`make -C c`) and confirm `ga/fitness.py` reports `USE_C_ENGINE = True` when imported.
3. Record a baseline so later phases can be compared: evaluate `best_model/strategy.json` vs Big Money over 400 seeds with `evaluate_vs_opponent` and note `win_rate`, `mean_turns`, and `mean_actions_per_turn`. Write the numbers into a new file `docs/baseline.md`.

**Done when:** tests green, C engine loads, baseline numbers recorded.

---

## Phase 1 — Fitness shaping (config + logging only, no genome change)

### 1.1 Re-weight the blended fitness (`ga/main.py`)

```python
VP_MARGIN_WEIGHT = 0.30   # was 0.5
SPEED_WEIGHT     = 0.0    # was 0.2 — speed bonus punishes engine setup turns
ENGINE_WEIGHT    = 0.15   # was 0.05
```

Win rate now carries weight 0.55. Pass these through `config` as today (the keys already exist in `run_ga`).

### 1.2 Optional engine-weight annealing (`ga/ga.py`)

Inside `run_ga`, support a linear decay of the engine bonus so early generations are pushed towards action play but the eventual winner is judged mostly on win rate:

- New config keys: `engine_weight_end` (default: same as `engine_weight`) and `engine_anneal_gens` (default 300).
- At the top of each generation compute:
  ```python
  progress = min(1.0, (gen - start_gen) / max(1, engine_anneal_gens))
  engine_weight_now = engine_weight + (engine_weight_end - engine_weight) * progress
  ```
- Use `engine_weight_now` inside `_blended` (move `_blended` so it closes over the per-generation value, or pass it as an argument).
- In `ga/main.py` set `ENGINE_WEIGHT = 0.25`, `ENGINE_WEIGHT_END = 0.10`, `ENGINE_ANNEAL_GENS = 300` and wire them into `config`.

### 1.3 Observability

- Add a `mean_actions_per_turn` column to the CSV written in `run_ga` (value from `best_eval_result`), and append `apt={x:.2f}` to the per-generation console line.
- Note: the CSV header changes — keep `csv_append` behaviour intact (header only written on fresh files, as today).

### 1.4 Verification

- `python -m pytest tests/ -q` still green.
- Run a short training smoke (e.g. temporarily `GENERATIONS=3`, `POP_SIZE=20`, `GAMES_PER_EVAL=20`) and confirm the new CSV column and console field appear and the run completes with both `WORKERS=1` and `WORKERS>1`.

---

## Phase 2 — Genome change: split action priority into non-terminal / terminal lists with a hard "non-terminals first" rule

**Rationale:** Playing +action cards before terminals is (in this card pool) never worse and usually strictly better. Encoding it as a hard rule removes an entire axis of the search space the GA currently wastes generations on, while the *within-group* ordering stays evolvable.

### 2.1 `core/strategy.py` — Strategy dataclass

Replace the single field:

```python
action_priority: list[str] = field(default_factory=list)
```

with two fields plus the Phase 3 scalar (add it now so the genome layout is changed only once):

```python
nonterminal_priority: list[str] = field(default_factory=list)  # cards with actions > 0
terminal_priority: list[str] = field(default_factory=list)     # cards with actions == 0
terminal_slack: int = 1  # used by Phase 3 buy rule; inert until then
```

Update helpers:

- Replace `get_action_priority(strategy, phase)` with two functions `get_nonterminal_priority(strategy)` and `get_terminal_priority(strategy)` (no phase argument — the lists are shared across phases, as today). Keep a thin deprecated `get_action_priority` that returns `nonterminal_priority + terminal_priority` so any straggler call sites keep working; grep and migrate them anyway (see 2.7).
- `random_strategy`: build the lists with the existing helpers `_nonterminal_actions(kingdom)` / `_terminal_actions(kingdom)`, each independently shuffled. Initialise `terminal_slack = rng.randint(0, 3)`.
- Seed archetypes (`big_money_strategy`, `gardens_strategy`, `engine_strategy`): split their current `action_priority` content by `ALL_CARDS[c].actions > 0`. For `engine_strategy`, use `nonterminal_priority` ordered `Village > Laboratory > Market > Merchant > Festival` (filtered to kingdom) and `terminal_priority` ordered `Chapel > Smithy > Throne Room > Moneylender > Mine > Council Room > Militia > Witch > Moat` (filtered to kingdom). Set `terminal_slack=1` in all three archetypes.
- `describe` / `summarize`: print both lists (`Non-terminals: ...` / `Terminals: ...`) instead of the single `Actions:` line, and include `terminal_slack` in the output.
- `load_strategy` backward compatibility (extend the existing compat block):
  1. If `nonterminal_priority` in data → use directly.
  2. Else if `action_priority` in data → split it by `ALL_CARDS[c].actions > 0`, preserving order within each group.
  3. Else if `early_nonterminal_priority` in data (oldest format) → use `early_nonterminal_priority` / `early_terminal_priority`.
  4. Else → defaults from `NONTERMINAL_ACTIONS` / `TERMINAL_ACTIONS`.
  Also read `terminal_slack` with `data.get("terminal_slack", 1)`.

### 2.2 `core/engine.py` — action phase

Replace `play_action_phase` with the two-tier scan. Re-scan from the top after every play so newly drawn cards are considered:

```python
def play_action_phase(state, strategy, opponents=None):
    from core.strategy import (get_current_phase, get_nonterminal_priority,
                               get_terminal_priority)
    phase = get_current_phase(state.turn, state.supply["Province"], strategy.transitions)
    nt_prio = get_nonterminal_priority(strategy)
    t_prio = get_terminal_priority(strategy)

    actions_played = 0
    while state.actions > 0:
        played = False
        # Tier 1: always exhaust non-terminals first (never worse in this pool)
        for card_name in nt_prio:
            if card_name in state.hand:
                resolve_action(state, card_name)
                _handle_special(state, card_name, strategy, phase, opponents)
                actions_played += 1
                played = True
                break
        if played:
            continue
        # Tier 2: then terminals
        for card_name in t_prio:
            if card_name in state.hand:
                resolve_action(state, card_name)
                _handle_special(state, card_name, strategy, phase, opponents)
                actions_played += 1
                played = True
                break
        if not played:
            break
    return actions_played
```

Update `militia_discard`: it currently ranks actions by index in the merged list. New rule: non-terminals are always kept over terminals; within each group, lower index in the respective priority list = keep longer (i.e. discard sooner the *later* it appears). Implement `_action_rank` so terminals get rank `index_in_t_prio` and non-terminals get `1000 + index_in_nt_prio` reversed appropriately — preserve the existing semantics ("lower rank = discard sooner") and add a unit test pinning it.

`play_throne_room` is unchanged (still driven by `throne_room_priority`).

### 2.3 New genome layout — `ga/c_bridge.py` and `c/dominion.c`

Both files must change **together**. New flat layout (replaces the old one entirely):

| Offset | Field | Slots |
|---|---|---|
| 0 | early_to_mid_turn | 1 |
| 1 | mid_to_late_provinces | 1 |
| 2 | mid_to_late_turn | 1 |
| 3 | chapel_max_trash | 1 |
| 4 | terminal_slack | 1 |
| 5 | S_EARLY_BUY | 20 |
| 25 | S_MID_BUY | 20 |
| 45 | S_LATE_BUY | 20 |
| 65 | S_NT_ACTION | 8 |
| 73 | S_T_ACTION | 12 |
| 85 | S_EARLY_CHAPEL | 6 |
| 91 | S_MID_CHAPEL | 6 |
| 97 | S_LATE_CHAPEL | 6 |
| 103 | S_THRONE_ROOM_PRIO | 12 |
| 115 | S_MINE_TRASH_PRIO | 4 |
| 119 | S_BUY_TARGETS | 20 |
| **139** | **STRATEGY_SIZE** | |

In `c_bridge.py::strategy_to_ints`: write `buf[4] = strategy.terminal_slack`, then `_write_list` calls with the new offsets; `STRATEGY_SIZE = 139`. Buy-targets sentinel guard becomes `offset < 137`.

In `c/dominion.c`:
- Update all `S_*` defines and `STRATEGY_SIZE` per the table; add `#define S_TERMINAL_SLACK 4`.
- `play_action_phase`: two-tier scan mirroring 2.2 exactly — first loop over `strat + S_NT_ACTION` (8 slots, -1 terminated), `continue` the outer while-loop on a play; second loop over `strat + S_T_ACTION` (12 slots).
- `action_keep_rank` (militia discard): mirror the new keep/discard semantics from 2.2.
- The buy-phase will read `S_TERMINAL_SLACK` in Phase 3; in this phase just define the constant.

### 2.4 `ga/ga.py` — operators

- `crossover`: replace the single `action_priority=order_crossover(...)` line with one `order_crossover` per new list; add `terminal_slack=rng.choice([p1.terminal_slack, p2.terminal_slack])`.
- `mutate`: mutate both lists with `_mutate_list(..., allow_pass=False)`; add scalar jitter:
  ```python
  if rng.random() < rate:
      s.terminal_slack = max(0, min(4, s.terminal_slack + rng.choice([-1, 1])))
  ```

### 2.5 `play/trace.py`

Line ~203 uses `get_action_priority(strategy, phase)` to replicate engine decisions for tracing. Port it to the same two-tier scan as 2.2 so traces match actual play. Any trace output that prints "action priority" should print both lists.

### 2.6 Tests (`tests/test_smoke.py`)

- Mechanical migration: every `Strategy(... action_priority=[...] ...)` construction → split into `nonterminal_priority` / `terminal_priority` by `ALL_CARDS[c].actions > 0`.
- Update the genome-validity test (~line 198) to assert each new list contains exactly the right card set with no duplicates.
- **New tests:**
  1. *Chain ordering:* hand = `["Smithy", "Village", "Copper"]`, `terminal_priority` deliberately lists Smithy first overall — assert Village is played before Smithy and both get played (2 actions consumed correctly, Smithy not stranded).
  2. *Re-scan after draw:* stack the deck so Smithy draws a Village; assert the drawn Village is played when actions remain.
  3. *Serialisation round-trip:* `strategy_to_ints` writes `terminal_slack` at offset 4, NT list at 65, T list at 73, sentinel `-1` present in every list, buy-target pairs intact at 119.
  4. *Backward compat:* `load_strategy` on (a) a dict with merged `action_priority`, (b) a dict with the oldest `early_nonterminal_priority` format — both produce valid split lists.
  5. *Python/C agreement smoke:* run `evaluate_vs_opponent` with `need_deck=True` (pure Python) and the C path on the same strategies and confirm both run without error and produce win rates in [0,1]. (Exact equality is not expected — different RNGs.)

### 2.7 Grep checklist before finishing the phase

```bash
grep -rn "action_priority" --include="*.py" .
```

Known call sites to migrate: `core/strategy.py`, `core/engine.py`, `ga/ga.py`, `ga/c_bridge.py`, `play/trace.py`, `tests/test_smoke.py`. Also grep `viz/`, `gui.py`, `benchmark.py`, `play/terminal.py`, `play/gui.py` for `get_action_priority` and the string `"Actions:"`. Update `REPORT.md` / `phase_aware_ga_plan.md` docs sections describing the genome.

**Done when:** full test suite green, C library rebuilds cleanly (the `_ensure_built` mtime check will trigger automatically), and a 3-generation smoke training run completes.

---

## Phase 3 — Buy-phase terminal-density rule (activates `terminal_slack`)

**Rule:** never buy a terminal action the deck cannot support. Define:

- `extra_action_capacity` = Σ over all owned cards of `max(0, card.actions - 1)` (Village/Festival contribute 1 each; Market/Lab/Merchant contribute 0).
- `terminals_owned` = count of owned action cards with `actions == 0`, **excluding Throne Room** (it is functionally a pseudo-village, not a chain-stopper).
- When the buy scan reaches a terminal action card (again excluding Throne Room): if `terminals_owned >= extra_action_capacity + strategy.terminal_slack`, `continue` to the next card in the priority list (exactly like the existing buy-target skip).

### 3.1 `core/engine.py::play_buy_phase`

The function already counts `owned` when `buy_targets` is non-empty. Always compute `owned` now (it is needed for the density rule regardless), derive the two aggregates above once before the buy loop, and **update them incrementally after each purchase** inside the loop (a bought Village raises capacity immediately; a bought Smithy raises `terminals_owned`).

### 3.2 `c/dominion.c::play_buy_phase`

Mirror exactly: compute capacity/terminal counts from deck+hand+discard+play arrays before the loop using `card_actions[]`/`card_type[]`, read slack from `strat[S_TERMINAL_SLACK]`, skip with `continue`, update incrementally after each buy. Exclude `THRONE_ROOM` from the terminal count.

### 3.3 Tests

1. Strategy with `terminal_slack=0`, owned deck = starting deck (0 capacity, 0 terminals), $4, buy priority `["Smithy", "Silver", "PASS"]` → with slack 0... starting deck has 0 terminals, so `0 >= 0 + 0` is true → buys Silver, not Smithy. Same setup with `terminal_slack=1` → buys Smithy.
2. Deck containing 1 Village and 1 Smithy, `terminal_slack=1` → capacity 1, terminals 1, `1 >= 1+1` false → a second Smithy may be bought; with `terminal_slack=0` → `1 >= 1` true → skipped.
3. Throne Room is never blocked by the rule and never counted as a terminal.
4. Incremental update: with 2 buys (Festival in play), slack such that the first terminal purchase is allowed, assert the second is correctly blocked/allowed given the updated counts.

**Done when:** tests green, 3-generation smoke run completes, and `mean_actions_per_turn` for the engine seed archetype (evaluate `engine_strategy()` vs Big Money over 200 seeds) does not regress versus Phase 2.

---

## Phase 4 — Buy-target initialisation and mutation tuning

In `core/strategy.py::random_strategy`, replace the uniform `randint(1, 5)`:

```python
for card in kingdom_cards:
    c = ALL_CARDS[card]
    if c.card_type == CardType.ACTION and c.actions > 0:
        buy_targets[card] = rng.randint(2, 8)    # engine components
    elif c.card_type == CardType.ACTION:
        buy_targets[card] = rng.randint(1, 3)    # terminals (density rule also guards these)
    else:
        buy_targets[card] = rng.randint(2, 8)    # victory kingdom cards (e.g. Gardens)
```

In `ga/ga.py::mutate`, widen the jitter: `rng.choice([-2, -1, 0, 1, 2])`, clamp to `[1, 12]`.

In `engine_strategy`, raise the seed targets: `Village 5, Laboratory 4, Market 3, Festival 3, Merchant 3, Smithy 3, Chapel 1, Throne Room 2, Mine 1, Moneylender 1, Council Room 2` (filtered to kingdom as today).

**Done when:** tests green (update any test pinning the old `randint(1,5)` behaviour).

---

## Phase 5 — Training configuration and validation runs

### 5.1 Environment check first (`ga/main.py`)

Set `KINGDOM = KINGDOM1` (no Militia/Witch). Militia chops engine hands to 3 and Witch fills decks with Curses — both suppress chains regardless of GA quality. Run A must isolate the GA from that.

### 5.2 Run A — sanity (KINGDOM1, ~200 generations)

Success criteria, read from the CSV:
- `mean_actions_per_turn` of the best strategy trends above **1.8** (baseline ≈ 1.0–1.3).
- Best strategy's early/mid buy lists contain Village/Laboratory/Market in the top ranks, and `avg_final_deck` in `best_model/summary.txt` shows ≥ 5 non-terminal actions.
- Win rate vs hall remains ≥ 0.55 after the engine-weight anneal has decayed.

If criteria fail, debug in this order: (1) confirm the C engine actually rebuilt (delete `dominion.so`, rerun), (2) confirm Phase 2's tier-1/tier-2 scan with a trace via `play/trace.py`, (3) raise `ENGINE_WEIGHT` start to 0.3.

### 5.3 Run B — KINGDOM2 (with attacks)

Switch back to `KINGDOM2` and run with `--continue` disabled (fresh hall). Expectation management: an engine may legitimately *not* be optimal here; the interesting output is whether the GA now at least sustains higher `mean_actions_per_turn` candidates in the population (visible in the CSV) before money strategies win, and whether Moat appears in evolved buy lists.

### 5.4 Documentation

Update `REPORT.md` with the new genome layout table, the two-tier action policy, the terminal-density rule, and Run A/B results next to the Phase 0 baseline.

---

## Out of scope (deliberately)

- Evolvable per-phase action lists (re-adding phase split would triple the action-ordering search space the hard rule just removed).
- Smarter Throne Room targeting (e.g. prefer doubling non-terminals when terminals are stranded) — revisit only if Run A succeeds and TR remains unused in evolved decks.
- Draw-probability-aware play (knowing deck contents) — different project.
