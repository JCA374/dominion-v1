# Next Dev: Deck-Aware Buy Balancing

## Problem

The buy priority is a static list per phase. If you want 5 each of two cards
(e.g. Laboratory and Market), it's often optimal to alternate buying them based
on current deck composition. The current system always buys whichever is higher
in the priority list first.

## Recommended Approach: Balance Groups

Add `balance_groups: list[list[str]]` to the Strategy genome.

Example: `[["Laboratory", "Market"]]`

At buy time, dynamically reorder the priority list so the least-owned card in
each group gets promoted ahead of the others. The GA can mutate groups
(pair/unpair cards, add/remove groups).

### Implementation Steps

1. Add `balance_groups: list[list[str]]` field to `Strategy` dataclass
2. In `play_buy_phase` (Python, `core/engine.py`): before iterating the buy
   priority, build a reordered list that promotes the least-owned card in each
   balance group
3. In `play_buy_phase` (C, `c/dominion.c`): same logic — encode groups as flat
   array of card ID pairs terminated by -1 in the strategy buffer
4. Update `c_bridge.py` serialization to write balance groups into the buffer
5. GA mutations (`ga/ga.py`): occasionally pair two adjacent action cards into a
   group, split existing groups, or swap members
6. Update `random_strategy()`, `describe()`, `summarize()`, serialization
   (`save/load`) to handle the new field

### Alternative Approaches Considered

**Deck-aware tiebreaking (simplest, no genome change):** When iterating buy
priority, if the next card is also affordable and you own more of the current
card, skip to the next. Zero genome changes but always balances adjacent cards
whether you want it or not.

**Parallel buy lists with balance_with map:** Add a per-card `"balance_with"`
map (e.g. `{"Laboratory": "Market"}`). When buying, if the top candidate has a
balance partner and you own more of it, buy the partner instead. Simple logic
but limited to pairs.
