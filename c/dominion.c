/*
 * dominion.c -- Fast C game engine for Dominion GA evaluation.
 *
 * Build: gcc -O3 -shared -fPIC -o dominion.so dominion.c
 *
 * All game logic ported from engine.py. Uses integer card IDs and flat arrays
 * for maximum speed. Called from Python via ctypes (see c_bridge.py).
 */

#include <stdint.h>
#include <string.h>

/* ── Card constants (must match cards.py CARD_ID) ── */
#define COPPER       0
#define SILVER       1
#define GOLD         2
#define ESTATE       3
#define DUCHY        4
#define PROVINCE     5
#define VILLAGE      6
#define SMITHY       7
#define MARKET       8
#define LABORATORY   9
#define FESTIVAL    10
#define CHAPEL      11
#define THRONE_ROOM 12
#define COUNCIL_ROOM 13
#define MONEYLENDER 14
#define GARDENS     15
#define MINE        16
#define MERCHANT    17
#define CURSE       18
#define MILITIA     19
#define WITCH       20
#define MOAT        21

#define NUM_CARDS   22
#define PASS_ID     22
#define STOP_ID     23

/* Card types */
#define TYPE_TREASURE 1
#define TYPE_VICTORY  2
#define TYPE_ACTION   3

/* Special card codes */
#define SPECIAL_NONE        0
#define SPECIAL_CHAPEL      1
#define SPECIAL_THRONE_ROOM 2
#define SPECIAL_MONEYLENDER 3
#define SPECIAL_GARDENS     4
#define SPECIAL_MINE        5
#define SPECIAL_MERCHANT    6
#define SPECIAL_MILITIA     7
#define SPECIAL_WITCH       8
#define SPECIAL_MOAT        9

/* ── Card data arrays (initialized from Python) ── */
static int card_cost[NUM_CARDS];
static int card_coins[NUM_CARDS];
static int card_vp[NUM_CARDS];
static int card_draw[NUM_CARDS];
static int card_actions[NUM_CARDS];
static int card_buys[NUM_CARDS];
static int card_type[NUM_CARDS];
static int card_special[NUM_CARDS];

/* ── Strategy layout offsets ── */
/* Strategy is a flat int array. Priority lists are -1 terminated. */
#define S_EARLY_TO_MID_TURN     0
#define S_MID_TO_LATE_PROV      1
#define S_MID_TO_LATE_TURN      2
#define S_CHAPEL_MAX_TRASH      3
#define S_EARLY_BUY             4    /* 20 slots */
#define S_MID_BUY              24    /* 20 slots */
#define S_LATE_BUY             44    /* 20 slots */
#define S_ACTION               64    /* 16 slots (single, shared) */
#define S_EARLY_CHAPEL         80    /* 6 slots */
#define S_MID_CHAPEL           86    /* 6 slots */
#define S_LATE_CHAPEL          92    /* 6 slots */
#define S_THRONE_ROOM_PRIO     98    /* 12 slots */
#define S_MINE_TRASH_PRIO     110    /* 4 slots */
#define S_BUY_TARGETS         114    /* 20 slots: (card_id, max) pairs, -1 terminated */
#define STRATEGY_SIZE         134

/* ── Limits ── */
#define MAX_DECK   200
#define MAX_SUPPLY  20
#define TURN_CAP    40

/* ── PRNG: xorshift64 ── */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* ── Player state ── */
typedef struct {
    int deck[MAX_DECK];
    int deck_n;
    int hand[MAX_DECK];
    int hand_n;
    int discard[MAX_DECK];
    int discard_n;
    int play_area[MAX_DECK];
    int play_n;
    int actions;
    int buys;
    int coins;
    int turn;
    uint64_t rng;
} Player;

/* ── Array helpers ── */

/* Fisher-Yates shuffle */
static void shuffle(int *arr, int n, uint64_t *rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(xorshift64(rng) % (uint64_t)(i + 1));
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* Remove first occurrence of val from arr[0..n-1], shift left, return 1 if found */
static int arr_remove(int *arr, int *n, int val) {
    for (int i = 0; i < *n; i++) {
        if (arr[i] == val) {
            /* shift left */
            for (int j = i; j < *n - 1; j++)
                arr[j] = arr[j + 1];
            (*n)--;
            return 1;
        }
    }
    return 0;
}

/* Check if val is in arr[0..n-1] */
static int arr_contains(const int *arr, int n, int val) {
    for (int i = 0; i < n; i++)
        if (arr[i] == val) return 1;
    return 0;
}

/* Count occurrences of val in arr[0..n-1] */
static int arr_count(const int *arr, int n, int val) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] == val) c++;
    return c;
}

/* ── Draw cards ── */
static void draw_cards(Player *p, int n) {
    for (int i = 0; i < n; i++) {
        if (p->deck_n == 0) {
            if (p->discard_n == 0) return;
            /* Move discard to deck */
            memcpy(p->deck, p->discard, p->discard_n * sizeof(int));
            p->deck_n = p->discard_n;
            p->discard_n = 0;
            shuffle(p->deck, p->deck_n, &p->rng);
        }
        p->hand[p->hand_n++] = p->deck[--p->deck_n];
    }
}

/* ── Resolve action: play card from hand, apply effects ── */
static void resolve_action(Player *p, int card_id) {
    arr_remove(p->hand, &p->hand_n, card_id);
    p->play_area[p->play_n++] = card_id;
    p->actions--;
    p->actions += card_actions[card_id];
    p->coins += card_coins[card_id];
    p->buys += card_buys[card_id];
    draw_cards(p, card_draw[card_id]);
}

/* ── Apply action effects without moving card (Throne Room 2nd play) ── */
static void apply_action_effects(Player *p, int card_id) {
    p->actions += card_actions[card_id];
    p->coins += card_coins[card_id];
    p->buys += card_buys[card_id];
    draw_cards(p, card_draw[card_id]);
}

/* ── Auto-play treasures ── */
static void auto_play_treasures(Player *p) {
    int had_silver = 0;
    /* Scan backwards so removal doesn't skip elements */
    for (int i = p->hand_n - 1; i >= 0; i--) {
        int c = p->hand[i];
        if (card_type[c] == TYPE_TREASURE) {
            if (c == SILVER) had_silver = 1;
            p->play_area[p->play_n++] = c;
            p->coins += card_coins[c];
            /* Remove from hand by shifting */
            for (int j = i; j < p->hand_n - 1; j++)
                p->hand[j] = p->hand[j + 1];
            p->hand_n--;
        }
    }
    /* Merchant bonus: +$1 per Merchant in play area for first Silver */
    if (had_silver) {
        p->coins += arr_count(p->play_area, p->play_n, MERCHANT);
    }
}

/* ── Buy card ── */
static void buy_card(Player *p, int card_id, int *supply) {
    supply[card_id]--;
    p->discard[p->discard_n++] = card_id;
    p->coins -= card_cost[card_id];
    p->buys--;
}

/* ── Trash card from hand ── */
static void trash_card(Player *p, int card_id) {
    arr_remove(p->hand, &p->hand_n, card_id);
    /* We don't need to track trash pile in C — not used for scoring */
}

/* ── Moneylender: trash Copper, +$3 ── */
static int play_moneylender(Player *p) {
    if (arr_contains(p->hand, p->hand_n, COPPER)) {
        trash_card(p, COPPER);
        p->coins += 3;
        return 1;
    }
    return 0;
}

/* ── Mine: trash treasure, gain better one to hand ── */
static void play_mine(Player *p, const int *strat, int *supply) {
    const int *prio = strat + S_MINE_TRASH_PRIO;
    for (int i = 0; prio[i] != -1 && i < 4; i++) {
        int treasure = prio[i];
        if (!arr_contains(p->hand, p->hand_n, treasure)) continue;
        int tcost = card_cost[treasure];
        int max_gain_cost = tcost + 3;
        /* Try Gold, Silver, Copper (highest first) */
        int gains[] = {GOLD, SILVER, COPPER};
        for (int j = 0; j < 3; j++) {
            int g = gains[j];
            int gcost = card_cost[g];
            if (gcost <= max_gain_cost && gcost > tcost && supply[g] > 0) {
                trash_card(p, treasure);
                supply[g]--;
                p->hand[p->hand_n++] = g;
                return;
            }
        }
    }
}

/* ── Chapel: trash up to N cards from hand ── */
static void play_chapel(Player *p, const int *strat, int phase) {
    int max_trash = strat[S_CHAPEL_MAX_TRASH];
    if (max_trash > 4) max_trash = 4;
    int trashed = 0;

    const int *prio;
    if (phase == 0)      prio = strat + S_EARLY_CHAPEL;
    else if (phase == 1) prio = strat + S_MID_CHAPEL;
    else                 prio = strat + S_LATE_CHAPEL;

    for (int i = 0; i < 6 && prio[i] != -1; i++) {
        int card_id = prio[i];
        if (card_id == STOP_ID || trashed >= max_trash) break;
        while (arr_contains(p->hand, p->hand_n, card_id) && trashed < max_trash) {
            trash_card(p, card_id);
            trashed++;
        }
    }
}

/* ── Check if player has Moat in hand ── */
static int has_moat(const Player *p) {
    return arr_contains(p->hand, p->hand_n, MOAT);
}

/* Forward declaration */
static int get_phase(int turn, int provinces_remaining, const int *strat);

/* ── Rank an action card for militia discard using evolved action priority ──
 * Lower rank = discard first (least valuable).
 * Higher position in the priority list = more valuable = higher keep rank.
 */
static int action_keep_rank(int card_id, const int *opp_strat) {
    const int *prio = opp_strat + S_ACTION;
    for (int i = 0; i < 16 && prio[i] != -1; i++) {
        if (prio[i] == card_id)
            return 100 + (16 - i);  /* top = 116, lowest = 101 */
    }
    return 0;  /* unknown action: discard first */
}

/* ── Militia discard: opponent discards down to 3 cards ── */
/* Fixed heuristic: Curse > Estate > Copper > Duchy > duplicate actions
 * (worst first) > unique actions (worst first) > Silver > Gold > Province */
static void militia_discard(Player *p, const int *opp_strat, const int *supply) {
    if (p->hand_n <= 3) return;

    while (p->hand_n > 3) {
        int best_idx = -1;
        int best_rank = 999;

        /* Count action cards in hand to detect duplicates */
        int action_count[NUM_CARDS];
        memset(action_count, 0, sizeof(action_count));
        for (int i = 0; i < p->hand_n; i++) {
            int c = p->hand[i];
            if (card_type[c] == TYPE_ACTION)
                action_count[c]++;
        }

        for (int i = 0; i < p->hand_n; i++) {
            int c = p->hand[i];
            int rank;

            if (c == CURSE) rank = 0;
            else if (c == ESTATE) rank = 10;
            else if (c == COPPER) rank = 20;
            else if (c == DUCHY) rank = 30;
            else if (card_type[c] == TYPE_ACTION) {
                int keep = action_keep_rank(c, opp_strat);
                if (action_count[c] > 1) {
                    /* Duplicate action: discard before unique actions */
                    rank = 500 - keep;
                } else {
                    /* Unique action: keep longer */
                    rank = 700 - keep;
                }
            }
            else if (c == SILVER) rank = 800;
            else if (c == GOLD) rank = 850;
            else if (c == PROVINCE) rank = 900;
            else rank = 400;

            if (rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;

        /* Move card from hand to discard */
        int card = p->hand[best_idx];
        for (int j = best_idx; j < p->hand_n - 1; j++)
            p->hand[j] = p->hand[j + 1];
        p->hand_n--;
        p->discard[p->discard_n++] = card;
    }
}

/* ── Play Militia attack against opponent ── */
static void play_militia_attack(Player *opponent, const int *opp_strat, const int *supply) {
    if (has_moat(opponent)) return;
    militia_discard(opponent, opp_strat, supply);
}

/* ── Play Witch attack against opponent ── */
static void play_witch_attack(Player *opponent, int *supply) {
    if (has_moat(opponent)) return;
    if (supply[CURSE] > 0) {
        supply[CURSE]--;
        opponent->discard[opponent->discard_n++] = CURSE;
    }
}

/* ── Throne Room: double an action ── */
static void play_throne_room(Player *p, const int *strat, int phase, int *supply,
                             Player *opponent, const int *opp_strat) {
    /* Find highest-priority action in hand */
    const int *prio = strat + S_THRONE_ROOM_PRIO;
    int target = -1;
    for (int i = 0; i < 12 && prio[i] != -1; i++) {
        int c = prio[i];
        if (arr_contains(p->hand, p->hand_n, c) && card_type[c] == TYPE_ACTION) {
            target = c;
            break;
        }
    }
    if (target == -1) return;

    /* Move target to play area */
    arr_remove(p->hand, &p->hand_n, target);
    p->play_area[p->play_n++] = target;

    /* Apply effects twice */
    for (int t = 0; t < 2; t++) {
        apply_action_effects(p, target);
        int sp = card_special[target];
        if (sp == SPECIAL_CHAPEL)
            play_chapel(p, strat, phase);
        else if (sp == SPECIAL_MONEYLENDER)
            play_moneylender(p);
        else if (sp == SPECIAL_MINE)
            play_mine(p, strat, supply);
        else if (sp == SPECIAL_MILITIA && opponent)
            play_militia_attack(opponent, opp_strat, supply);
        else if (sp == SPECIAL_WITCH && opponent)
            play_witch_attack(opponent, supply);
    }
}

/* ── Handle special card after resolving action ── */
static void handle_special(Player *p, int card_id, const int *strat,
                           int phase, int *supply,
                           Player *opponent, const int *opp_strat) {
    int sp = card_special[card_id];
    if (sp == SPECIAL_CHAPEL)
        play_chapel(p, strat, phase);
    else if (sp == SPECIAL_MONEYLENDER)
        play_moneylender(p);
    else if (sp == SPECIAL_THRONE_ROOM)
        play_throne_room(p, strat, phase, supply, opponent, opp_strat);
    else if (sp == SPECIAL_MINE)
        play_mine(p, strat, supply);
    else if (sp == SPECIAL_MILITIA && opponent)
        play_militia_attack(opponent, opp_strat, supply);
    else if (sp == SPECIAL_WITCH && opponent)
        play_witch_attack(opponent, supply);
}

/* ── Determine phase: 0=early, 1=mid, 2=late ── */
static int get_phase(int turn, int provinces_remaining, const int *strat) {
    if (turn <= strat[S_EARLY_TO_MID_TURN])
        return 0;
    else if (provinces_remaining > strat[S_MID_TO_LATE_PROV]
             && turn < strat[S_MID_TO_LATE_TURN])
        return 1;
    else
        return 2;
}

/* ── Play action phase (single merged priority list) ── */
static void play_action_phase(Player *p, const int *strat, int *supply,
                              Player *opponent, const int *opp_strat) {
    int phase = get_phase(p->turn, supply[PROVINCE], strat);

    const int *prio = strat + S_ACTION;

    while (p->actions > 0) {
        int played = 0;
        for (int i = 0; i < 16 && prio[i] != -1; i++) {
            int c = prio[i];
            if (p->actions > 0 && arr_contains(p->hand, p->hand_n, c)) {
                resolve_action(p, c);
                handle_special(p, c, strat, phase, supply, opponent, opp_strat);
                played = 1;
                break; /* re-scan from top */
            }
        }
        if (!played) break;
    }
}

/* ── Play buy phase ── */
static void play_buy_phase(Player *p, const int *strat, int *supply) {
    auto_play_treasures(p);

    int phase = get_phase(p->turn, supply[PROVINCE], strat);

    const int *buy_prio;
    if (phase == 0)      buy_prio = strat + S_EARLY_BUY;
    else if (phase == 1) buy_prio = strat + S_MID_BUY;
    else                 buy_prio = strat + S_LATE_BUY;

    /* Parse buy targets into a lookup array */
    int buy_target[NUM_CARDS];
    memset(buy_target, -1, sizeof(buy_target)); /* -1 = no limit */
    {
        const int *bt = strat + S_BUY_TARGETS;
        for (int i = 0; i < 20 && bt[i] != -1; i += 2) {
            int cid = bt[i];
            int maxn = bt[i + 1];
            if (cid >= 0 && cid < NUM_CARDS)
                buy_target[cid] = maxn;
        }
    }

    /* Count owned cards for buy target checks */
    int has_targets = 0;
    for (int i = 0; i < NUM_CARDS; i++)
        if (buy_target[i] >= 0) { has_targets = 1; break; }

    int owned[NUM_CARDS];
    if (has_targets) {
        memset(owned, 0, sizeof(owned));
        for (int i = 0; i < p->deck_n; i++) owned[p->deck[i]]++;
        for (int i = 0; i < p->hand_n; i++) owned[p->hand[i]]++;
        for (int i = 0; i < p->discard_n; i++) owned[p->discard[i]]++;
        for (int i = 0; i < p->play_n; i++) owned[p->play_area[i]]++;
    }

    while (p->buys > 0) {
        int bought = 0;
        for (int i = 0; i < 20 && buy_prio[i] != -1; i++) {
            int c = buy_prio[i];
            if (c == PASS_ID) {
                p->buys = 0;
                break;
            }
            if (c < 0 || c >= NUM_CARDS) continue;
            if (supply[c] <= 0) continue;
            if (card_cost[c] > p->coins) continue;

            /* Buy target limit */
            if (has_targets && buy_target[c] >= 0) {
                if (owned[c] >= buy_target[c]) continue;
            }

            buy_card(p, c, supply);
            if (has_targets) owned[c]++;
            bought = 1;
            break; /* re-scan from top */
        }
        if (!bought) break;
    }
}

/* ── Cleanup: discard hand + play area, draw 5 ── */
static void cleanup(Player *p) {
    for (int i = 0; i < p->hand_n; i++)
        p->discard[p->discard_n++] = p->hand[i];
    for (int i = 0; i < p->play_n; i++)
        p->discard[p->discard_n++] = p->play_area[i];
    p->hand_n = 0;
    p->play_n = 0;
    draw_cards(p, 5);
}

/* ── Game over check ── */
static int is_game_over(const int *supply, int num_supply, int turn) {
    if (supply[PROVINCE] == 0) return 1;
    int empty = 0;
    for (int i = 0; i < num_supply; i++)
        if (supply[i] == 0) empty++;  /* -1 = pile doesn't exist, not counted */
    if (empty >= 3) return 1;
    if (turn >= TURN_CAP) return 1;
    return 0;
}

/* ── Count VP ── */
static int count_vp(const Player *p) {
    int total_cards = p->deck_n + p->hand_n + p->discard_n + p->play_n;
    int vp = 0;
    int gardens = 0;

    /* Sum VP from all zones */
    for (int i = 0; i < p->deck_n; i++) {
        vp += card_vp[p->deck[i]];
        if (p->deck[i] == GARDENS) gardens++;
    }
    for (int i = 0; i < p->hand_n; i++) {
        vp += card_vp[p->hand[i]];
        if (p->hand[i] == GARDENS) gardens++;
    }
    for (int i = 0; i < p->discard_n; i++) {
        vp += card_vp[p->discard[i]];
        if (p->discard[i] == GARDENS) gardens++;
    }
    for (int i = 0; i < p->play_n; i++) {
        vp += card_vp[p->play_area[i]];
        if (p->play_area[i] == GARDENS) gardens++;
    }

    if (gardens)
        vp += gardens * (total_cards / 10);

    return vp;
}

/* ── Initialize a player with starting deck ── */
static void init_player(Player *p, uint64_t seed) {
    memset(p, 0, sizeof(Player));
    p->rng = seed;
    if (p->rng == 0) p->rng = 1; /* xorshift can't have state 0 */

    /* 7 Copper + 3 Estate */
    for (int i = 0; i < 7; i++) p->deck[i] = COPPER;
    for (int i = 7; i < 10; i++) p->deck[i] = ESTATE;
    p->deck_n = 10;

    shuffle(p->deck, p->deck_n, &p->rng);
    draw_cards(p, 5);
}

/* ── Initialize supply ── */
static int init_supply(int *supply, const int *kingdom_ids, int kingdom_n,
                       int num_players) {
    /* -1 = pile does not exist (not in this game's supply) */
    for (int i = 0; i < MAX_SUPPLY; i++) supply[i] = -1;

    int starting_coppers = 7 * num_players;
    int starting_estates = 3 * num_players;

    supply[COPPER] = 60 - starting_coppers;
    supply[SILVER] = 40;
    supply[GOLD] = 30;
    supply[ESTATE] = 12 - starting_estates;
    supply[DUCHY] = 12;
    supply[PROVINCE] = 12;

    int num_supply = 6; /* base piles: Copper through Province (IDs 0-5) */

    int has_attack = 0;
    for (int i = 0; i < kingdom_n; i++) {
        int cid = kingdom_ids[i];
        if (cid >= 0 && cid < NUM_CARDS) {
            if (card_type[cid] == TYPE_VICTORY)
                supply[cid] = 12;
            else
                supply[cid] = 10;
            if (cid >= num_supply) num_supply = cid + 1;
            if (card_special[cid] == SPECIAL_MILITIA || card_special[cid] == SPECIAL_WITCH)
                has_attack = 1;
        }
    }

    /* Add Curse pile if any attack card is in kingdom */
    if (has_attack) {
        supply[CURSE] = 10 * (num_players - 1);
        if (CURSE >= num_supply) num_supply = CURSE + 1;
    }

    return num_supply;
}

/* ══════════════════════════════════════════════════════════════
 *  EXPORTED FUNCTIONS (called from Python via ctypes)
 * ══════════════════════════════════════════════════════════════ */

/* Initialize card data from Python arrays.
 * data layout: 8 ints per card (cost, coins, vp, draw, actions, buys, type, special)
 */
void init_cards(const int *data) {
    for (int i = 0; i < NUM_CARDS; i++) {
        int base = i * 8;
        card_cost[i]    = data[base + 0];
        card_coins[i]   = data[base + 1];
        card_vp[i]      = data[base + 2];
        card_draw[i]    = data[base + 3];
        card_actions[i] = data[base + 4];
        card_buys[i]    = data[base + 5];
        card_type[i]    = data[base + 6];
        card_special[i] = data[base + 7];
    }
}

/* Play a batch of 2-player games.
 * Each seed is played TWICE (strat1 as P1, then strat1 as P2) for fairness.
 * Output arrays must be pre-allocated with 2*num_games entries.
 */
void play_games_batch(
    const int *strat1, const int *strat2,
    const uint64_t *seeds, int num_games,
    const int *kingdom_ids, int kingdom_n,
    int *out_vp1, int *out_vp2, int *out_turns
) {
    for (int g = 0; g < num_games; g++) {
        uint64_t master_rng = seeds[g];
        if (master_rng == 0) master_rng = 1;

        /* Derive two player seeds from master */
        uint64_t seed_a = xorshift64(&master_rng);
        uint64_t seed_b = xorshift64(&master_rng);

        /* ── Game 1: strat1 as P1, strat2 as P2 ── */
        {
            int supply[MAX_SUPPLY];
            int num_supply = init_supply(supply, kingdom_ids, kingdom_n, 2);
            Player p1, p2;
            init_player(&p1, seed_a);
            init_player(&p2, seed_b);

            int round_num = 0;
            int game_over = 0;
            while (!game_over) {
                round_num++;

                /* P1 turn */
                if (is_game_over(supply, num_supply, round_num - 1)) {
                    /* Check with turn = round_num - 1 to match Python:
                     * Python checks turn_cap=40 against player.turn which
                     * hasn't been set yet for this round. Actually Python
                     * checks is_game_over(player, turn_cap=40) before setting
                     * player.turn. The player.turn is from the PREVIOUS round.
                     * For round 1, player.turn = 0 (initial). */
                    game_over = 1; break;
                }
                p1.turn = round_num;
                p1.actions = 1; p1.buys = 1; p1.coins = 0;
                play_action_phase(&p1, strat1, supply, &p2, strat2);
                play_buy_phase(&p1, strat1, supply);
                cleanup(&p1);

                /* P2 turn */
                if (is_game_over(supply, num_supply, round_num - 1)) {
                    game_over = 1; break;
                }
                p2.turn = round_num;
                p2.actions = 1; p2.buys = 1; p2.coins = 0;
                play_action_phase(&p2, strat2, supply, &p1, strat1);
                play_buy_phase(&p2, strat2, supply);
                cleanup(&p2);
            }

            int idx = g * 2;
            out_vp1[idx] = count_vp(&p1);
            out_vp2[idx] = count_vp(&p2);
            out_turns[idx] = round_num - 1;
        }

        /* ── Game 2: strat2 as P1, strat1 as P2 (swap seats) ── */
        {
            /* Re-derive seeds for consistency */
            uint64_t mr2 = seeds[g];
            if (mr2 == 0) mr2 = 1;
            uint64_t sa2 = xorshift64(&mr2);
            uint64_t sb2 = xorshift64(&mr2);

            int supply[MAX_SUPPLY];
            int num_supply = init_supply(supply, kingdom_ids, kingdom_n, 2);
            Player p1, p2;
            init_player(&p1, sa2);
            init_player(&p2, sb2);

            int round_num = 0;
            int game_over = 0;
            while (!game_over) {
                round_num++;

                if (is_game_over(supply, num_supply, round_num - 1)) {
                    game_over = 1; break;
                }
                p1.turn = round_num;
                p1.actions = 1; p1.buys = 1; p1.coins = 0;
                play_action_phase(&p1, strat2, supply, &p2, strat1);
                play_buy_phase(&p1, strat2, supply);
                cleanup(&p1);

                if (is_game_over(supply, num_supply, round_num - 1)) {
                    game_over = 1; break;
                }
                p2.turn = round_num;
                p2.actions = 1; p2.buys = 1; p2.coins = 0;
                play_action_phase(&p2, strat1, supply, &p1, strat2);
                play_buy_phase(&p2, strat1, supply);
                cleanup(&p2);
            }

            /* In game 2, strat1 is P2 */
            int idx = g * 2 + 1;
            out_vp1[idx] = count_vp(&p2);  /* strat1's VP (was P2) */
            out_vp2[idx] = count_vp(&p1);  /* strat2's VP (was P1) */
            out_turns[idx] = round_num - 1;
        }
    }
}
