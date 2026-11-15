/*
 * ga_agent.c -- the evolved genome as an agent of the shared C engine (engine/c/engine.c).
 *
 * C twin of core/agent.py: every decision is answered the same way, so a genome plays the same
 * games here as in Python (tests/test_agent_parity.py). Change both together.
 *
 * Built with the engine into dominion.so by ga/c_bridge.py.
 */
#include <string.h>
#include "engine.h"

/* Genome layout (= ga/c_bridge.py strategy_to_ints). Lists hold card ids, end with -1. */
#define GA_LIST        40
#define GA_E2M_TURN     0
#define GA_M2L_PROV     1
#define GA_M2L_TURN     2
#define GA_CHAPEL_MAX   3
#define GA_SLACK        4
#define GA_BUY          5                        /* 3 lists: early, mid, late */
#define GA_NT         (GA_BUY + 3 * GA_LIST)
#define GA_T          (GA_NT + GA_LIST)
#define GA_CHAPEL     (GA_T + GA_LIST)           /* 3 lists */
#define GA_THRONE     (GA_CHAPEL + 3 * GA_LIST)
#define GA_MINE       (GA_THRONE + GA_LIST)
#define GA_TARGETS    (GA_MINE + GA_LIST)        /* DE_NUM_CARDS ints, -1 = no limit */
#define GA_SIZE       (GA_TARGETS + DE_NUM_CARDS)
#define GA_PASS (-2)
#define GA_STOP (-3)

typedef struct {
    const int *s;
    int actions_played;
} ga_agent;

int ga_strategy_size(void) { return GA_SIZE; }

static int in(const int *a, int n, int v) {
    for (int i = 0; i < n; i++) if (a[i] == v) return 1;
    return 0;
}

static int list_index(const int *list, int v) {
    for (int i = 0; i < GA_LIST && list[i] != -1; i++) if (list[i] == v) return i;
    return -1;
}

static int phase(const ga_agent *a, const de_game *g, int player) {
    int turn = g->p[player].turns + 1;
    if (turn <= a->s[GA_E2M_TURN]) return 0;
    if (g->supply[DE_PROVINCE] > a->s[GA_M2L_PROV] && turn < a->s[GA_M2L_TURN]) return 1;
    return 2;
}

static int keep_rank(const ga_agent *a, int c) {
    int i = list_index(a->s + GA_NT, c);
    if (i >= 0) return 60 - i;
    i = list_index(a->s + GA_T, c);
    if (i >= 0) return 30 - i;
    return 0;
}

static int discard_rank(const ga_agent *a, int c, const de_player *p) {
    switch (c) {
    case DE_CURSE: return 0;
    case DE_ESTATE: return 10;
    case DE_COPPER: return 20;
    case DE_DUCHY: return 30;
    case DE_GARDENS: return 35;
    case DE_SILVER: return 800;
    case DE_GOLD: return 850;
    case DE_PROVINCE: return 32;       /* dead in hand, like Duchy */
    }
    if (DE_CARDS[c].types & DE_ACTION) {
        int copies = 0;
        if (p) for (int i = 0; i < p->hand_n; i++) copies += p->hand[i] == c;
        return (copies > 1 ? 500 : 700) - keep_rank(a, c);
    }
    return 400;
}

static int worst(const ga_agent *a, const int *opts, int n, const de_player *p) {
    int best = opts[0], r = discard_rank(a, opts[0], p);
    for (int i = 1; i < n; i++) {
        int ri = discard_rank(a, opts[i], p);
        if (ri < r) { r = ri; best = opts[i]; }
    }
    return best;
}

static int is_terminal(int c) {
    return (DE_CARDS[c].types & DE_ACTION) && DE_CARDS[c].actions == 0 && c != DE_THRONE_ROOM;
}

static void owned(const de_player *p, int *counts) {
    memset(counts, 0, DE_NUM_CARDS * sizeof(int));
    for (int i = 0; i < p->deck_n; i++) counts[p->deck[i]]++;
    for (int i = 0; i < p->hand_n; i++) counts[p->hand[i]]++;
    for (int i = 0; i < p->discard_n; i++) counts[p->discard[i]]++;
    for (int i = 0; i < p->play_n; i++) counts[p->play[i]]++;
}

static int allowed(const ga_agent *a, int c, const int *own) {
    int target = a->s[GA_TARGETS + c];
    if (target >= 0 && own[c] >= target) return 0;
    if (is_terminal(c)) {
        int terminals = 0, capacity = 0;
        for (int x = 0; x < DE_NUM_CARDS; x++) {
            if (is_terminal(x)) terminals += own[x];
            if ((DE_CARDS[x].types & DE_ACTION) && DE_CARDS[x].actions > 1)
                capacity += own[x] * (DE_CARDS[x].actions - 1);
        }
        if (terminals >= capacity + a->s[GA_SLACK]) return 0;
    }
    return 1;
}

static int gain(const ga_agent *a, const de_game *g, int player, const int *opts, int n) {
    int own[DE_NUM_CARDS];
    owned(&g->p[player], own);
    const int *buy = a->s + GA_BUY + phase(a, g, player) * GA_LIST;
    for (int i = 0; i < GA_LIST && buy[i] != -1; i++)
        if (buy[i] >= 0 && in(opts, n, buy[i]) && allowed(a, buy[i], own)) return buy[i];
    int best = opts[0];
    for (int i = 0; i < n; i++) {
        int c = opts[i];
        if (c != DE_CURSE && (best == DE_CURSE || DE_CARDS[c].cost > DE_CARDS[best].cost)) best = c;
    }
    return best;
}

/* first card of list (up to -1) that is in opts, or -1 */
static int first_in(const int *list, const int *opts, int n) {
    for (int i = 0; i < GA_LIST && list[i] != -1; i++)
        if (list[i] >= 0 && in(opts, n, list[i])) return list[i];
    return -1;
}

static int ga_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass) {
    ga_agent *a = ctx;
    const int *s = a->s;
    const de_player *p = &g->p[me];
    int c;
    (void)may_pass;

    switch (kind) {
    case DE_PLAY_ACTION:
        c = first_in(s + GA_NT, opts, n);
        if (c < 0) c = first_in(s + GA_T, opts, n);
        if (c >= 0) a->actions_played++;
        return c;
    case DE_BUY: {
        int own[DE_NUM_CARDS];
        owned(p, own);
        const int *buy = s + GA_BUY + phase(a, g, me) * GA_LIST;
        for (int i = 0; i < GA_LIST && buy[i] != -1; i++) {
            if (buy[i] == GA_PASS) return -1;
            if (buy[i] >= 0 && in(opts, n, buy[i]) && allowed(a, buy[i], own)) return buy[i];
        }
        return -1;
    }
    case DE_THRONE:
        return first_in(s + GA_THRONE, opts, n);
    case DE_CHAPEL_TRASH: {
        int max = s[GA_CHAPEL_MAX] < 4 ? s[GA_CHAPEL_MAX] : 4;
        if (subject >= max) return -1;
        const int *list = s + GA_CHAPEL + phase(a, g, me) * GA_LIST;
        for (int i = 0; i < GA_LIST && list[i] != -1; i++) {
            if (list[i] == GA_STOP) return -1;
            if (list[i] >= 0 && in(opts, n, list[i])) return list[i];
        }
        return -1;
    }
    case DE_MONEYLENDER_TRASH:
        return DE_COPPER;
    case DE_MINE_TRASH: {
        const int *list = s + GA_MINE;
        static const int gains[3] = {DE_GOLD, DE_SILVER, DE_COPPER};
        for (int i = 0; i < GA_LIST && list[i] != -1; i++) {
            int t = list[i];
            if (t < 0 || !in(opts, n, t)) continue;
            for (int j = 0; j < 3; j++) {
                int gc = DE_CARDS[gains[j]].cost, tc = DE_CARDS[t].cost;
                if (g->supply[gains[j]] > 0 && tc < gc && gc <= tc + 3) return t;
            }
        }
        return -1;
    }
    case DE_MINE_GAIN: {
        static const int gains[3] = {DE_GOLD, DE_SILVER, DE_COPPER};
        for (int j = 0; j < 3; j++)
            if (in(opts, n, gains[j]) && DE_CARDS[gains[j]].cost > DE_CARDS[subject].cost) return gains[j];
        return opts[n - 1];
    }
    case DE_MILITIA_DISCARD:
    case DE_POACHER_DISCARD:
    case DE_BUREAUCRAT_TOPDECK:
        return worst(a, opts, n, p);
    case DE_BANDIT_TRASH:
        return opts[0];
    case DE_CELLAR_DISCARD:
        for (int i = 0; i < n; i++)
            if (!(DE_CARDS[opts[i]].types & (DE_ACTION | DE_TREASURE))) return opts[i];
        return -1;
    case DE_HARBINGER_TOPDECK:
        if (in(opts, n, DE_GOLD)) return DE_GOLD;
        c = first_in(s + GA_NT, opts, n);
        if (c < 0) c = first_in(s + GA_T, opts, n);
        if (c < 0 && in(opts, n, DE_SILVER)) c = DE_SILVER;
        return c;
    case DE_VASSAL_PLAY:
        return (list_index(s + GA_NT, subject) >= 0 || list_index(s + GA_T, subject) >= 0) ? subject : -1;
    case DE_LIBRARY_SKIP:
        return g->actions == 0 ? subject : -1;
    case DE_SENTRY_CARD: {
        const int *list = s + GA_CHAPEL + phase(a, g, me) * GA_LIST;
        for (int i = 0; i < GA_LIST && list[i] != -1 && list[i] != GA_STOP; i++)
            if (list[i] == subject) return DE_SENTRY_TRASH;
        if (!(DE_CARDS[subject].types & (DE_ACTION | DE_TREASURE))) return DE_SENTRY_DISCARD;
        return DE_SENTRY_KEEP;
    }
    case DE_SENTRY_TOP:
        return discard_rank(a, opts[1], NULL) > discard_rank(a, opts[0], NULL) ? opts[1] : opts[0];
    case DE_REMODEL_TRASH: {
        int late = phase(a, g, me) == 2;
        if (late && in(opts, n, DE_GOLD) && g->supply[DE_PROVINCE] > 0) return DE_GOLD;
        if (in(opts, n, DE_CURSE)) return DE_CURSE;
        if (in(opts, n, DE_ESTATE) && !late) return DE_ESTATE;
        return worst(a, opts, n, p);
    }
    case DE_REMODEL_GAIN:
    case DE_WORKSHOP_GAIN:
    case DE_ARTISAN_GAIN:
        return gain(a, g, me, opts, n);
    case DE_ARTISAN_TOPDECK:
        c = first_in(s + GA_NT, opts, n);
        if (c < 0) c = first_in(s + GA_T, opts, n);
        if (c >= 0) return c;
        for (int i = 0; i < n; i++) if (DE_CARDS[opts[i]].types & DE_ACTION) return opts[i];
        return in(opts, n, DE_SILVER) ? DE_SILVER : worst(a, opts, n, p);
    }
    return may_pass ? -1 : opts[0];
}

/* One game; seat 0 = s0. out: vp0, vp1, turns (seat 0), actions played by seat 0 and 1,
 * winners mask, capped. */
static void play_one(const int *s0, const int *s1, uint64_t seed, const int *kingdom, int kn,
                     int *out, uint64_t *trace) {
    de_game g;
    ga_agent a[2] = {{s0, 0}, {s1, 0}};
    de_agent agents[2] = {{ga_decide, &a[0]}, {ga_decide, &a[1]}};
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, agents);
    out[0] = de_score(&g, 0);
    out[1] = de_score(&g, 1);
    out[2] = g.p[0].turns;
    out[3] = a[0].actions_played;
    out[4] = a[1].actions_played;
    out[5] = de_winners(&g);
    out[6] = g.capped | (g.error << 1);
    if (trace) *trace = g.trace;
}

void ga_play_one(const int *s0, const int *s1, uint64_t seed, const int *kingdom, int kn,
                 int *out, uint64_t *trace) {
    play_one(s0, s1, seed, kingdom, kn, out, trace);
}

/* Each seed twice, s1 first then s2 first. Per game (index 2*i and 2*i+1), from s1's side:
 * vp1, vp2, turns, actions1, result (1 win, 0 shared win, -1 loss). */
void ga_play_batch(const int *s1, const int *s2, const uint64_t *seeds, int num, const int *kingdom,
                   int kn, int *vp1, int *vp2, int *turns, int *actions1, int *result) {
    int o[7];
    for (int i = 0; i < num; i++) {
        for (int swap = 0; swap < 2; swap++) {
            int idx = 2 * i + swap;
            play_one(swap ? s2 : s1, swap ? s1 : s2, seeds[i], kingdom, kn, o, NULL);
            int me = swap, mask = o[5];
            vp1[idx] = o[me];
            vp2[idx] = o[1 - me];
            turns[idx] = o[2];
            actions1[idx] = o[3 + me];
            result[idx] = mask == 3 ? 0 : (mask >> me & 1) ? 1 : -1;
        }
    }
}

/* ── search: rollouts on top of a genome (policy/search.py) ──
 * At a buy, the genome's choice competes with its next legal choices and buying nothing: each is
 * played out `rollouts` times from a copy of the game (de_hide + de_resume_buy), both players
 * following genomes (me_s for me, opp_s as the model of the opponent). The same hidden shuffle is
 * used for every candidate of a rollout. The genome's choice is kept unless another candidate
 * scores more than `margin` percent better. Not reentrant into itself: rollouts use plain agents. */
typedef struct {
    ga_agent self;
    const int *opp_s;
    int rollouts, top, margin;
    uint64_t rng;
    int searched, changed;
    int imagine;      /* 1: the opponent's cards are unknown; model them as a mirror of mine */
} ga_search;

/* The opponent as a player can imagine it: a copy of my whole deck, with as many Provinces as
 * they must own (8 - left - mine), shuffled, 5 in hand. Turns are known. */
static void imagine_opponent(de_game *g, int me, uint64_t seed) {
    de_player *o = &g->p[1 - me], *p = &g->p[me];
    int cards[DE_ZONE], n = 0, mine = 0;
    for (int z = 0; z < 4; z++) {
        const int *a = z == 0 ? p->deck : z == 1 ? p->hand : z == 2 ? p->discard : p->play;
        int an = z == 0 ? p->deck_n : z == 1 ? p->hand_n : z == 2 ? p->discard_n : p->play_n;
        for (int i = 0; i < an && n < DE_ZONE; i++) {
            if (a[i] == DE_PROVINCE) { mine++; continue; }
            cards[n++] = a[i];
        }
    }
    int start = g->nplayers == 2 ? 8 : 12;
    int theirs = start - (g->supply[DE_PROVINCE] > 0 ? g->supply[DE_PROVINCE] : 0) - mine;
    for (int i = 0; i < theirs && n < DE_ZONE; i++) cards[n++] = DE_PROVINCE;
    uint64_t r = seed | 1;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(de_rng_next(&r) % (uint64_t)(i + 1));
        int t = cards[i]; cards[i] = cards[j]; cards[j] = t;
    }
    int h = n < 5 ? n : 5;
    memcpy(o->hand, cards, (size_t)h * sizeof(int));
    o->hand_n = h;
    memcpy(o->deck, cards + h, (size_t)(n - h) * sizeof(int));
    o->deck_n = n - h;
    o->discard_n = o->play_n = 0;
}

static int search_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                         const int *opts, int n, int may_pass) {
    ga_search *s = ctx;
    int pick = ga_decide(&s->self, g, kind, me, subject, opts, n, may_pass);
    if (kind != DE_BUY || g->nplayers != 2) return pick;
    int cand[8], nc = 0, own[DE_NUM_CARDS];
    cand[nc++] = pick;
    owned(&g->p[me], own);
    const int *buy = s->self.s + GA_BUY + phase(&s->self, g, me) * GA_LIST;
    for (int i = 0; i < GA_LIST && buy[i] != -1 && nc < s->top; i++) {
        int c = buy[i], dup = 0;
        if (c < 0 || !in(opts, n, c) || !allowed(&s->self, c, own)) continue;
        for (int k = 0; k < nc; k++) dup |= cand[k] == c;
        if (!dup) cand[nc++] = c;
    }
    int has_pass = 0;
    for (int k = 0; k < nc; k++) has_pass |= cand[k] == -1;
    if (!has_pass) cand[nc++] = -1;
    if (nc == 1) return pick;
    s->searched++;
    double v[8] = {0};
    for (int r = 0; r < s->rollouts; r++) {
        uint64_t seed = de_rng_next(&s->rng);
        for (int k = 0; k < nc; k++) {
            de_game c = *g;
            ga_agent ra[2];
            ra[me] = (ga_agent){s->self.s, 0};
            ra[1 - me] = (ga_agent){s->opp_s, 0};
            de_agent rag[2] = {{ga_decide, &ra[0]}, {ga_decide, &ra[1]}};
            if (s->imagine) imagine_opponent(&c, me, seed ^ 0x5bd1e995);
            de_hide(&c, me, seed);
            de_resume_buy(&c, rag, cand[k]);
            int w = de_winners(&c);
            v[k] += w == 3 ? 0.5 : (w >> me & 1) ? 1.0 : 0.0;
        }
    }
    int best = 0;
    for (int k = 1; k < nc; k++) if (v[k] > v[best]) best = k;
    if (best && v[best] - v[0] > s->margin * s->rollouts / 100.0) { s->changed++; return cand[best]; }
    return pick;
}

/* Search (genome s, opponent model opp_model) vs plain genome opp: each seed twice (search in
 * seat 0, then seat 1). result[2*i+swap]: 1 win, 0 shared, -1 loss for the search side.
 * stats: buys searched, buys changed. */
void ga_search_batch(const int *s, const int *opp_model, const int *opp, const uint64_t *seeds,
                     int num, const int *kingdom, int kn, int rollouts, int top, int margin,
                     uint64_t search_seed, int imagine, int *result, int *stats) {
    stats[0] = stats[1] = 0;
    for (int i = 0; i < num; i++) {
        for (int swap = 0; swap < 2; swap++) {
            de_game g;
            ga_search srch = {{s, 0}, opp_model, rollouts, top, margin,
                              (search_seed ^ (seeds[i] * 0x9E3779B97F4A7C15ULL)) | 1, 0, 0, imagine};
            ga_agent plain = {opp, 0};
            de_agent agents[2];
            agents[swap] = (de_agent){search_decide, &srch};
            agents[1 - swap] = (de_agent){ga_decide, &plain};
            de_init(&g, kingdom, kn, 2, seeds[i]);
            de_run(&g, agents);
            int mask = de_winners(&g);
            result[2 * i + swap] = mask == 3 ? 0 : (mask >> swap & 1) ? 1 : -1;
            stats[0] += srch.searched;
            stats[1] += srch.changed;
        }
    }
}
