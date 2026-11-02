// AM'S EYES IN C: scene_am.levels (AM's shelf, look.STEPS plus -x and 0, up to DEPTH 3 steps) computed exactly as the
// Python does it, only faster.  The same steps (their order does not matter: every level is sorted), the same int64
// wrap-around (numpy's), floor division and a remainder with the divisor's sign (Python's), the same filter
// (|y| < LIM on every input, numpy's abs, which leaves INT64_MIN negative), the same dedupe (a row already seen at a
// shallower depth is not new), the same order (rows sorted lexicographically as signed int64, as numpy.unique(axis=0)),
// the same stops (an empty level; a level over 400,000 rows is the last).  Duplicates are found by hashing, not sorting.
//     clang -O2 -fwrapv -shared -o libam_eyes.dylib am_eyes.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t i64;
typedef uint64_t u64;

#define DEPTH 3
#define LIM ((i64)1 << 40)
#define STOP 400000

enum { ADD, SUB, OR, XOR, MAX, MIN, AND, MUL, FDIV, FMOD, SHL, SHR, ABS, SQ, CUBE, NEG, ZERO };
typedef struct { int op; i64 c; } Step;
static Step STEPS[160];
static int NSTEPS = 0;

static void build_steps(void) {
    if (NSTEPS) return;
    for (i64 c = 1; c <= 12; c++) {
        STEPS[NSTEPS++] = (Step){ADD, c};
        STEPS[NSTEPS++] = (Step){SUB, c};
        STEPS[NSTEPS++] = (Step){OR, c};
        STEPS[NSTEPS++] = (Step){XOR, c};
        STEPS[NSTEPS++] = (Step){MAX, c};
        STEPS[NSTEPS++] = (Step){MIN, 3 * c};
        STEPS[NSTEPS++] = (Step){AND, 5 * c};
        if (c >= 2) {
            STEPS[NSTEPS++] = (Step){MUL, c};
            STEPS[NSTEPS++] = (Step){FDIV, c};
            STEPS[NSTEPS++] = (Step){FMOD, c};
        }
    }
    for (i64 k = 1; k <= 4; k++) {
        STEPS[NSTEPS++] = (Step){SHL, k};
        STEPS[NSTEPS++] = (Step){SHR, k};
    }
    STEPS[NSTEPS++] = (Step){ABS, 0};
    STEPS[NSTEPS++] = (Step){SQ, 0};
    STEPS[NSTEPS++] = (Step){CUBE, 0};
    STEPS[NSTEPS++] = (Step){NEG, 0};
    STEPS[NSTEPS++] = (Step){ZERO, 0};
}

static inline i64 wmul(i64 a, i64 b) { return (i64)((u64)a * (u64)b); }
static inline i64 wneg(i64 a) { return (i64)(0 - (u64)a); }
static inline i64 fdiv(i64 a, i64 c) { i64 q = a / c; if ((a % c) != 0 && ((a < 0) != (c < 0))) q--; return q; }
static inline i64 fmodp(i64 a, i64 c) { i64 r = a % c; if (r != 0 && ((r < 0) != (c < 0))) r += c; return r; }

static inline i64 apply(Step s, i64 x) {
    switch (s.op) {
        case ADD: return (i64)((u64)x + (u64)s.c);
        case SUB: return (i64)((u64)x - (u64)s.c);
        case OR: return x | s.c;
        case XOR: return x ^ s.c;
        case MAX: return x > s.c ? x : s.c;
        case MIN: return x < s.c ? x : s.c;
        case AND: return x & s.c;
        case MUL: return wmul(x, s.c);
        case FDIV: return fdiv(x, s.c);
        case FMOD: return fmodp(x, s.c);
        case SHL: return (i64)((u64)x << s.c);
        case SHR: return fdiv(x, (i64)1 << s.c);
        case ABS: return x < 0 ? wneg(x) : x;
        case SQ: return wmul(x, x);
        case CUBE: return wmul(wmul(x, x), x);
        case NEG: return wneg(x);
        default: return 0;
    }
}

static inline int keep(i64 y) {                       // numpy: np.abs(y) < LIM, with abs(INT64_MIN) == INT64_MIN
    i64 a = y < 0 ? wneg(y) : y;
    return a < LIM;
}

// ---- rows: an arena of n-wide rows, a hash set of row indices
static int N;
static i64 *ARENA = NULL;
static size_t ACAP = 0, AROWS = 0;
static int64_t *HT = NULL;                            // slots: row index + 1, 0 = empty
static size_t HCAP = 0, HUSED = 0;

static inline u64 rowhash(const i64 *r) {
    u64 h = 1469598103934665603ULL;
    for (int i = 0; i < N; i++) { h ^= (u64)r[i]; h *= 1099511628211ULL; h ^= h >> 29; }
    return h;
}

static void ht_grow(void);

static int64_t ht_find_or_add(const i64 *r, int *added) {   // returns the row index; *added when it is new
    if (HUSED * 2 >= HCAP) ht_grow();
    u64 h = rowhash(r);
    size_t m = HCAP - 1, i = (size_t)h & m;
    while (HT[i]) {
        i64 *q = ARENA + (size_t)(HT[i] - 1) * N;
        if (memcmp(q, r, sizeof(i64) * N) == 0) { *added = 0; return HT[i] - 1; }
        i = (i + 1) & m;
    }
    if (AROWS == ACAP) {
        ACAP = ACAP ? ACAP * 2 : 1024;
        ARENA = realloc(ARENA, ACAP * N * sizeof(i64));
    }
    memcpy(ARENA + AROWS * N, r, sizeof(i64) * N);
    HT[i] = (int64_t)AROWS + 1;
    HUSED++;
    *added = 1;
    return (int64_t)AROWS++;
}

static void ht_grow(void) {
    size_t ncap = HCAP ? HCAP * 2 : 1 << 16;
    int64_t *nt = calloc(ncap, sizeof(int64_t));
    for (size_t j = 0; j < HCAP; j++) {
        if (!HT[j]) continue;
        u64 h = rowhash(ARENA + (size_t)(HT[j] - 1) * N);
        size_t i = (size_t)h & (ncap - 1);
        while (nt[i]) i = (i + 1) & (ncap - 1);
        nt[i] = HT[j];
    }
    free(HT);
    HT = nt;
    HCAP = ncap;
}

static int cmp_rows(const void *a, const void *b) {
    const i64 *x = ARENA + (size_t)(*(const int64_t *)a) * N, *y = ARENA + (size_t)(*(const int64_t *)b) * N;
    for (int i = 0; i < N; i++) {
        if (x[i] < y[i]) return -1;
        if (x[i] > y[i]) return 1;
    }
    return 0;
}

// ---- the result: per depth, the sorted row indices
static int64_t *LEVEL[DEPTH + 1];
static int64_t LCOUNT[DEPTH + 1];
static int NLEVELS = 0;

void am_eyes_free(void) {
    free(ARENA); ARENA = NULL; ACAP = AROWS = 0;
    free(HT); HT = NULL; HCAP = HUSED = 0;
    for (int d = 0; d <= DEPTH; d++) { free(LEVEL[d]); LEVEL[d] = NULL; LCOUNT[d] = 0; }
    NLEVELS = 0;
}

// compute the levels of xs[0..n-1]; returns the number of levels (depth 0 included); counts per level in counts[]
int am_eyes_levels(const i64 *xs, int n, int64_t *counts) {
    build_steps();
    am_eyes_free();
    N = n;
    int added;
    i64 *row = malloc(sizeof(i64) * n);
    int64_t r0 = ht_find_or_add(xs, &added);
    LEVEL[0] = malloc(sizeof(int64_t));
    LEVEL[0][0] = r0;
    LCOUNT[0] = 1;
    NLEVELS = 1;
    for (int d = 1; d <= DEPTH; d++) {
        int64_t prev = LCOUNT[d - 1];
        size_t cap = 1024, cnt = 0;
        int64_t *lv = malloc(cap * sizeof(int64_t));
        for (int64_t p = 0; p < prev; p++) {
            for (int s = 0; s < NSTEPS; s++) {
                const i64 *src = ARENA + (size_t)LEVEL[d - 1][p] * N;   // re-read: the arena may move
                int ok = 1;
                for (int i = 0; i < n; i++) {
                    i64 y = apply(STEPS[s], src[i]);
                    if (!keep(y)) { ok = 0; break; }
                    row[i] = y;
                }
                if (!ok) continue;
                int64_t idx = ht_find_or_add(row, &added);
                if (!added) continue;
                if (cnt == cap) { cap *= 2; lv = realloc(lv, cap * sizeof(int64_t)); }
                lv[cnt++] = idx;
            }
        }
        if (cnt == 0) { free(lv); break; }
        qsort(lv, cnt, sizeof(int64_t), cmp_rows);
        LEVEL[d] = lv;
        LCOUNT[d] = (int64_t)cnt;
        NLEVELS = d + 1;
        if (cnt > STOP) break;
    }
    free(row);
    for (int d = 0; d < NLEVELS; d++) counts[d] = LCOUNT[d];
    return NLEVELS;
}

// copy level d into out (count x n, row-major)
void am_eyes_copy(int d, i64 *out) {
    for (int64_t p = 0; p < LCOUNT[d]; p++)
        memcpy(out + (size_t)p * N, ARENA + (size_t)LEVEL[d][p] * N, sizeof(i64) * N);
}
