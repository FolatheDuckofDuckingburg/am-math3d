"""THE BRAIN'S EYES -- a plain enumerator over the basic-Python shelf.  It decides nothing: it lists every rule up to
DEPTH steps that reproduces the data, per view and per upper lane, with what each predicts.  The brain's decisions
(which candidate, which view, whole or lane, answer or CANNOT) are the synthesised laws, not this file's.
Bottom-up over chains of unary steps; programs that behave the same on the probe inputs are kept once (shortest
first), so depth d holds every distinct behaviour of d steps.
    candidates(data) -> list of dict(view, lane, depth, pred)"""
import numpy as np

DEPTH = 3
LIM = 1 << 40


def steps():
    S = []
    for c in range(1, 13):
        S += [(f"x+{c}", lambda x, c=c: x + c), (f"x-{c}", lambda x, c=c: x - c), (f"x|{c}", lambda x, c=c: x | c),
              (f"x^{c}", lambda x, c=c: x ^ c), (f"max(x,{c})", lambda x, c=c: np.maximum(x, c)),
              (f"min(x,{3 * c})", lambda x, c=c: np.minimum(x, 3 * c)), (f"x&{5 * c}", lambda x, c=c: x & (5 * c))]
        if c >= 2:
            S += [(f"{c}*x", lambda x, c=c: c * x), (f"x//{c}", lambda x, c=c: x // c), (f"x%{c}", lambda x, c=c: x % c)]
    for k in range(1, 5):
        S += [(f"x<<{k}", lambda x, k=k: x << k), (f"x>>{k}", lambda x, k=k: x >> k)]
    S += [("abs(x)", np.abs), ("x*x", lambda x: x * x), ("x*x*x", lambda x: x * x * x)]
    return S


STEPS = steps()


def chains(xs, want, depth=DEPTH):
    """every distinct behaviour of <= depth steps on xs; those equal to want on the first len(want) inputs"""
    xs = np.array(xs, dtype=np.int64)
    m = len(want)
    want = np.array(want, dtype=np.int64)
    found = []
    level = xs[None, :]                                      # behaviours of the current depth
    seen = {level[0].tobytes()}
    if np.array_equal(level[0, :m], want):
        found.append((0, tuple(int(v) for v in level[0, m:])))
    for d in range(1, depth + 1):
        nxt = []
        for _, f in STEPS:
            with np.errstate(all="ignore"):
                y = f(level)
            ok = np.all(np.abs(y) < LIM, axis=1)
            nxt.append(y[ok])
        Y = np.concatenate(nxt) if nxt else np.zeros((0, len(xs)), np.int64)
        if not len(Y):
            break
        Y = np.unique(Y, axis=0)
        keep = np.array([r.tobytes() not in seen for r in Y], bool)
        Y = Y[keep]
        for r in Y:
            seen.add(r.tobytes())
        hit = np.all(Y[:, :m] == want, axis=1)
        for r in Y[hit]:
            found.append((d, tuple(int(v) for v in r[m:])))
        level = Y
        if len(level) > 400000:                              # the eyes' own limit: deeper levels are not looked at
            break
    return found


def candidates(data):
    out = []
    if data["kind"] == "rule":
        P = [tuple(p) for p in data["pairs"]]
        q = list(data["queries"])
        S = sorted(P)
        for k in range(0, len(S) - 4):                       # lane law: every upper lane with >= 5 examples (k = 0: whole)
            lane = S[k:]
            for d, pred in chains([x for x, _ in lane] + q, [y for _, y in lane]):
                out.append(dict(view="x", lane=k, depth=d, pred=pred))
        return out
    a = list(data["terms"])
    n = len(a)
    for d, pred in chains(list(range(1, n + 1)) + [n + 1, n + 2], a):
        out.append(dict(view="n", lane=0, depth=d, pred=pred))
    # successor views: one step function f, iterated twice for the answer
    for view, X in (("succ", [a[i - 1] for i in range(1, n)]),):
        for d, _ in chains(X, a[1:], depth=2):
            pass
        res = _iterate_succ(a)
        out += res
    return out


def _iterate_succ(a):
    """successor a(n) = f(a(n-1)) with f a chain of <= 2 steps, and alternating f_odd / f_even of 1 step each;
    two-term a(n) = p a(n-1) + q a(n-2) + r with small integers; each iterated to the next two terms"""
    out = []
    n = len(a)
    X = np.array(a[:-1], np.int64)
    Y = np.array(a[1:], np.int64)
    # one chain f for every step
    for name1, f1 in [("id", lambda x: x)] + STEPS:
        with np.errstate(all="ignore"):
            z = f1(X)
        for name2, f2 in [("id", lambda x: x)] + STEPS:
            with np.errstate(all="ignore"):
                if np.array_equal(f2(z), Y):
                    v = a[-1]
                    p = []
                    for _ in range(2):
                        v = int(f2(f1(np.array([v], np.int64)))[0]); p.append(v)
                    out.append(dict(view="succ", lane=0, depth=(name1 != "id") + (name2 != "id"), pred=tuple(p)))
    # alternating: f for even positions, g for odd
    for parity in (0, 1):
        idx_f = [i for i in range(1, n) if i % 2 == parity]
        idx_g = [i for i in range(1, n) if i % 2 != parity]
        F = [nm for nm, f in STEPS if all(int(f(np.array([a[i - 1]], np.int64))[0]) == a[i] for i in idx_f)]
        G = [nm for nm, g in STEPS if all(int(g(np.array([a[i - 1]], np.int64))[0]) == a[i] for i in idx_g)]
        D = dict(STEPS)
        for fn in F:
            for gn in G:
                v, p = a[-1], []
                for i in (n, n + 1):
                    h = D[fn] if i % 2 == parity else D[gn]
                    v = int(h(np.array([v], np.int64))[0]); p.append(v)
                out.append(dict(view="alt", lane=0, depth=2, pred=tuple(p)))
    # two-term linear: p, q in -3..3, r in -12..12
    for p in range(-3, 4):
        for qq in range(-3, 4):
            r = a[2] - p * a[1] - qq * a[0]
            if abs(r) <= 12 and all(a[i] == p * a[i - 1] + qq * a[i - 2] + r for i in range(2, n)):
                x1 = p * a[-1] + qq * a[-2] + r
                x2 = p * x1 + qq * a[-1] + r
                out.append(dict(view="rec2", lane=0, depth=(p != 0) + (qq != 0) + (r != 0), pred=(x1, x2)))
    return out
