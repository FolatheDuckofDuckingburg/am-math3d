"""SCENE AM: AM's own machinery on a scene's numbers, with the material scenes need.
EYES: AM's eyes, unchanged in method (every distinct behaviour of <= 3 steps over the inputs, deduplicated on them;
those equal to the data kept), over AM's shelf plus two steps animation needs: -x (a minus sign: things move left
and get darker) and 0 (nothing changes).  This is material, not a decision.
A SOURCE is one bee's view of a property (own numbers, relative to the first, or minus the shared memory's camera /
exposure numbers, and that relative); it reports, as AM's sources do, the shallowest depth that fits, whether every
fit at that depth predicts the same, and the most common prediction there (ties to the smallest).
THE STATE is AM's own 14-bit state: bit 0 kind (0, rule); 3 bits per source (depth code 1-3, unique << 2), sources in
the order own, aff, comb, comb_aff; bit 13 when every source that fits predicts the same.
THE LAW (the AM on top) reads the state and names the source to answer with, or 0 = CANNOT.  It is machine-authored
by integer program synthesis from practice scenes (lanes_scene.json); it is never hand-written."""
import collections, json, os, sys
from collections import OrderedDict, Counter
import numpy as np

sys.path.insert(0, os.path.expanduser("~/neooos/am/python"))
from am import look                                     # noqa: E402  (AM's eyes: the shelf, the depth, the limit)

HERE = os.path.dirname(os.path.abspath(__file__))
SHELF = look.STEPS + [("-x", lambda x: -x), ("0", lambda x: x * 0)]
SOURCES = ("own", "aff", "comb", "comb_aff")
MIN_PAIRS = 5                                           # AM's eyes look at a lane only with >= 5 examples
_LEVELS = OrderedDict()
CACHE_N = int(os.environ.get("SCENE_AM_CACHE", "4"))


def levels(xs):
    key = np.array(xs, dtype=np.int64).tobytes()
    hit = _LEVELS.get(key)
    if hit is not None:
        _LEVELS.move_to_end(key)
        return hit
    xs = np.array(xs, dtype=np.int64)
    level = xs[None, :]
    out = [(0, level)]
    seen = {level[0].tobytes()}
    for d in range(1, look.DEPTH + 1):
        nxt = []
        for _, f in SHELF:
            with np.errstate(all="ignore"):
                y = f(level)
            ok = np.all(np.abs(y) < look.LIM, axis=1)
            nxt.append(y[ok])
        Y = np.unique(np.concatenate(nxt), axis=0)
        keep = np.array([r.tobytes() not in seen for r in Y], bool)
        Y = Y[keep]
        for r in Y:
            seen.add(r.tobytes())
        if not len(Y):
            break
        out.append((d, Y))
        level = Y
        if len(level) > 400000:
            break
    _LEVELS[key] = out
    while len(_LEVELS) > CACHE_N:
        _LEVELS.popitem(last=False)
    return out


def source(pairs, queries):
    """one bee: None when it sees nothing, else dict(d, uniq, top) with top the prediction at the queries"""
    if len(pairs) < MIN_PAIRS:
        return None
    pairs = sorted(pairs)
    xs = [t for t, _ in pairs] + list(queries)
    want = np.array([y for _, y in pairs], dtype=np.int64)
    m = len(want)
    found = []
    for d, Y in levels(xs):
        hit = np.all(Y[:, :m] == want, axis=1)
        if hit.any():
            found += [(d, tuple(int(v) for v in r[m:])) for r in Y[hit]]
            break                                            # the shallowest depth is all the state reads
    if not found:
        return None
    d = found[0][0]
    votes = Counter(p for _, p in found)
    top = max(votes.items(), key=lambda kv: (kv[1], [-v for v in kv[0]]))[0]
    return dict(d=d, uniq=int(len(votes) == 1), top=list(top))


def source_aff(pairs, queries):
    """one bee that cancels scale and offset: the shallowest shape s (a behaviour of <= 3 steps) with
    y = a*s + b on every example for whole numbers a, b (solved from the data, so the numbers cancel); reported like
    any source (depth of the shape, unique, most common prediction).  A series that never changes is depth 0."""
    if len(pairs) < MIN_PAIRS:
        return None
    pairs = sorted(pairs)
    xs = [t for t, _ in pairs] + list(queries)
    y = np.array([v for _, v in pairs], dtype=np.int64)
    m = len(y)
    dy = y - y[0]
    nz = np.nonzero(dy)[0]
    if not len(nz):
        return dict(d=0, uniq=1, top=[int(y[0])] * len(queries))
    j = int(nz[0])
    for d, Y in levels(xs):
        ds = Y[:, :m] - Y[:, :1]
        den = ds[:, j]
        ok = den != 0
        a = np.zeros(len(Y), np.int64)
        a[ok] = dy[j] // den[ok]
        ok &= a * den == dy[j]
        ok &= np.all(ds * a[:, None] == dy[None, :], axis=1)
        if ok.any():
            A = a[ok]
            B = y[0] - A * Y[ok, 0]
            P = Y[ok, m:] * A[:, None] + B[:, None]
            votes = Counter(tuple(int(v) for v in r) for r in P)
            top = max(votes.items(), key=lambda kv: (kv[1], [-v for v in kv[0]]))[0]
            return dict(d=d, uniq=int(len(votes) == 1), top=list(top))
    return None


def state(srcs, preds):
    """srcs: the four sources (None or a report); preds: each source's prediction in the property's own units"""
    n = 0
    for i, s in enumerate(srcs):
        if s is not None:
            n |= ((1 if s["d"] <= 1 else min(s["d"], 3)) | s["uniq"] << 2) << (1 + 3 * i)
    found = [p for p in preds if p is not None]
    if found and all(p == found[0] for p in found):
        n |= 1 << 13
    return n


# ---------------------------------------------- the law (machine-authored by integer program synthesis)
import ast, operator
_W = lambda v: ((v & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.RShift: operator.rshift,
        ast.LShift: operator.lshift}
_TREES = {}


def _ev(n, x):
    if isinstance(n, ast.Name):
        return x
    if isinstance(n, ast.Constant):
        return _W(n.value)
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, x)
        return _W(-v if isinstance(n.op, ast.USub) else ~v if isinstance(n.op, ast.Invert) else v)
    return _W(_OPS[type(n.op)](_ev(n.left, x), _ev(n.right, x)))


def _E(expr, x):
    """an authored lane, evaluated in int32 at every step, as the engine evaluates it (the LVC core's E)"""
    if expr not in _TREES:
        _TREES[expr] = ast.parse(expr, mode="eval").body
    return _ev(_TREES[expr], _W(x))


_LAW = None


def law(n):
    """the act: the lowest-numbered source whose lane fires (1-4), else 0 = CANNOT"""
    global _LAW
    if _LAW is None:
        lanes = json.load(open(os.path.join(HERE, "lanes_scene.json")))
        _LAW = collections.defaultdict(list)
        for name, info in lanes.items():
            if info.get("expr"):
                _LAW[int(info["act"])].append(info["expr"])
    for j in (1, 2, 3, 4):
        if any(_E(e, n) & 1 for e in _LAW.get(j, ())):
            return j
    return 0
