"""AM AS ANCHOR: for every number of every object (X, Y, Z, yaw), AM's hives look for the rule the measured frames
follow within the eyes' noise band, letting frames break it (at most 40% of them: the rule must be the majority's).
AM's law reads what the hives report and names the hive whose rule anchors the number, or CANNOT.  The measured frames
outside the anchoring rule's band are AM's flags; the rule's values at every frame are its anchors.
  THE BAND: tau = 3 sigma + 0.5 (the eyes' noise, and the rounding).
  AM'S EYES: every distinct behaviour of <= 3 steps over AM's shelf (scene_am.levels, unchanged) over frames 0-47.
  THE HIVES:
    1 EXACT   behaviours within the band of the most frames; the shallowest depth that reaches that many
    2 SHAPE   y = a s + b with whole a (not 0) and b, a from least squares on the frames near a first fit, b the median
              (so broken frames do not pull it); a series that never leaves the band of its median is depth 0
    3 LOCAL   a straight line through the measured frames within 5 of each frame, refitted without those far from it
    4 REPEAT  every period 2-24: each frame anchored at the median of the frames a whole number of periods away; the
              period with the fewest frames outside the band
    Each reports: its anchors at every frame (None where it cannot), its outliers, unique (every best fit agrees within
    the band at every frame).
  THE STATE (18 bits): per hive (order above) 4 bits: answered, unique, outliers 0 / 1-2 / 3-8 / 9+; bit 16 every
  answering hive's anchors agree within the band; bit 17 SHORT (fewer than 24 measured frames).
  THE FRAME LAW (the AM over the four numbers of one object at one frame): for every measured object-frame that some
  number flags, and every frame an object is hidden or popped, AM's frame state (17 bits): bit 0 measured; per number
  (X, Y, Z, A) 4 bits: answered (its law named a hive), flagged here, far (more than 3 bands off), global (EXACT or
  SHAPE).  Its law (lanes_frame.json, machine-authored by integer program synthesis from practice) says: 0 leave the frame as it is, 1 flag
  it and stay silent, 2 flag it and anchor it (each number at its rule, or at its measurement where its law said
  CANNOT; on a frame not measured every number needs a rule).
    anchor(v) -> flags {(k, t)}, anchors {(k, t): {X, Y, Z, A}}, info"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/neooos/scene_dna"))
import scene_am as S                                            # noqa: E402
import ai_world as V                                            # noqa: E402

N = V.N
LAW = os.path.join(HERE, "lanes_anchor.json")
FRAME_LAW = os.path.join(HERE, "lanes_frame.json")
NBITS = 18
FRAME_NBITS = 17
HIVES = ("exact", "shape", "local", "repeat")
CHUNK = 60000
_EYES = None


def eyes():
    global _EYES
    if _EYES is None:
        _EYES = [(d, Y) for d, Y in S.levels(list(range(N)))]
    return _EYES


def band(key, ys):
    return 3 * V.sigma(key, float(np.median(ys))) + 0.5


def _report(cands, ts, y, tau):
    """cands: list of anchor vectors (len N) at the best inlier count and shallowest depth"""
    A = np.array(cands, dtype=float)
    uniq = int(float((A.max(0) - A.min(0)).max()) <= tau)
    anc = np.rint(np.median(A, 0)).astype(np.int64)
    out = int((np.abs(y - anc[ts]) > tau).sum())
    return dict(anc=[int(v) for v in anc], uniq=uniq, out=out)


def _search(chunk_fit):
    """the most frames within the band over all of AM's eyes, and the behaviours that reach it at the shallowest
    depth that does"""
    best, cands = -1, []
    for d, Y in eyes():
        dmax, dc = -1, []
        for c in range(0, len(Y), CHUNK):
            inl, make = chunk_fit(Y[c:c + CHUNK])
            mx = int(inl.max()) if len(inl) else -1
            if mx < 0 or mx < dmax:
                continue
            rows = list(make(np.nonzero(inl == mx)[0][:2000]))
            if mx > dmax:
                dmax, dc = mx, rows
            else:
                dc += rows[:2000 - len(dc)]
        if dmax > best:
            best, cands = dmax, dc
    return best, cands


def exact(ts, y, tau, K):
    best, cands = _search(lambda B: ((np.abs(B[:, ts] - y) <= tau).sum(1), lambda sel: B[sel]))
    return _report(cands, ts, y, tau) if best >= len(ts) - K else None


def shape(ts, y, tau, K):
    m = len(ts)
    yf = y.astype(float)

    def fit(B):
        Sx = B[:, ts].astype(float)
        sm = Sx.mean(1)
        ds = Sx - sm[:, None]
        var = (ds * ds).sum(1)
        ok = var > 0
        a = np.where(ok, (ds * (yf - yf.mean())).sum(1) / np.where(ok, var, 1.0), 0.0)
        b = np.median(yf - a[:, None] * Sx, axis=1)
        w = np.abs(yf - (a[:, None] * Sx + b[:, None])) <= 3 * tau
        nw = np.maximum(w.sum(1), 1)
        sw = (Sx * w).sum(1) / nw
        yw = (yf * w).sum(1) / nw
        dsw = (Sx - sw[:, None]) * w
        varw = (dsw * (Sx - sw[:, None])).sum(1)
        a2 = np.where(varw > 0, (dsw * (yf - yw[:, None])).sum(1) / np.where(varw > 0, varw, 1.0), 0.0)
        ai = np.rint(a2)
        ok &= ai != 0
        bi = np.rint(np.median(yf - ai[:, None] * Sx, axis=1))
        inl = (np.abs(yf - (ai[:, None] * Sx + bi[:, None])) <= tau).sum(1)
        inl[~ok] = -1
        return inl, lambda sel: B[sel] * ai[sel, None] + bi[sel, None]

    best, cands = _search(fit)
    med = int(round(float(np.median(yf))))
    const = int((np.abs(yf - med) <= tau).sum())
    if const >= best:                                          # depth 0, the shallowest: a number that never moves
        best, cands = const, [[med] * N]
    return _report(cands, ts, y, tau) if best >= m - K else None


def local(ts, y, tau, K, W=5):
    tsa, yf = np.array(ts), y.astype(float)
    anc = []
    for t in range(N):
        sel = np.abs(tsa - t) <= W
        if sel.sum() < 4:
            anc.append(None)
            continue
        x, z = tsa[sel].astype(float), yf[sel]
        p = np.polyfit(x, z, 1)
        keep = np.abs(z - np.polyval(p, x)) <= 2 * tau
        if keep.sum() >= 3 and np.ptp(x[keep]) > 0:
            p = np.polyfit(x[keep], z[keep], 1)
        anc.append(int(round(float(np.polyval(p, t)))))
    defined = [i for i, t in enumerate(ts) if anc[t] is not None]
    out = sum(abs(int(y[i]) - anc[ts[i]]) > tau for i in defined)
    if out > K:
        return None
    return dict(anc=anc, uniq=1, out=int(out))


def repeat(ts, y, tau, K):
    val = dict(zip(ts, (int(v) for v in y)))
    best = None
    for p in range(2, 25):
        anc = []
        for t in range(N):
            vs = [val[u] for u in range(t % p, N, p) if u in val]
            anc.append(int(round(float(np.median(vs)))) if vs else None)
        out = sum(abs(val[t] - anc[t]) > tau for t in ts)
        if any(sum(1 for u in range(r, N, p) if u in val) >= 2 for r in range(p)) and (best is None or out < best[0]):
            best = (out, p, anc, [])
        elif best is not None and out == best[0]:
            best[3].append(anc)
    if best is None or best[0] > K:
        return None
    out, p, anc, ties = best
    uniq = int(all(all(a is None or b is None or abs(a - b) <= tau for a, b in zip(anc, other)) for other in ties))
    return dict(anc=anc, uniq=uniq, out=int(out))


VIEWS = (exact, shape, local, repeat)


def reports(ts, y, tau):
    K = int(0.4 * len(ts))
    return [f(ts, y, tau, K) if len(ts) >= S.MIN_PAIRS else None for f in VIEWS]


def state(reps, tau, m):
    n = 0
    for i, r in enumerate(reps):
        if r is not None:
            oc = 0 if r["out"] == 0 else (1 if r["out"] <= 2 else (2 if r["out"] <= 8 else 3))
            n |= (1 | r["uniq"] << 1 | oc << 2) << (4 * i)
    got = [r["anc"] for r in reps if r is not None]
    if got and all(all(a is None or b is None or abs(a - b) <= tau for a, b in zip(got[0], g)) for g in got[1:]):
        n |= 1 << 16
    if m < 24:
        n |= 1 << 17
    return n


_LAWS = {}


def law(n, path=LAW):
    if path not in _LAWS:
        L = collections.defaultdict(list)
        for name, info in json.load(open(path)).items():
            if info.get("expr"):
                L[int(info["act"])].append(info["expr"])
        _LAWS[path] = L
    for j in (1, 2, 3, 4):
        if any(S._E(e, n) & 1 for e in _LAWS[path].get(j, ())):
            return j
    return 0


def series(v, k, key):
    ts = sorted(t for (kk, t) in v["eyes"] if kk == k)
    return ts, np.array([v["eyes"][(k, t)][key] for t in ts], dtype=np.int64)


def numbers(v, act_of=None):
    """every number's report: its act (the law's hive, 0 CANNOT), anchors, band"""
    act_of = act_of or law
    per = {}
    for k in v["objs"]:
        for key in V.KEYS:
            ts, y = series(v, k, key)
            if len(ts) < S.MIN_PAIRS:
                per[(k, key)] = dict(act=0, anc=None, tau=None, n=None)
                continue
            tau = band(key, y)
            reps = reports(ts, y, tau)
            n = state(reps, tau, len(ts))
            act = act_of(n)
            per[(k, key)] = dict(act=act if act and reps[act - 1] is not None else 0,
                                 anc=reps[act - 1]["anc"] if act and reps[act - 1] is not None else None, tau=tau, n=n)
    return per


def frames(v, per):
    """the object-frames the frame law reads: measured ones some number flags, and every hidden or popped one; for
    each its frame state and the anchor it would give (None when it cannot form one)"""
    out = {}
    for k in v["objs"]:
        for t in range(N):
            e = v["eyes"].get((k, t))
            n, flagged, anc = int(e is not None), False, {}
            for i, key in enumerate(V.KEYS):
                r = per[(k, key)]
                a = r["anc"][t] if r["anc"] is not None else None
                bits = 0
                if a is not None:
                    bits |= 1
                    if r["act"] in (1, 2):
                        bits |= 8
                    if e is not None and abs(e[key] - a) > r["tau"]:
                        bits |= 2
                        flagged = True
                        if abs(e[key] - a) > 3 * r["tau"]:
                            bits |= 4
                    anc[key] = a
                elif e is not None:
                    anc[key] = e[key]
                n |= bits << (1 + 4 * i)
            if e is None or flagged:
                out[(k, t)] = dict(n=n, anchor=anc if len(anc) == 4 and anc["Z"] > 0 else None)
    return out


_FLAWS = {}


def frame_law(n, path=FRAME_LAW):
    if path not in _FLAWS:
        L = collections.defaultdict(list)
        for name, info in json.load(open(path)).items():
            if info.get("expr"):
                L[int(info["act"])].append(info["expr"])
        _FLAWS[path] = L
    for j in (2, 1):
        if any(S._E(e, n) & 1 for e in _FLAWS[path].get(j, ())):
            return j
    return 0


def anchor(v, act_of=None, frame_act=None):
    """AM's flags and anchors for a whole video: the number law on every number, the frame law on every object-frame"""
    frame_act = frame_act or frame_law
    per = numbers(v, act_of)
    flags, anchors = set(), {}
    for kt, f in frames(v, per).items():
        act = frame_act(f["n"])
        if act and kt in v["eyes"]:
            flags.add(kt)
        if act == 2 and f["anchor"] is not None:
            anchors[kt] = f["anchor"]
    return flags, anchors, per
