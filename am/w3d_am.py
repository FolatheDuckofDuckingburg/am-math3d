"""AM FOR THE 3D WORLD: AM's five hives on one number's measured frames, AM's state with the question's place in it, and
the law machine-authored by integer program synthesis for this world (lanes_w3d.json).
  THE QUESTIONS: every stretch of frames an object is hidden (from every frame it is seen), and frames 36-47 from
           frames 0-35; one question per number (2D eyes: screen x, y; 3D eyes: X, Y, Z).
  WHERE: AHEAD when every asked frame lies after everything seen, BACK when every one lies before, else BETWEEN.
  THE HIVES (scene_am and scene_am5 read-only, as sealed; the changes are here and only for this world):
    1 EXACT     scene_am.source, unchanged
    2 SHAPE     scene_am.source_aff, unchanged
    3 LATEST    SHAPE on the 6 measured frames nearest the asked ones (AHEAD: the last 6, as in scene_am5; BACK: the
                first 6; BETWEEN: the nearest on either side)
    4 VELOCITY  scene_am5.velocity: SHAPE on the frame-to-frame changes, summed from the measured frame nearest the
                asked ones; AHEAD forward in time, BACK backward (time reversed); BETWEEN is not its to answer
    5 REPEAT    every period p = 2-24 (scene_am5: 2-8) with y(t) = y(t - p) on every measured pair (at least 4); an
                asked frame repeats the nearest measured frame a whole number of periods away, on either side (scene_am5:
                only earlier).  Depth 1; unique when every period that fits predicts the same.
  THE STATE: scene_am5.state (bits 1-15 the hives, bit 16 every answering hive agrees), with bit 0 set when the
           question is not AHEAD, bit 17 set when it is BETWEEN, and bit 18 (FAR) set when the farthest asked frame lies
           more frames from the nearest measured frame than there are measured frames.
    answers(series, queries) -> {"w3d": {t: value} or None, "scene5": ...}   (None = CANNOT)"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/neooos/scene_dna"))
import scene_am as S                                            # noqa: E402
import scene_am5 as S5                                          # noqa: E402

LAW = os.path.join(HERE, "lanes_w3d.json")
NBITS = 19


def where(pairs, queries):
    ts = [t for t, _ in pairs]
    return "ahead" if min(queries) > max(ts) else ("back" if max(queries) < min(ts) else "between")


def latest(pairs, queries):
    pairs = sorted(pairs)
    if len(pairs) < 6:
        return None
    w = where(pairs, queries)
    if w == "ahead":
        near = pairs[-6:]
    elif w == "back":
        near = pairs[:6]
    else:
        near = sorted(sorted(pairs, key=lambda p: (min(abs(p[0] - q) for q in queries), p[0]))[:6])
    return S.source_aff(near, queries)


def velocity(pairs, queries):
    pairs = sorted(pairs)
    w = where(pairs, queries)
    if w == "ahead":
        return S5.velocity(pairs, queries)
    if w == "between":
        return None
    r = max(t for t, _ in pairs)                       # BACK: the video played backward, s = r - t
    return S5.velocity(sorted((r - t, y) for t, y in pairs), [r - q for q in queries])


def repeat(pairs, queries):
    y = dict(pairs)
    preds = []
    for p in range(2, 25):
        conf = [(t, t - p) for t in y if t - p in y]
        if len(conf) < 4 or any(y[a] != y[b] for a, b in conf):
            continue
        pr = []
        for q in queries:
            k = next((q + s * m * p for m in range(1, 40) for s in (-1, 1) if q + s * m * p in y), None)
            if k is None:
                break
            pr.append(y[k])
        if len(pr) == len(queries):
            preds.append(pr)
    if not preds:
        return None
    return dict(d=1, uniq=int(all(p == preds[0] for p in preds)), top=list(preds[0]))


VIEWS = (S.source, S.source_aff, latest, velocity, repeat)


def runs(ts):
    """consecutive frames grouped: [3, 4, 5, 9] -> [[3, 4, 5], [9]]"""
    out = []
    for t in sorted(ts):
        if out and t == out[-1][-1] + 1:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def questions(fr, k, n):
    """job 2: each hidden stretch, from every frame the object is seen; job 3: frames 36-47 seen, from frames 0-35"""
    vis = [t for t in range(n) if fr[t][k] is not None]
    hid = [t for t in range(n) if fr[t][k] is None]
    return dict(occlusion=[(vis, r) for r in runs(hid)],
                future=[([t for t in vis if t < 36], [t for t in vis if t >= 36])] if any(t >= 36 for t in vis) else [])


def reports(pairs, queries):
    return [f(pairs, queries) if len(pairs) >= S.MIN_PAIRS else None for f in VIEWS]


def far(pairs, queries):
    ts = [t for t, _ in pairs]
    return max(min(abs(q - t) for t in ts) for q in queries) > len(ts)


def state(pairs, queries, srcs, preds):
    w = where(pairs, queries)
    return S5.state(srcs, preds) | int(w != "ahead") | int(w == "between") << 17 | int(far(pairs, queries)) << 18


_LAWS = {}


def law(n, path=LAW):
    if path not in _LAWS:
        L = collections.defaultdict(list)
        for name, info in json.load(open(path)).items():
            if info.get("expr"):
                L[int(info["act"])].append(info["expr"])
        _LAWS[path] = L
    for j in (1, 2, 3, 4, 5):
        if any(S._E(e, n) & 1 for e in _LAWS[path].get(j, ())):
            return j
    return 0


def answers(series, queries, laws=("w3d", "scene5")):
    """AM's answer at the asked frames under each law, or None (CANNOT).  The scene law, unchanged, reads the sealed
    five hives (scene_am5.reports, its velocity silent where a question is not AHEAD) and their own state."""
    pairs = sorted(series.items())
    if len(pairs) < S.MIN_PAIRS:
        return {which: None for which in laws}
    w = where(pairs, queries)
    ex, sh = S.source(pairs, queries), S.source_aff(pairs, queries)          # the same two hives in both
    srcs = [ex, sh, latest(pairs, queries), velocity(pairs, queries), repeat(pairs, queries)]
    srcs5 = [ex, sh, srcs[2] if w == "ahead" else S5.latest(pairs, queries), srcs[3] if w == "ahead" else None,
             S5.repeat(pairs, queries)]
    out = {}
    for which in laws:
        sv = srcs if which == "w3d" else srcs5
        preds = [None if s is None else s["top"] for s in sv]
        act = law(state(pairs, queries, sv, preds)) if which == "w3d" else S5.law(S5.state(sv, preds))
        out[which] = dict(zip(queries, preds[act - 1])) if act and preds[act - 1] is not None else None
    return out
