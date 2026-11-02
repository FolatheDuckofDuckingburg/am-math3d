"""FIVE HIVES, one memory, one AM on top.  Every number (one object's x, y or value over the frames) is seen by five
hives at once, each a different way of seeing motion, each using AM's eyes (scene_am: every distinct behaviour of
<= 3 steps over AM's shelf plus -x and 0) on its own view:
  1 EXACT     a rule over t that gives the numbers exactly (scene_am.source)
  2 SHAPE     a shape over t, numbers = a*shape + b, a and b solved from the data (scene_am.source_aff)
  3 LATEST    SHAPE on the last 6 measured frames only: the motion's latest regime (a clamp, a new step)
  4 VELOCITY  SHAPE on the frame-to-frame changes, summed back up from the last measured value
  5 REPEAT    every period p = 2-8 with y(t) = y(t - p) on every measured pair (at least 4 pairs); the future
              repeats the past.  Its depth is 1; unique when every period that fits predicts the same.
All five read the shared memory: where the camera / exposure is settled they see the number with it cancelled.
THE STATE: AM's own layout widened to five sources: bit 0 kind (0); 3 bits per hive (depth code 1-3, unique << 2);
bit 16 when every hive that answers predicts the same.
THE LAW (lanes_scene5.json, machine-authored by integer program synthesis from practice scenes) names the hive to answer with, or CANNOT."""
import collections, json, os
import numpy as np
import scene_am as S

HERE = os.path.dirname(os.path.abspath(__file__))
HIVES = ("exact", "shape", "latest", "velocity", "repeat")


def latest(pairs, queries):
    pairs = sorted(pairs)
    return S.source_aff(pairs[-6:], queries) if len(pairs) >= 6 else None


def velocity(pairs, queries):
    pairs = sorted(pairs)
    d = [(t, y - y0) for (t0, y0), (t, y) in zip(pairs, pairs[1:]) if t == t0 + 1]
    last_t, last_y = pairs[-1]
    need = list(range(last_t + 1, max(queries) + 1))
    if not need:
        return None
    s = S.source_aff(d, need)
    if s is None:
        return None
    v, out = last_y, {}
    for t, dv in zip(need, s["top"]):
        v += dv
        out[t] = v
    return dict(d=s["d"], uniq=s["uniq"], top=[out[q] for q in queries])


def repeat(pairs, queries):
    y = dict(pairs)
    preds = []
    for p in range(2, 9):
        conf = [(t, t - p) for t in y if t - p in y]
        if len(conf) < 4 or any(y[a] != y[b] for a, b in conf):
            continue
        pr = []
        for q in queries:
            k = next((q - m * p for m in range(1, 40) if q - m * p in y), None)
            if k is None:
                break
            pr.append(y[k])
        if len(pr) == len(queries):
            preds.append(pr)
    if not preds:
        return None
    return dict(d=1, uniq=int(all(p == preds[0] for p in preds)), top=list(preds[0]))


VIEWS = (S.source, S.source_aff, latest, velocity, repeat)


def reports(pairs, queries):
    out = []
    for f in VIEWS:
        r = f(pairs, queries) if len(pairs) >= S.MIN_PAIRS else None
        out.append(r)
    return out


def state(srcs, preds):
    n = 0
    for i, s in enumerate(srcs):
        if s is not None:
            n |= ((1 if s["d"] <= 1 else min(s["d"], 3)) | s["uniq"] << 2) << (1 + 3 * i)
    found = [p for p in preds if p is not None]
    if found and all(p == found[0] for p in found):
        n |= 1 << 16
    return n


_LAW = None


def law(n):
    global _LAW
    if _LAW is None:
        lanes = json.load(open(os.path.join(HERE, "lanes_scene5.json")))
        _LAW = collections.defaultdict(list)
        for name, info in lanes.items():
            if info.get("expr"):
                _LAW[int(info["act"])].append(info["expr"])
    for j in (1, 2, 3, 4, 5):
        if any(S._E(e, n) & 1 for e in _LAW.get(j, ())):
            return j
    return 0
