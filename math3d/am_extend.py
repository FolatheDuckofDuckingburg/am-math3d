"""AM EXTENDS THE VIDEO: AM reasons over the clip's own numbers, its rules generalise to frames it never saw, and the
rules carried past the last frame are the math for the extension.
  REASON: am3d.spine, unchanged: for each own number (rX, rY, rZ, W, H, oB of every object; the camera track mX, mY, mZ)
    AM's anchor hives propose rules within the band and AM's number law names one, or CANNOT; the frames that keep
    every rule are the object's clean frames.
  GENERALISE: the new frames are asked of AM's five 3D hives as an AHEAD question under AM's 3D law (w3d_am.answers),
    the law built for frames past the record, from the clean frames: an answer, or CANNOT.  (The spine's rules carried
    forward were audited on hidden frames, 1308 right / 686 wrong, and are not used for the future.)
  PROOF ON THIS VIDEO: the same reasoning with the clip's last H frames hidden from AM; its answers there are checked
    against what the eyes measured, each within the number's own band.  Holding the last seen value is scored beside it.
  THE MATH FOR THE EXTENSION: own numbers + the camera track -> X, Y, Z, W, H (cm) -> the stated lens -> the box on
    screen (x = cx + f X/Z, y = cy + f Y/Z, w = f W/Z, h = f H/Z), frame by frame.  AM never fills a number it answered
    CANNOT; the map says so.  Two directing choices are the user's and are marked wherever they are used: the camera
    is held where it was at the last frame where AM has no rule for it, and ("held" boxes) an own number AM cannot
    extend is held at its last measured value.
    python3 am_extend.py MAPDIR [seconds]     -> MAPDIR/extend.json"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am3d                                                    # noqa: E402

C, M = am3d.C, am3d.M
IDX = dict(rX=11, rY=12, rZ=13, W=8, H=9, oB=14)              # own numbers in map.json's rows
PLACE = ("rX", "rY", "rZ", "W", "H")                           # what a box on screen needs
GAP = 2                                                        # an object seen this close to the last frame goes on


def own_series(m):
    return {o["id"]: {r[0]: {k: int(r[i]) for k, i in IDX.items()} for r in o["rows"]} for o in m["objects"]}


def camera_series(m):
    return {int(t): {"m" + a: int(v[a]) for a in "XYZ"} for t, v in m["marker"].items()}


def ask(series, frames):
    """AM over one object's numbers ({t: {key: v}}): the spine marks the frames that keep every rule (the map's proof),
    and AM's 3D law, the law built for frames past the record, answers the asked frames from those frames, or CANNOT.
    (Carrying the spine's own rules forward was audited on hidden frames and is not used: 1308 right, 686 wrong.)"""
    ts = sorted(series)
    keys = list(series[ts[0]])
    out, broken = {}, set()
    for k in keys:
        rule, _ = am3d.spine(ts, np.array([series[t][k] for t in ts], dtype=np.int64))
        out[k] = dict(rule=rule)
        broken |= set(rule.get("broken_frames", []))
    clean = [t for t in ts if t not in broken]
    for k in keys:
        ans = M.answers({t: series[t][k] for t in clean}, frames, laws=("w3d",))["w3d"] if len(clean) >= 5 else None
        out[k].update(got={t: (int(ans[t]) if ans else None) for t in frames},
                      src={t: ("3D law" if ans else None) for t in frames})
    return out


def _rule_view(rule):
    v = dict(rule=rule["rule"], band=rule["band"])
    if rule["rule"] != "CANNOT":
        v.update(formula=rule.get("formula", ""), kept=rule["kept"], measured=rule["measured"])
    return v


def proof(m, H):
    """AM with the clip's last H frames hidden from it: its answers there against what the eyes measured"""
    n = m["frames"]
    cut = n - H
    C._length(n)
    total = dict(asked=0, answered=0, right=0, wrong=0, cannot=0, hold_right=0)

    def score(series, hidden):
        seen = {t: v for t, v in series.items() if t < cut}
        res = ask(seen, hidden)
        last = seen[max(seen)]
        nums = {}
        for k, r in res.items():
            band = float(r["rule"]["band"])
            pred, right, wrong, cannot, hold = [], 0, 0, 0, 0
            for t in hidden:
                truth, got = series[t][k], r["got"][t]
                hold += int(abs(truth - last[k]) <= band)
                if got is None:
                    cannot += 1
                elif abs(truth - got) <= band:
                    right += 1
                else:
                    wrong += 1
                pred.append([t, got, r["src"][t], truth])
            nums[k] = dict(_rule_view(r["rule"]), asked=len(hidden), answered=right + wrong, right=right, wrong=wrong,
                           cannot=cannot, hold_right=hold, pred=pred)
            for q, v in (("asked", len(hidden)), ("answered", right + wrong), ("right", right), ("wrong", wrong),
                         ("cannot", cannot), ("hold_right", hold)):
                total[q] += v
        return nums

    objects, skipped = [], []
    for oid, series in own_series(m).items():
        seen = [t for t in series if t < cut]
        hidden = sorted(t for t in series if t >= cut)
        if len(seen) < C.MIN_TRACK or not hidden:
            skipped.append(oid)
            continue
        objects.append(dict(id=oid, seen=len(seen), numbers=score(series, hidden)))
    cam = camera_series(m)
    camera = score(cam, sorted(t for t in cam if t >= cut))
    return dict(hidden_from=cut, frames=H, objects=objects, camera=camera, skipped=skipped, total=total)


def extension(m, K):
    """AM's values at the K frames after the clip, and the boxes they put on screen"""
    n, L = m["frames"], m["lens"]
    frames = list(range(n, n + K))
    C._length(n)
    cam_series = camera_series(m)
    cam = ask(cam_series, frames)
    last_mk = cam_series[max(cam_series)]
    mk = {t: {a: (cam["m" + a]["got"][t], cam["m" + a]["got"][t] is None) for a in "XYZ"} for t in frames}
    objects, left = [], []
    for oid, series in own_series(m).items():
        if max(series) < n - 1 - GAP or len(series) < C.MIN_TRACK:
            left.append(oid)
            continue
        res = ask(series, frames)
        last = series[max(series)]
        boxes = {}
        for hold in (False, True):
            row = []
            for t in frames:
                own = {k: res[k]["got"][t] for k in res}
                held = [k for k in PLACE if own[k] is None]
                if held and not hold:
                    row.append(None)
                    continue
                own.update({k: last[k] for k in held})
                cam_held = [a for a in "XYZ" if mk[t][a][1]]
                X, Y, Z = ((last_mk["m" + a] if mk[t][a][1] else mk[t][a][0]) + own["r" + a] for a in "XYZ")
                if Z <= 0:
                    row.append(None)
                    continue
                f = L["f"]
                row.append(dict(t=t, x=round(L["cx"] + f * X / Z), y=round(L["cy"] + f * Y / Z),
                                w=round(f * own["W"] / Z), h=round(f * own["H"] / Z), X=X, Y=Y, Z=Z, W=own["W"],
                                H=own["H"], held=held, camera_held=cam_held))
            boxes["held" if hold else "am"] = row
        nums = {k: dict(_rule_view(r["rule"]), values=[r["got"][t] for t in frames], src=[r["src"][t] for t in frames],
                        last=last[k]) for k, r in res.items()}
        objects.append(dict(id=oid, numbers=nums, boxes=boxes))
    camera = {k: dict(_rule_view(r["rule"]), values=[r["got"][t] for t in frames], src=[r["src"][t] for t in frames],
                      last=last_mk[k]) for k, r in cam.items()}
    return dict(first=n, frames=K, camera=camera, objects=objects, left=left)


def extend(mapdir, seconds=2.0, progress=print):
    m = json.load(open(os.path.join(mapdir, "map.json")))
    n, fps = m["frames"], m["fps"]
    K = max(1, int(round(seconds * fps)))
    H = max(1, min(K, n // 3))
    w0 = time.time()
    progress("proof")
    pr = proof(m, H)
    w1 = time.time()
    progress("extend")
    ex = extension(m, K)
    w2 = time.time()
    out = dict(seconds=seconds, fps=fps, clip_frames=n, lens=m["lens"], proof=pr, extension=ex,
               stages=dict(proof=round(w1 - w0, 1), extend=round(w2 - w1, 1)), tokens=0,
               cost_usd=round((w2 - w0) / 3600 * 0.03, 5))
    tmp = os.path.join(mapdir, "extend.json.tmp")
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, os.path.join(mapdir, "extend.json"))
    progress("done")
    return out


def say(e):
    p, x = e["proof"], e["extension"]
    T = p["total"]
    print("PROOF: the last %d frames hidden from AM (from frame %d); objects %s, camera" % (
        p["frames"], p["hidden_from"], [o["id"] for o in p["objects"]]))
    print("  asked %d number-frames: AM answered %d (right %d, wrong %d), CANNOT %d; holding the last value: %d right"
          % (T["asked"], T["answered"], T["right"], T["wrong"], T["cannot"], T["hold_right"]))
    for o in p["objects"] + [dict(id="camera", numbers=p["camera"])]:
        print("  #%s" % o["id"], {k: ("%s %d/%d right, %d wrong, hold %d" % (v["rule"], v["right"], v["asked"], v["wrong"],
                                                                         v["hold_right"]))
                                   for k, v in o["numbers"].items()})
    print("EXTENSION: frames %d-%d; camera %s; left before the end: %s" % (
        x["first"], x["first"] + x["frames"] - 1,
        {k: "%s, %d of %d frames" % (v["rule"], sum(a is not None for a in v["values"]), x["frames"])
         for k, v in x["camera"].items()}, x["left"]))
    for o in x["objects"]:
        print("  #%d" % o["id"], {k: "%s, %d of %d frames" % (v["rule"], sum(a is not None for a in v["values"]), x["frames"])
                                  for k, v in o["numbers"].items()},
              "boxes: AM's %d, with held %d" % (sum(b is not None for b in o["boxes"]["am"]),
                                                 sum(b is not None for b in o["boxes"]["held"])))
    print("time: proof %.1f s, extension %.1f s; about $%s; 0 tokens" % (e["stages"]["proof"], e["stages"]["extend"],
                                                                         e["cost_usd"]))


if __name__ == "__main__":
    say(extend(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0))
