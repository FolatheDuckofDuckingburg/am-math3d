"""AM MAP, live: drop a video and watch the map AM builds of it.  AM's eyes (Apple Vision) box every object, metric
depth lifts each box into 3D through a stated lens, background patches cancel the camera, and AM's core reasoner
decides every number: a rule with its proof, or CANNOT.  Everything runs on this machine; files never leave it.
    python3 map_app.py          then open http://127.0.0.1:8792"""
import hashlib, json, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "math3d"))
PORT = int(os.environ.get("PORT", "8792"))
MAPS = os.path.join(HERE, "maps")
UI = os.path.join(HERE, "ui", "map.html")
JOBS = {}
WORK = threading.Lock()                                        # one map at a time: depth model and AM's eyes use memory
TYPES = {".json": "application/json", ".mp4": "video/mp4", ".html": "text/html; charset=utf-8", ".png": "image/png",
         ".js": "application/javascript", ".txt": "text/plain; charset=utf-8", ".sha256": "text/plain; charset=utf-8"}
SHOT_FILES = r"plan\.json|plan\.sha256|guide\.mp4|guide_end\.png|prompt\.txt|shot\.json|take\.mp4|score\.json|labelled\.mp4"


def run_shot(jid, mid, q):
    def progress(stage):
        JOBS[jid]["stage"] = stage
    try:
        with WORK:
            import am_shot
            am_shot.plan(os.path.join(MAPS, mid), q, progress=progress)
        JOBS[jid]["done"] = True
    except Exception as e:
        JOBS[jid]["error"] = "%s: %s" % (type(e).__name__, e)


def run_score(jid, mid, sid):
    def progress(stage):
        JOBS[jid]["stage"] = stage
    try:
        with WORK:
            import am_shot
            d = os.path.join(MAPS, mid, "shots", sid)
            am_shot.score(os.path.join(MAPS, mid), sid, os.path.join(d, "take.mp4"), progress=progress)
        JOBS[jid]["done"] = True
    except Exception as e:
        JOBS[jid]["error"] = "%s: %s" % (type(e).__name__, e)


def shot_question(mid, body):
    """the intent from the page, checked before AM works on it"""
    q = json.loads(body or b"{}")
    pz = json.load(open(os.path.join(MAPS, mid, "puzzle.json")))
    ids = {p["id"] for p in pz["pieces"]}
    out = dict(mover=int(q["mover"]), mode=q.get("mode", "slide"), start=float(q.get("start", 0.5)),
               length=float(q.get("length", 3.2)), name=str(q.get("name", ""))[:80], fill=q.get("fill", "smooth"))
    if out["mover"] not in ids or out["mode"] not in ("slide", "jump") or out["fill"] not in ("smooth", "stage", "nearest"):
        raise ValueError("unknown piece, motion or fill")
    if not (0 <= out["start"] <= 20 and 1 <= out["length"] <= 20):
        raise ValueError("the start or the length is out of range")
    if out["mode"] == "slide":
        out["duration"] = float(q.get("duration", 1.0))
        if not 0.1 <= out["duration"] <= 10:
            raise ValueError("the slide's time is out of range")
    else:
        out["height"] = float(q.get("height", 14))
        if not 0.5 <= out["height"] <= 300:
            raise ValueError("the arc's height is out of range")
    if q.get("onto") not in (None, ""):
        out["onto"] = int(q["onto"])
        if out["onto"] not in ids or out["onto"] == out["mover"]:
            raise ValueError("the target piece is not on the puzzle")
    else:
        out["across"], out["forward"] = round(float(q["across"]), 1), round(float(q["forward"]), 1)
    return out


def shots_of(mid):
    d = os.path.join(MAPS, mid, "shots")
    out = []
    for sid in os.listdir(d) if os.path.isdir(d) else []:
        p = os.path.join(d, sid, "shot.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        sp = os.path.join(d, sid, "score.json")
        if os.path.exists(sp):
            sc = json.load(open(sp))
            s["score"] = {k: sc[k] for k in ("verdict", "why", "median_px", "worst_px", "worst_t", "end_px", "measured",
                                             "in_plan", "lost", "camera_drift_max_px", "lock_ok", "take", "seconds")}
        s["mtime"] = os.path.getmtime(p)
        out.append(s)
    return sorted(out, key=lambda s: -s["mtime"])


def run_job(jid, path, kind, seconds=2.0):
    def progress(stage):
        JOBS[jid]["stage"] = stage
    try:
        with WORK:
            import am_map, am_puzzle
            am_map.build(path, os.path.join(MAPS, jid), kind, progress=progress)
            am_puzzle.build(os.path.join(MAPS, jid), progress=progress)               # the last frame as AM's puzzle
        JOBS[jid]["done"] = True
    except Exception as e:                                     # the page shows what went wrong
        JOBS[jid]["error"] = "%s: %s" % (type(e).__name__, e)


def run_puzzle(jid, mid, frame):
    def progress(stage):
        JOBS[jid]["stage"] = "puzzle-" + stage
    try:
        with WORK:
            import am_puzzle
            am_puzzle.build(os.path.join(MAPS, mid), frame, progress=progress)
        JOBS[jid]["done"] = True
    except Exception as e:
        JOBS[jid]["error"] = "%s: %s" % (type(e).__name__, e)


def name_of(jid):
    p = os.path.join(MAPS, jid, "name.txt")
    return open(p).read().strip() if os.path.exists(p) else ("Sora sample: mouse chef" if jid == "sample" else jid)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path):
        size = os.path.getsize(path)
        ctype = TYPES.get(os.path.splitext(path)[1], "application/octet-stream")
        rng = self.headers.get("Range")
        with open(path, "rb") as f:
            if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):          # video seeking needs ranges
                a = int(m.group(1)) if m.group(1) else 0
                b = int(m.group(2)) if m.group(2) else size - 1
                f.seek(a)
                body = f.read(b - a + 1)
                return self._send(206, body, ctype, {"Content-Range": "bytes %d-%d/%d" % (a, b, size),
                                                     "Accept-Ranges": "bytes"})
            self._send(200, f.read(), ctype, {"Accept-Ranges": "bytes"})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(UI)
        if u.path == "/vendor/three.min.js":                       # three.js r128, kept here so the app works offline
            return self._file(os.path.join(HERE, "ui", "vendor", "three.min.js"))
        if u.path == "/api/maps":
            ids = sorted((d for d in os.listdir(MAPS) if os.path.exists(os.path.join(MAPS, d, "map.json"))),
                         key=lambda d: (d != "sample", -os.path.getmtime(os.path.join(MAPS, d, "map.json"))))
            return self._json([dict(id=d, name=name_of(d)) for d in ids])
        if u.path.startswith("/api/status/"):
            jid = u.path.rsplit("/", 1)[1]
            if jid not in JOBS and re.fullmatch(r"[A-Za-z0-9_-]+", jid) and os.path.exists(os.path.join(MAPS, jid, "map.json")):
                return self._json(dict(stage="done", done=True))       # built before this app started
            return self._json(JOBS.get(jid, dict(error="unknown map")))
        m = re.fullmatch(r"/maps/([A-Za-z0-9_-]+)/(map\.json|puzzle\.json|puzzle_frame\.png|puzzle_labels\.png|clip\.mp4)", u.path)
        if m and os.path.exists(os.path.join(MAPS, m.group(1), m.group(2))):
            return self._file(os.path.join(MAPS, m.group(1), m.group(2)))
        m = re.fullmatch(r"/api/shots/([A-Za-z0-9_-]+)", u.path)
        if m:
            return self._json(shots_of(m.group(1)))
        m = re.fullmatch(r"/maps/([A-Za-z0-9_-]+)/shots/([0-9a-f]{10})/(%s)" % SHOT_FILES, u.path)
        if m and os.path.exists(os.path.join(MAPS, m.group(1), "shots", m.group(2), m.group(3))):
            return self._file(os.path.join(MAPS, m.group(1), "shots", m.group(2), m.group(3)))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        m = re.fullmatch(r"/api/shot/([A-Za-z0-9_-]+)", u.path)
        if m:                                                  # an intent on the puzzle: AM's numbers, the lock, the guide
            mid = m.group(1)
            if not os.path.exists(os.path.join(MAPS, mid, "puzzle.json")):
                return self._json(dict(error="this map has no puzzle yet"), 404)
            try:
                q = shot_question(mid, self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, KeyError, TypeError) as e:
                return self._json(dict(error="The shot could not be read: %s" % e), 400)
            import am_shot
            sid = am_shot.shot_id(os.path.join(MAPS, mid), q)
            jid = "shot-%s-%s" % (mid, sid)
            if os.path.exists(os.path.join(MAPS, mid, "shots", sid, "shot.json")):
                JOBS[jid] = dict(stage="done", done=True)
            elif jid not in JOBS or JOBS[jid].get("error"):
                JOBS[jid] = dict(stage="queued")
                threading.Thread(target=run_shot, args=(jid, mid, q), daemon=True).start()
            return self._json(dict(id=jid, shot=sid))
        m = re.fullmatch(r"/api/take/([A-Za-z0-9_-]+)/([0-9a-f]{10})", u.path)
        if m:                                                  # the take a video model painted: measured against the lock
            mid, sid = m.group(1), m.group(2)
            d = os.path.join(MAPS, mid, "shots", sid)
            if not os.path.exists(os.path.join(d, "guide.mp4")):
                return self._json(dict(error="no guide for this shot"), 404)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if not body:
                return self._json(dict(error="no video received"), 400)
            with open(os.path.join(d, "take.mp4"), "wb") as f:
                f.write(body)
            for old in ("score.json", "labelled.mp4"):
                if os.path.exists(os.path.join(d, old)):
                    os.remove(os.path.join(d, old))
            jid = "take-%s-%s-%s" % (mid, sid, hashlib.sha1(body).hexdigest()[:8])
            JOBS[jid] = dict(stage="queued")
            threading.Thread(target=run_score, args=(jid, mid, sid), daemon=True).start()
            return self._json(dict(id=jid, shot=sid))
        m = re.fullmatch(r"/api/puzzle/([A-Za-z0-9_-]+)", u.path)
        if m:                                                  # AM's puzzle of one frame of a map already built
            mid = m.group(1)
            if not os.path.exists(os.path.join(MAPS, mid, "map.json")):
                return self._json(dict(error="no such map"), 404)
            fr = parse_qs(u.query).get("frame", [None])[0]
            frame = int(fr) if fr and fr.isdigit() else None
            jid = "pz-%s-%s" % (mid, "last" if frame is None else frame)
            if jid not in JOBS or JOBS[jid].get("error") or JOBS[jid].get("done"):
                JOBS[jid] = dict(stage="queued")
                threading.Thread(target=run_puzzle, args=(jid, mid, frame), daemon=True).start()
            return self._json(dict(id=jid))
        if u.path != "/api/map":
            return self._send(404, b"not found", "text/plain")
        kind = parse_qs(u.query).get("kind", ["indoor"])[0]
        kind = kind if kind in ("indoor", "outdoor") else "indoor"
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if not body:
            return self._json(dict(error="no video received"), 400)
        jid = hashlib.sha1(body + kind.encode()).hexdigest()[:12]
        d = os.path.join(MAPS, jid)
        if os.path.exists(os.path.join(d, "map.json")):
            JOBS[jid] = dict(stage="done", done=True)
            return self._json(dict(id=jid))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "name.txt"), "w") as f:
            f.write(unquote(self.headers.get("X-Name", "clip.mp4"))[:120])
        path = os.path.join(d, "clip.mp4")
        with open(path, "wb") as f:
            f.write(body)
        if jid not in JOBS or JOBS[jid].get("error"):
            JOBS[jid] = dict(stage="queued")
            threading.Thread(target=run_job, args=(jid, path, kind), daemon=True).start()
        self._json(dict(id=jid))


if __name__ == "__main__":
    os.makedirs(MAPS, exist_ok=True)
    print("AM Map on http://127.0.0.1:%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
