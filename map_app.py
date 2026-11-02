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
         ".js": "application/javascript"}


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
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
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
