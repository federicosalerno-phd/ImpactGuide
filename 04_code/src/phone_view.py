"""The live AR on the phone's own screen, or on any phone or tablet on the local network (T20.3).

The app composes the frames (live_ar.LiveAR); this serves them over HTTP on a port of the PC: a minimal full-screen
page in the colours of the app, and its frames one at a time. The page asks for the next frame once it has painted the
last one, so a slow link shows fewer frames, never older ones, and the delay cannot pile up. The same request says
which frame was painted, so the time from a frame composed on the PC to the phone having painted it and asking for
the next is measured here, on the PC's clock: the added latency (an upper bound, it includes the request's way back).

Two ways to the page:
  the phone of the bench, by cable or Wi-Fi: phone_link.Link.open_url opens it in the phone's browser through adb
      (adb reverse: the phone's 127.0.0.1 is this PC's, so neither the network nor the PC's firewall is involved, and
      the browser treats the page as secure: full screen and the screen kept on)
  any device on the same network: http://<this PC>:<port>/v/<token>/, from a QR code in the app. Windows asks once
      to let Python through its firewall on private networks; eduroam and similar networks keep devices apart.

    server = ViewServer(source, words=..., colors=..., lang="it")   # source.next_frame(after, timeout)
    server.url_phone        the page for the phone through adb reverse (127.0.0.1)
    server.url_lan(ip)      the page on the local network
    server.stats()          viewers, frames per second on the page, added latency, kB per frame
    server.close()
    qr_svg(text)            a QR code as SVG markup, dark modules in currentColor

A source gives (JPEG bytes, frame counter, time.perf_counter() when it was composed) for a frame newer than `after`,
or (None, after, None) after `timeout` s.

    python phone_view.py serve [--fps N] [--seconds S]   a test pattern on the network, its address and QR code
    python phone_view.py test       page, frames, wrong token, latency with a client on this PC, a slow client; exit 1
                                    on failure
"""
import argparse
import json
import secrets
import socket
import statistics
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PORT = 8790                     # tried first; another one if it is taken
WAIT_S = 1.0                    # a request for a frame waits this long for a newer one (then 204)
SEEN_S = 3.0                    # a page that asked within this long is a viewer
WORDS = {"off": "Live view off", "off_text": "Start it in ImpactGuide, on the Live AR page.",
         "lost": "No answer from the PC: trying again", "ended": "This view has ended: open it again from ImpactGuide.",
         "tap": "Tap for full screen"}
# The roles the page is painted with (the app passes its palette; these are SlantUI's gold-dark, for the tools here).
COLORS = {"surface-0": "#0D0D0F", "text-2": "#C8C8D0", "text-3": "#A0A0AA", "text-4": "#80808A",
          "warn-text": "#E0A93C", "overlay": "#101013F2"}

PAGE = """<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="__BG__">
<title>ImpactGuide</title>
<style>
:root{__VARS__}
html,body{margin:0;height:100%;background:var(--surface-0);overflow:hidden;touch-action:manipulation;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-user-select:none;user-select:none}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;display:block}
#v.off{visibility:hidden}
#msg{position:fixed;left:24px;right:24px;top:50%;transform:translateY(-50%);text-align:center}
#msg b{display:block;font-size:20px;font-weight:600;color:var(--text-2);margin-bottom:8px}
#msg span{font-size:15px;line-height:1.5;color:var(--text-3)}
#msg.warn b{color:var(--warn-text)}
#tap{position:fixed;left:50%;bottom:calc(14px + env(safe-area-inset-bottom));transform:translateX(-50%);
  padding:6px 14px;border-radius:999px;background:var(--overlay);color:var(--text-4);font-size:13px;white-space:nowrap}
.hide{display:none}
</style>
</head>
<body>
<img id="v" class="off" alt="">
<div id="msg"><b></b><span></span></div>
<div id="tap"></div>
<script>
"use strict";
const W = __WORDS__;
const v = document.getElementById('v'), msg = document.getElementById('msg'), tap = document.getElementById('tap');
const client = Math.random().toString(36).slice(2, 10);
let after = 0, shown = 0, paint = 0, old = '', lock = null;
tap.textContent = W.tap;
function say(title, text, warn) {
  msg.querySelector('b').textContent = title || '';
  msg.querySelector('span').textContent = text || '';
  msg.classList.toggle('hide', !title);
  msg.classList.toggle('warn', !!warn);
  if (title) v.classList.add('off');
}
say(W.off, W.off_text);
function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
function painted() {    // the next paint; a page out of sight gets no paint, so at most 100 ms
  return new Promise(function (r) {
    let done = false;
    const go = function () { if (!done) { done = true; r(); } };
    requestAnimationFrame(go);
    setTimeout(go, 100);
  });
}
async function awake() {
  try { if (navigator.wakeLock && !lock) { lock = await navigator.wakeLock.request('screen');
    lock.addEventListener('release', function () { lock = null; }); } } catch (e) { lock = null; }
}
document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'visible') awake(); });
document.addEventListener('fullscreenchange', function () { tap.classList.toggle('hide', !!document.fullscreenElement); });
document.body.addEventListener('click', function () {
  awake();
  const d = document.documentElement;
  if (!document.fullscreenElement && d.requestFullscreen) {
    d.requestFullscreen({ navigationUI: 'hide' }).then(function () {
      if (screen.orientation && screen.orientation.lock) screen.orientation.lock('landscape').catch(function () {});
    }).catch(function () {});
  }
});
async function loop() {
  for (;;) {
    let r;
    try {
      r = await fetch('frame?after=' + after + '&shown=' + shown + '&paint=' + paint.toFixed(1) + '&c=' + client,
                      { cache: 'no-store' });
    } catch (e) { say(W.lost, '', true); await sleep(1000); continue; }
    if (r.status === 404) { say(W.ended, ''); return; }
    if (r.status === 204) { shown = 0; say(W.off, W.off_text); continue; }
    if (!r.ok) { say(W.lost, '', true); await sleep(1000); continue; }
    const n = +r.headers.get('X-Frame');
    let blob;
    try { blob = await r.blob(); } catch (e) { continue; }
    const t1 = performance.now();
    const url = URL.createObjectURL(blob);
    v.src = url;
    try { await v.decode(); } catch (e) {}
    if (old) URL.revokeObjectURL(old);
    old = url;
    v.classList.remove('off');
    say('');
    await painted();
    paint = performance.now() - t1;
    after = n;
    shown = n;
  }
}
awake();
loop();
</script>
</body>
</html>
"""


def qr_svg(text):
    """The QR code of `text` as SVG markup, its quiet zone included: dark modules in currentColor, nothing behind
    them (the page puts the light tile under it; a scanner wants dark on light)."""
    import cv2
    import numpy as np
    m = cv2.QRCodeEncoder.create().encode(text)
    n = m.shape[0]
    rows = np.flatnonzero((m < 128).any(axis=1))
    q = max(0, 4 - int(rows[0])) if len(rows) else 0     # a quiet zone of 4 modules (OpenCV leaves 2)
    d = []
    for y in range(n):
        x = 0
        while x < n:
            if m[y, x] < 128:
                x0 = x
                while x < n and m[y, x] < 128:
                    x += 1
                d.append(f"M{x0} {y}h{x - x0}v1h-{x - x0}z")
            else:
                x += 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-q} {-q} {n + 2 * q} {n + 2 * q}" '
            f'shape-rendering="crispEdges"><path fill="currentColor" d="{"".join(d)}"/></svg>')


def lan_ip(towards=""):
    """This PC's IPv4 address on the network that reaches `towards` (the phone's address), else on the default route;
    nothing is sent."""
    for target in ([towards] if towards else []) + ["192.0.2.1"]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                ip = s.getsockname()[0]
            if ip and not ip.startswith(("127.", "0.")):
                return ip
        except OSError:
            pass
    return "127.0.0.1"


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A page closed, a phone gone off the network: nothing to report."""
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class ViewServer:
    """The page and its frames on a port of this PC, every interface (see the module)."""

    def __init__(self, source, host="0.0.0.0", port=PORT, words=None, colors=None, lang="en"):
        self.source = source
        self.token = secrets.token_urlsafe(9)
        self.words, self.colors, self.lang = {**WORDS, **(words or {})}, {**COLORS, **(colors or {})}, lang
        self._lock = threading.Lock()
        self._sent = {}                     # frame -> (t composed, bytes), the last ones served
        self._acks = deque(maxlen=240)      # (t, client, added ms, painting on the page ms, bytes)
        self._clients = {}                  # client -> last time it asked
        self.closed = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"   # one connection per page, kept: no handshake per frame

            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_GET(self):
                url = urlsplit(self.path)
                base = f"/v/{server.token}"
                try:
                    if url.path == base:
                        self._send(301, extra={"Location": base + "/"})
                    elif url.path == base + "/":
                        self._send(200, server.page().encode("utf-8"), "text/html; charset=utf-8")
                    elif url.path == base + "/frame":
                        self._frame(parse_qs(url.query))
                    else:
                        self._send(404, b"not found")
                except (ConnectionError, OSError):
                    pass

            def _frame(self, q):
                def num(k, kind=int):
                    try:
                        return kind(q.get(k, ["0"])[0])
                    except ValueError:
                        return kind(0)
                now = time.perf_counter()
                server._seen(q.get("c", [self.client_address[0]])[0], now, num("shown"), num("paint", float))
                jpeg, n, t = server.source.next_frame(num("after"), timeout=WAIT_S)
                if jpeg is None or server.closed:
                    self._send(204)
                    return
                with server._lock:
                    server._sent[n] = (t, len(jpeg))
                    if len(server._sent) > 240:
                        for k in sorted(server._sent)[:-120]:
                            del server._sent[k]
                self._send(200, jpeg, "image/jpeg", {"X-Frame": str(n)})

        self.httpd = None
        for p in (port, 0):
            try:
                self.httpd = _Server((host, p), Handler)
                break
            except OSError:
                continue
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="phone_view")
        self.thread.start()

    def _seen(self, client, now, shown, paint_ms):
        with self._lock:
            self._clients[client] = now
            sent = self._sent.get(shown)
            if shown and sent is not None and sent[0] is not None:
                self._acks.append((now, client, (now - sent[0]) * 1e3, paint_ms, sent[1]))

    def page(self):
        vars_ = "".join(f"--{k}:{v};" for k, v in self.colors.items())
        words = json.dumps(self.words, ensure_ascii=False).replace("</", "<\\/")
        return (PAGE.replace("__LANG__", self.lang).replace("__BG__", self.colors["surface-0"])
                .replace("__VARS__", vars_).replace("__WORDS__", words))

    @property
    def path(self):
        return f"/v/{self.token}/"

    @property
    def url_phone(self):
        """The page for the phone through adb reverse (phone_link.Link.open_url(url, port))."""
        return f"http://127.0.0.1:{self.port}{self.path}"

    def url_lan(self, ip=None):
        return f"http://{ip or lan_ip()}:{self.port}{self.path}"

    def stats(self):
        """Over the last SEEN_S: viewers, frames per second per viewer, added latency (median and 95th percentile,
        ms), painting on the page (ms), kB per frame and Mbit/s."""
        now = time.perf_counter()
        with self._lock:
            acks = [a for a in self._acks if now - a[0] < SEEN_S]
            viewers = sum(1 for t in self._clients.values() if now - t < SEEN_S)
        out = {"viewers": viewers, "fps": 0.0, "added_ms": None, "added_p95_ms": None, "page_ms": None, "kb": None,
               "mbit_s": 0.0, "frames": len(acks)}
        if len(acks) >= 2:
            span = max(acks[-1][0] - acks[0][0], 1e-6)
            per = {}
            for a in acks:
                per.setdefault(a[1], []).append(a[0])
            fps = [(len(ts) - 1) / max(ts[-1] - ts[0], 1e-6) for ts in per.values() if len(ts) >= 2]
            added = sorted(a[2] for a in acks)
            out.update({"fps": round(statistics.mean(fps), 1) if fps else 0.0,
                        "added_ms": round(statistics.median(added), 1),
                        "added_p95_ms": round(added[min(len(added) - 1, int(0.95 * len(added)))], 1),
                        "page_ms": round(statistics.median(a[3] for a in acks), 1),
                        "kb": round(statistics.mean(a[4] for a in acks) / 1024, 1),
                        "mbit_s": round(sum(a[4] for a in acks) * 8 / span / 1e6, 1)})
        return out

    def close(self):
        self.closed = True
        self.httpd.shutdown()
        self.httpd.server_close()


# ── a stand-in for the live AR ───────────────────────────────────────────────
class PatternSource:
    """Frames at `fps` like the live AR's (1280 x 960 JPEG at quality 80): a moving pattern with its counter."""

    def __init__(self, fps=30, size=(1280, 960), quality=80):
        import cv2
        import numpy as np
        self.fps, self.size, self.quality = fps, size, quality
        self._cv, self._np = cv2, np
        self._cond = threading.Condition()
        self._jpeg, self._n, self._t = None, 0, None
        self._stop = threading.Event()
        rng = np.random.default_rng(0)
        w, h = size
        # texture like a camera frame, so the JPEG has the size of a real one
        base = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2.0)
        self._base = cv2.addWeighted(base, 0.6, np.full_like(base, 90), 0.4, 0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cv2, np = self._cv, self._np
        t_next = time.perf_counter()
        while not self._stop.is_set():
            img = np.roll(self._base, self._n * 8, axis=1)
            cv2.putText(img, f"{self._n + 1}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (66, 197, 245), 6)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            with self._cond:
                self._jpeg, self._n, self._t = buf.tobytes(), self._n + 1, time.perf_counter()
                self._cond.notify_all()
            t_next += 1.0 / self.fps
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def next_frame(self, after=0, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._jpeg is not None and self._n != after, timeout=timeout)
            if self._jpeg is None or self._n == after:
                return None, after, None
            return self._jpeg, self._n, self._t

    def close(self):
        self._stop.set()


def client(url, seconds, paint_s=0.0, name="py"):
    """A page as the phone runs it, in Python: asks for the next frame, decodes it, `paint_s` stands in for painting,
    says which frame it showed. Returns the frames shown."""
    import http.client
    import cv2
    import numpy as np
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
    after = shown = 0
    paint = 0.0
    frames = 0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        conn.request("GET", f"{u.path}frame?after={after}&shown={shown}&paint={paint:.1f}&c={name}")
        r = conn.getresponse()
        body = r.read()
        if r.status != 200:
            continue
        t1 = time.perf_counter()
        cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if paint_s:
            time.sleep(paint_s)
        paint = (time.perf_counter() - t1) * 1e3
        after = shown = int(r.getheader("X-Frame"))
        frames += 1
    conn.close()
    return frames


def find_browser():
    """Edge or Chrome on this PC, for running the page in a real browser engine; None if neither."""
    import os
    import shutil
    cands = [shutil.which("msedge"), shutil.which("chrome")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    return next((c for c in cands if c and Path(c).is_file()), None)


def page_in_browser(browser, server, seconds=7.0, size=(1600, 1200)):
    """The page of `server` in `browser` without a window (its own throwaway profile) for `seconds`; the server's
    stats while it ran."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        p = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--mute-audio",
                              "--disable-extensions", f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}",
                              server.url_phone], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=0x08000000 if sys.platform == "win32" else 0)
        try:
            time.sleep(seconds)
            st = server.stats()
        finally:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True,
                               creationflags=0x08000000)
            else:
                p.kill()
            p.wait(timeout=10)
            time.sleep(0.5)
    return st


def run_test():
    import urllib.error
    import urllib.request
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    src = PatternSource(fps=30)
    srv = ViewServer(src, host="127.0.0.1", port=0, words={"tap": "Tocca per lo schermo intero"},
                     colors={"surface-0": "#000001"}, lang="it")
    try:
        with urllib.request.urlopen(srv.url_phone, timeout=5) as r:
            html = r.read().decode("utf-8")
        expect('lang="it"' in html and "--surface-0:#000001;" in html and "Tocca per lo schermo intero" in html
               and "</script>" in html and "requestFullscreen" in html,
               "page: language, palette and words of the app, full screen on a tap")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/v/wrongtoken/", timeout=5)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        expect(code == 404, "a page without the token: 404")
        with urllib.request.urlopen(srv.url_phone + "frame?after=0", timeout=5) as r:
            body, n = r.read(), int(r.headers["X-Frame"])
        expect(body[:2] == b"\xff\xd8" and n > 0, f"a frame: JPEG of {len(body) / 1024:.0f} kB, number {n}")

        frames = client(srv.url_phone, 4.0)
        time.sleep(0.05)
        st = srv.stats()
        expect(st["viewers"] == 1 and st["fps"] > 25 and st["added_ms"] is not None and st["added_ms"] < 40,
               f"client on this PC: {frames} frames, {st['fps']} fps, added latency median {st['added_ms']} ms "
               f"(95th {st['added_p95_ms']} ms), decoding {st['page_ms']} ms, {st['kb']} kB per frame, "
               f"{st['mbit_s']} Mbit/s")
        fast = st

        # the page itself, in a real browser engine (Edge or Chrome without a window), if there is one
        browser = find_browser()
        if browser:
            time.sleep(SEEN_S + 0.2)      # the Python client out of the window
            st = page_in_browser(browser, srv, seconds=7.0)
            expect(st["viewers"] == 1 and st["fps"] > 20 and st["added_ms"] is not None and st["added_ms"] < 60,
                   f"the page in {Path(browser).stem} (headless): {st['fps']} fps, added latency median "
                   f"{st['added_ms']} ms (95th {st['added_p95_ms']} ms), decoding and painting {st['page_ms']} ms")
            browser_st = st
        else:
            print("      no Edge or Chrome here: the page was not run in a browser")
            browser_st = None

        # a slow page (100 ms to paint): fewer frames, never older ones
        time.sleep(SEEN_S + 0.2)
        t0 = time.perf_counter()
        th = threading.Thread(target=client, args=(srv.url_phone, 6.0, 0.1, "slow"))
        th.start()
        time.sleep(2.5)
        early = srv.stats()
        th.join()
        late = srv.stats()
        expect(late["fps"] < 11 and late["added_ms"] < 160 and late["added_ms"] < early["added_ms"] + 30,
               f"slow page (100 ms per frame): {late['fps']} fps, added latency {early['added_ms']} ms after 2.5 s "
               f"and {late['added_ms']} ms after 6 s: it does not grow")
        src.close()
        time.sleep(0.2)
        t0 = time.perf_counter()
        with urllib.request.urlopen(srv.url_phone + "frame?after=" + str(src._n), timeout=5) as r:
            code = r.status
        expect(code == 204 and 0.8 < time.perf_counter() - t0 < 2.0, "no newer frame within 1 s: 204, the page waits")
        import cv2
        import numpy as np
        import re
        read = 0
        for k in range(20):             # addresses of every length the app makes
            url = f"http://192.168.43.{10 + k}:{8790 + k}/v/{secrets.token_urlsafe(9)}/"
            svg = qr_svg(url)
            x0, y0, size = (int(v) for v in re.search(r'viewBox="(-?\d+) (-?\d+) (\d+) ', svg).groups())
            img = np.full((size, size), 255, np.uint8)
            for x, y, w in re.findall(r"M(\d+) (\d+)h(\d+)", svg):
                img[int(y) - y0, int(x) - x0:int(x) - x0 + int(w)] = 0
            big = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            # OpenCV's plain detector misses about 1 sharp synthetic code in 100; its ArUco-based one reads them all
            read += url in (cv2.QRCodeDetectorAruco().detectAndDecode(big)[0], cv2.QRCodeDetector().detectAndDecode(big)[0])
        expect(read == 20, f"QR codes of the address, as drawn, read back: {read} of 20")
        expect(lan_ip() != "", f"address of this PC on the network: {lan_ip()}")
    finally:
        srv.close()
        src.close()
    print(f"\nadded latency on this PC: Python client median {fast['added_ms']} ms (95th {fast['added_p95_ms']} ms)"
          + (f", browser page median {browser_st['added_ms']} ms (95th {browser_st['added_p95_ms']} ms)"
             if browser_st else ""))
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["serve", "test"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=120.0)
    a = ap.parse_args()
    if a.command == "test":
        sys.exit(run_test())
    src = PatternSource(a.fps)
    srv = ViewServer(src)
    url = srv.url_lan()
    print(f"{url}\n(adb reverse tcp:{srv.port} tcp:{srv.port} for {srv.url_phone})", flush=True)
    try:
        import cv2
        qr = cv2.QRCodeEncoder.create().encode(url)
        for row in qr[::1]:
            print("".join("  " if v else "██" for v in row))
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            time.sleep(2.0)
            print(srv.stats(), flush=True)
    finally:
        srv.close()
        src.close()


if __name__ == "__main__":
    main()
