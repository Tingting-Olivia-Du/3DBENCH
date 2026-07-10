#!/usr/bin/env python3
"""Review server: static files from out/ + POST /submit receives reviewer
verdicts/notes and writes them to out/verdicts/ (watched by Claude via Monitor).
Replaces the plain http.server on :8737."""
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/workspace/tingting/3dvla-stage0/out")
VERDICTS = ROOT / "verdicts"
VERDICTS.mkdir(exist_ok=True)


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        if self.path != "/submit":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            data = json.loads(body)
            page = str(data.get("page", "unknown"))[:60].replace("/", "_")
        except Exception:
            self.send_error(400)
            return
        ts = time.strftime("%m%d_%H%M%S")
        p = VERDICTS / f"{page}_{ts}.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8737), H).serve_forever()
