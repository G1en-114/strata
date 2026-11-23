"""Local helper for the strata server: picture -> strokes (tools/paint_image.py), kept warm in one process.

    python tools/trace_service.py [--port 11436]

POST /trace?sketch=0|1&steps=120&layers=auto|semantic|colour   body: the image file (any format PIL reads)
  -> {"records": [[kind, event, layer, attrs...], ...] (BOS .. EOS, draft order; sketch=1: up to the first REFINE),
      "psnr", "strokes", "lines", "seconds"}
GET  /health -> {"ok": true}
Listens on 127.0.0.1 only; the C++ server (strata serve) calls it for /v1/images/edits and /api/trace. The decomposer
is PyTorch on the GPU; text-to-image in the C++ server does not need this process.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paint_image  # noqa: E402

LOCK = threading.Lock()                     # one decomposition at a time on the GPU


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"ok": True}) if self.path.startswith("/health") else self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path != "/trace":
            return self._send(404, {"error": "not found"})
        q = parse_qs(u.query)
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 32 << 20:
            return self._send(400, {"error": "send the image as the request body (at most 32 MB)"})
        try:
            orig = Image.open(io.BytesIO(self.rfile.read(n)))
            orig.load()
            img = paint_image.load_square(orig)
        except Exception as e:                  # noqa: BLE001 - any undecodable upload
            return self._send(400, {"error": f"cannot read the image: {e}"})
        try:
            with LOCK:
                r = paint_image.trace(img, steps=int(q.get("steps", ["120"])[0]),
                                      sketch_only=q.get("sketch", ["0"])[0] == "1", layers=q.get("layers", ["auto"])[0],
                                      orig=orig)
        except Exception as e:                  # noqa: BLE001
            return self._send(500, {"error": f"tracing failed: {e}"})
        pal = np.array([[235, 235, 235], [80, 120, 200], [230, 180, 150], [220, 60, 60], [120, 80, 40], [240, 220, 60]],
                       np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pal[r["labels"]]).save(buf, format="PNG")
        self._send(200, {"records": r["records"].tolist(), "psnr": r["psnr"], "strokes": r["strokes"],
                         "lines": r["lines"], "seconds": r["seconds"], "layering": r["layering"], "parser": r["parser"],
                         "cropped": r["cropped"],
                         "layers_png": base64.b64encode(buf.getvalue()).decode()})

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write("trace: " + fmt % args + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11436)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"trace service on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
