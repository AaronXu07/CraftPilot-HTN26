"""Persistent TripoSR worker: loads the model once and serves reconstructions over localhost HTTP.

    python tools/recon_server.py [--port 7790] [--device mps]

    POST /reconstruct {"image": "/abs/in.png", "out": "/abs/out.ply", "resolution": 256, "keep_bg": false}
        -> {"ok": true, "vertices": N, "faces": M, "seconds": s, "infer_s": …, "mesh_s": …}
    GET  /health -> {"ok": true, "device": "mps", "warm": true}

Runs in tools/.venv-3d (torch + MPS). craftpilot.objects.recon starts it on first use and talks to it; a
cold subprocess call costs 5-18 s of model load per object, the warm server ~1 s inference + ~3 s meshing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "triposr"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

CHUNK = 32768  # triplane query chunk: 8192 (TripoSR default) meshes in ~8 s on an M4 Pro, 32768 in ~3 s
STATE: dict = {"model": None, "device": "cpu", "rembg": None, "warm": False, "lock": threading.Lock()}


def load(device: str) -> None:
    from tsr.system import TSR

    t = time.time()
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(CHUNK)
    model.to(device)
    STATE.update(model=model, device=device)
    print(json.dumps({"event": "loaded", "seconds": round(time.time() - t, 1), "device": device}), flush=True)
    # first MPS call compiles kernels (~10 s); take that hit now instead of on the first player request
    t = time.time()
    blank = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
    with torch.no_grad():
        codes = model([blank], device=device)
        model.extract_mesh(codes, has_vertex_color=False, resolution=64)
    # background removal: u2net is 0.2 s per image once loaded, but the first new_session() downloads and
    # loads 176 MB — do it here, not inside the first player's request
    t = time.time()
    try:
        import rembg

        STATE["rembg"] = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
        blank_rgba = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
        rembg.remove(blank_rgba, session=STATE["rembg"])
        print(json.dumps({"event": "rembg", "seconds": round(time.time() - t, 1)}), flush=True)
    except Exception as e:  # noqa: BLE001 — the flood matte still works without it
        print(json.dumps({"event": "rembg_failed", "error": str(e)[:200]}), flush=True)
    STATE["warm"] = True
    print(json.dumps({"event": "warm", "seconds": round(time.time() - t, 1)}), flush=True)


def flood_matte(img: Image.Image, tol: int = 28, min_fg: float = 0.02, max_fg: float = 0.90) -> Image.Image | None:
    """Alpha matte for a subject on a plain light background: the near-background-coloured region that is
    connected to the image border becomes transparent. Milliseconds, and it keeps white parts of the
    subject (a marble horse) because they are not connected to the border. Returns None when the result
    is implausible (almost nothing or almost everything is foreground) so the caller can fall back to rembg."""
    from scipy import ndimage

    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    near = (np.abs(rgb - bg).max(axis=2) <= tol)
    labels, n = ndimage.label(near)
    if n == 0:
        return None
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge_labels[edge_labels != 0])
    # bright highlights on a white subject touch the background through pixel-thin gaps at the silhouette:
    # opening the background removes those tendrils, then enclosed pockets inside the subject are filled
    k = max(3, int(round(min(h, w) / 200)))
    background = ndimage.binary_opening(background, structure=np.ones((k, k), bool))
    lab2, _ = ndimage.label(background)
    edge2 = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    background = np.isin(lab2, edge2[edge2 != 0])  # only what still touches the border is background
    fg = ndimage.binary_fill_holes(~background)
    share = fg.mean()
    if share < min_fg or share > max_fg:
        return None
    # soften the edge by one pixel so the reconstructor does not see a jagged cut-out
    alpha = ndimage.gaussian_filter(fg.astype(np.float32), 0.7)
    alpha = np.clip((alpha - 0.3) / 0.4, 0.0, 1.0)
    out = np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def prepare(image_path: str, keep_bg: bool, foreground_ratio: float = 0.85) -> tuple[Image.Image, str]:
    from tsr.utils import remove_background, resize_foreground

    img = Image.open(image_path)
    matte = "none"
    if not keep_bg:
        cut = None
        if STATE["rembg"] is not None:
            # neural matte first: it separates white marble from a white background, which no colour
            # threshold can; 0.2 s once the session is warm
            try:
                cut = remove_background(img, STATE["rembg"])
                alpha = np.asarray(cut)[:, :, 3]
                share = float((alpha > 127).mean())
                if share < 0.02 or share > 0.95:
                    cut = None
                else:
                    matte = "u2net"
            except Exception:  # noqa: BLE001
                cut = None
        if cut is None:
            cut = flood_matte(img)
            matte = "flood" if cut is not None else "none"
        if cut is not None:
            img = cut
    img = resize_foreground(img.convert("RGBA"), foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8)), matte


def reconstruct(req: dict) -> dict:
    t0 = time.time()
    img, matte = prepare(req["image"], bool(req.get("keep_bg", False)))
    prep_s = time.time() - t0
    out = req["out"]
    img.save(os.path.splitext(out)[0] + "_input.png")
    model, device = STATE["model"], STATE["device"]
    with STATE["lock"]:
        t = time.time()
        with torch.no_grad():
            codes = model([img], device=device)
        if device == "mps":
            torch.mps.synchronize()
        infer_s = time.time() - t
        t = time.time()
        mesh = model.extract_mesh(codes, has_vertex_color=True, resolution=int(req.get("resolution", 256)))[0]
        mesh_s = time.time() - t
    t = time.time()
    mesh.export(out)
    export_s = time.time() - t
    return {"ok": True, "out": out, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "seconds": round(time.time() - t0, 1), "prep_s": round(prep_s, 1), "infer_s": round(infer_s, 1),
            "mesh_s": round(mesh_s, 1), "export_s": round(export_s, 1), "matte": matte}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": STATE["model"] is not None, "device": STATE["device"], "warm": STATE["warm"]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/reconstruct":
            self._send(404, {"ok": False, "error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if STATE["model"] is None:
                self._send(503, {"ok": False, "error": "model still loading"})
                return
            self._send(200, reconstruct(req))
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("CRAFTPILOT_RECON_PORT", "7790")))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"event": "listening", "port": args.port}), flush=True)
    threading.Thread(target=load, args=(args.device,), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
