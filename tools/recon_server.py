"""Persistent reconstruction worker: loads the models once and serves image -> mesh over localhost HTTP.

    python tools/recon_server.py [--port 7790] [--device mps]

    POST /reconstruct {"image": "/abs/in.png", "out": "/abs/out.ply", "engine": "hunyuan"|"triposr",
                       "resolution": 256, "keep_bg": false}
        -> {"ok": true, "engine": …, "up_axis": "y"|"z", "front_axis": "+z"|"+x", "vertices": N, "faces": M,
            "seconds": s, "prep_s": …, "infer_s": …, "mesh_s": …, "matte": "u2net"|"flood"}
    GET  /health -> {"ok": true, "device": "mps", "warm": true, "engines": [...]}

Engines (both in tools/.venv-3d, torch + MPS):
- hunyuan  — Hunyuan3D-2 mini-turbo (0.6B flow-matching shape model, fp16, FlashVDM decoder): a real 3D
             body from one image, ~13 s on an M4 Pro at octree 256. Shape only, so the reference image is
             projected onto the mesh for vertex colours. Frame: y up, image-left = -x, camera on +z.
- triposr  — TripoSR: ~2 s, but a shallow relief-like guess of depth. Frame: z up, camera on +x.
craftpilot.objects.recon starts this server on first use; a cold subprocess (recon_worker.py, TripoSR only)
is the fallback when it cannot.
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
sys.path.insert(0, os.path.join(HERE, "hunyuan3d"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

CHUNK = 32768  # triplane query chunk: 8192 (TripoSR default) meshes in ~8 s on an M4 Pro, 32768 in ~3 s
HUNYUAN_REPO, HUNYUAN_SUBFOLDER = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"
HUNYUAN_STEPS = 5  # the turbo model is step-distilled; 5 is its intended budget
STATE: dict = {"model": None, "hunyuan": None, "device": "cpu", "rembg": None, "warm": False, "lock": threading.Lock()}


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
    # Hunyuan3D-2 mini-turbo after TripoSR, so the fast engine is available while this one loads (~20 s)
    if os.environ.get("CRAFTPILOT_HUNYUAN", "1").lower() not in ("0", "false", "off", "no"):
        t = time.time()
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            dtype = torch.float16 if device == "mps" else torch.float32
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN_REPO, subfolder=HUNYUAN_SUBFOLDER, device=device, dtype=dtype, use_safetensors=True, variant="fp16")
            pipe.enable_flashvdm(mc_algo="mc")  # 134 s -> 13 s per object on MPS
            from PIL import ImageDraw

            blob = Image.new("RGBA", (256, 256), (0, 0, 0, 0))  # warm-up subject: an opaque grey disc
            ImageDraw.Draw(blob).ellipse((48, 48, 208, 208), fill=(140, 140, 140, 255))
            with torch.no_grad():  # first MPS call compiles kernels
                pipe(image=blob, num_inference_steps=1, octree_resolution=64, generator=torch.Generator().manual_seed(0))
            STATE["hunyuan"] = pipe
            print(json.dumps({"event": "hunyuan", "seconds": round(time.time() - t, 1)}), flush=True)
        except Exception as e:  # noqa: BLE001 — TripoSR still serves
            print(json.dumps({"event": "hunyuan_failed", "error": str(e)[:300]}), flush=True)


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
    rgba = resize_foreground(img.convert("RGBA"), foreground_ratio)
    arr = np.array(rgba).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8)), matte, rgba


def project_colors(mesh, rgba: Image.Image, flip_x: bool = False):
    """Vertex colours for a shape-only mesh: orthographic projection of the reference image along the
    camera axis (z), mapping the mesh's x/y extent onto the subject's alpha bounding box (image-left = -x,
    image-up = +y). Vertices whose projection lands on transparent pixels (silhouette edges, and the far
    side, which shares the front's colours) take the nearest opaque pixel."""
    from scipy.spatial import cKDTree

    a = np.asarray(rgba)
    alpha = a[:, :, 3] > 127
    ys, xs = np.nonzero(alpha)
    if xs.size == 0:
        return np.full((len(mesh.vertices), 4), 160, dtype=np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    u = (v[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-6)
    if flip_x:
        u = 1.0 - u
    w = (v[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-6)
    col = np.clip(np.round(x0 + u * (x1 - x0)), 0, a.shape[1] - 1).astype(int)
    row = np.clip(np.round(y1 - w * (y1 - y0)), 0, a.shape[0] - 1).astype(int)
    hit = alpha[row, col]
    rgb = a[row, col, :3].copy()
    if hit.any() and not hit.all():
        # a vertex whose projection misses the silhouette (the far side, a foreshortened limb) takes the
        # colour of the nearest vertex that hit, in 3D — surface-coherent, unlike the nearest edge pixel,
        # which is the darkest outline of the drawing
        tree = cKDTree(v[hit])
        _, idx = tree.query(v[~hit], k=1)
        rgb[~hit] = rgb[hit][idx]
    return np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1)


def reconstruct(req: dict) -> dict:
    t0 = time.time()
    engine = str(req.get("engine") or os.environ.get("CRAFTPILOT_RECON_ENGINE", "hunyuan")).lower()
    if engine == "hunyuan" and STATE["hunyuan"] is None:
        engine = "triposr"  # not loaded (yet, or disabled): the fast engine answers
    img, matte, rgba = prepare(req["image"], bool(req.get("keep_bg", False)))
    prep_s = time.time() - t0
    out = req["out"]
    img.save(os.path.splitext(out)[0] + "_input.png")
    device = STATE["device"]
    resolution = int(req.get("resolution", 256))
    with STATE["lock"]:
        if engine == "hunyuan":
            pipe = STATE["hunyuan"]
            t = time.time()
            with torch.no_grad():
                mesh = pipe(image=rgba, num_inference_steps=HUNYUAN_STEPS, octree_resolution=resolution,
                            generator=torch.Generator().manual_seed(int(req.get("seed", 1))))[0]
            if device == "mps":
                torch.mps.synchronize()
            infer_s = time.time() - t
            t = time.time()
            import trimesh

            mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
            mesh.visual.vertex_colors = project_colors(mesh, rgba)
            mesh_s = time.time() - t
            up_axis, front_axis = "y", "+z"
        else:
            model = STATE["model"]
            t = time.time()
            with torch.no_grad():
                codes = model([img], device=device)
            if device == "mps":
                torch.mps.synchronize()
            infer_s = time.time() - t
            t = time.time()
            mesh = model.extract_mesh(codes, has_vertex_color=True, resolution=resolution)[0]
            mesh_s = time.time() - t
            up_axis, front_axis = "z", "+x"
    t = time.time()
    mesh.export(out)
    export_s = time.time() - t
    return {"ok": True, "out": out, "engine": engine, "up_axis": up_axis, "front_axis": front_axis,
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
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
            engines = [e for e, m in (("triposr", STATE["model"]), ("hunyuan", STATE["hunyuan"])) if m is not None]
            self._send(200, {"ok": STATE["model"] is not None, "device": STATE["device"], "warm": STATE["warm"], "engines": engines})
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
