"""Image -> vertex-coloured mesh with TripoSR. Runs in tools/.venv-3d (torch + MPS), called as a subprocess.

    python tools/recon_worker.py input.png out.ply [--resolution 256] [--device mps] [--keep-bg]

Writes `out.ply` (vertex colours) and prints one JSON line: {"ok": true, "vertices": N, "faces": M, "seconds": s}.
The model weights (stabilityai/TripoSR, ~1.7 GB) are downloaded to the Hugging Face cache on first use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "triposr"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--resolution", type=int, default=256, help="marching cubes grid (256 is TripoSR's default)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--keep-bg", action="store_true", help="skip rembg (image already has an alpha matte)")
    ap.add_argument("--foreground-ratio", type=float, default=0.85)
    args = ap.parse_args()
    t0 = time.time()

    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(args.device)
    t_model = time.time()

    img = Image.open(args.image)
    if not args.keep_bg:
        import rembg

        img = remove_background(img, rembg.new_session())
    img = resize_foreground(img.convert("RGBA"), args.foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5  # grey background, like the reference script
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(os.path.splitext(args.out)[0] + "_input.png")

    with torch.no_grad():
        scene_codes = model([img], device=args.device)
    t_codes = time.time()
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=args.resolution)
    mesh = meshes[0]
    mesh.export(args.out)
    print(json.dumps({
        "ok": True,
        "out": args.out,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "seconds": round(time.time() - t0, 1),
        "model_load_s": round(t_model - t0, 1),
        "infer_s": round(t_codes - t_model, 1),
        "mesh_s": round(time.time() - t_codes, 1),
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
