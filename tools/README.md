# tools/ — the local 3D reconstruction worker

The object path (`craftpilot object "…"`, `POST /object`) turns a FLUX reference image into a mesh with
**TripoSR** (Stability AI, MIT) running locally. It lives in its own virtualenv so torch never touches the
main environment; `craftpilot.objects.recon` calls `recon_worker.py` as a subprocess.

## One-time setup (Apple Silicon, ~10 min, ~2.5 GB)

```sh
cd tools
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git triposr
uv venv --python 3.11 .venv-3d
source .venv-3d/bin/activate
uv pip install torch torchvision
uv pip install omegaconf einops "transformers>=4.35,<5" trimesh "rembg[cpu]" huggingface-hub pillow numpy scipy
uv pip install scikit-build-core cmake ninja setuptools wheel pybind11
export CMAKE_PREFIX_PATH="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())'):$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
uv pip install --no-build-isolation "git+https://github.com/tatsy/torchmcubes.git"
```

Then patch TripoSR's marching cubes for MPS (torchmcubes only accepts CPU tensors there) — in
`triposr/tsr/models/isosurface.py`, catch `RuntimeError` as well as `AttributeError` around the
`self.mc_func(level.detach(), 0.0)` call and fall back to `level.detach().cpu()`.

Smoke test (downloads the 1.6 GB weights to the Hugging Face cache on first run):

```sh
python recon_worker.py triposr/examples/chair.png out_smoke/chair.ply
# {"ok": true, "vertices": …, "seconds": 12.4, "model_load_s": 5.9, "infer_s": 2.9, "mesh_s": 3.6}
```

Timings on an M4 Pro: model load ~5 s (per call for now), inference 3–11 s, meshing ~4 s.

## The persistent worker (what actually runs)

`recon_server.py` loads TripoSR once, warms the GPU kernels and the u2net matting model, and serves
`POST /reconstruct` on `127.0.0.1:7790`. `craftpilot.objects.recon` starts it automatically on the first
object and falls back to `recon_worker.py` (one-shot subprocess) if it cannot. Per object on an M4 Pro:
matte 0.2 s, inference 0.6 s, meshing 1.4 s. Log: `tools/recon_server.log`. Stop it with
`pkill -f recon_server.py`; set `CRAFTPILOT_RECON_SERVER=0` to force the subprocess path.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `CRAFTPILOT_RECON_PYTHON` | `tools/.venv-3d/bin/python` | interpreter that has torch + TripoSR |
| `CRAFTPILOT_IMAGE_DEPLOYMENT` | `FLUX-1.1-pro` | Azure AI Foundry image deployment (needs quota; capacity 30 recommended) |
| `CRAFTPILOT_OBJECT_DEPLOYMENT` | falls back to `AZURE_OPENAI_DEPLOYMENT` | LLM that writes the brief (gpt-5.4-mini is plenty) |

`recon_worker.py` prints one JSON line; anything else it writes goes to stderr. If the worker is missing the
object path fails fast with `ReconUnavailable` (HTTP 503 from `/object`) rather than hanging.
