"""build_object: request -> brief -> reference image -> mesh -> voxels -> SemanticGrid (+ preview, litematic).

Every stage writes its artefact under `out_dir/<label>_<stamp>/` so a bad result can be inspected: the
image the reconstructor saw, the mesh, the preview and the schematic. Timings per stage go in the report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from craftpilot.config import SETTINGS
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.objects import imagegen, recon
from craftpilot.objects.brief import ObjectBrief, compose_brief
from craftpilot.objects.voxelize import flatness, load_mesh, to_grid, voxelize_mesh


@dataclass
class ObjectResult:
    brief: ObjectBrief
    grid: SemanticGrid
    work_dir: Path
    image: Path
    mesh: Path
    preview: Path | None = None
    litematic: Path | None = None
    blocks: int = 0
    size: tuple[int, int, int] = (0, 0, 0)
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return round(sum(v for v in self.timings.values() if isinstance(v, (int, float))), 1)

    def to_dict(self) -> dict:
        return {
            "brief": self.brief.to_dict(), "work_dir": str(self.work_dir), "image": str(self.image), "mesh": str(self.mesh),
            "preview": str(self.preview) if self.preview else None, "litematic": str(self.litematic) if self.litematic else None,
            "blocks": self.blocks, "size": list(self.size), "timings": self.timings, "seconds": self.seconds, "notes": self.notes,
        }


MIN_FLATNESS = 0.32  # thinnest/largest extent below this = relief, not a body (a car is ~0.4, a statue ~0.5+)


def build_object(text: str, out_dir: Path | None = None, height: int | None = None, use_llm: bool = True,
                 preview: bool = True, schematic: bool = True, resolution: int = 256, seed: int = 0,
                 yaw_deg: float = 0.0, max_types: int = 6, max_recon_tries: int = 2) -> ObjectResult:
    """The whole object path. Raises `recon.ReconUnavailable` when the local 3D worker is missing."""
    timings: dict = {}
    notes: list[str] = []
    t = time.time()
    brief, meta = compose_brief(text, use_llm=use_llm, height=height)
    timings["brief_s"] = round(time.time() - t, 1)
    if meta.get("error"):
        notes.append(f"brief: LLM failed ({meta['error']}); used the keyword fallback")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = (out_dir or SETTINGS.schematics_dir / "objects") / f"{brief.label}_{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "brief.json").write_text(_json(brief.to_dict()))

    # image -> mesh, with a degeneracy gate: a relief-like mesh (flatness < MIN_FLATNESS) means the view
    # confused the reconstructor; try again with the next camera phrasing and keep the best of the tries.
    best: tuple[float, Path, Path] | None = None
    timings["image_s"] = 0.0
    timings["mesh_s"] = 0.0
    for cam in range(max_recon_tries):
        t = time.time()
        attempts = imagegen.prompt_attempts(brief.subject, brief.plinth, brief.style, request=text, camera=cam)
        img_path = work / ("reference.png" if cam == 0 else f"reference_{cam}.png")
        mesh_path = work / ("mesh.ply" if cam == 0 else f"mesh_{cam}.ply")
        img = imagegen.generate(attempts[0], img_path, attempts=attempts[1:])
        timings["image_s"] = round(timings["image_s"] + img["seconds"], 1)
        if cam == 0:
            (work / "prompt.txt").write_text(img["prompt"])
            if img.get("attempt"):
                notes.append(f"image: prompt {img['attempt']} of {len(attempts)} passed the content filter (earlier ones were rejected)")
        t = time.time()
        rep = recon.reconstruct(img_path, mesh_path, resolution=resolution)
        timings["mesh_s"] = round(timings["mesh_s"] + rep["wall_seconds"], 1)
        flat = flatness(load_mesh(mesh_path))
        notes.append(f"mesh{'' if cam == 0 else ' ' + str(cam)}: {rep.get('vertices')} vertices, infer {rep.get('infer_s')}s, flatness {flat:.2f}")
        if best is None or flat > best[0]:
            best = (flat, img_path, mesh_path)
        if flat >= MIN_FLATNESS:
            break
    assert best is not None
    if best[0] < MIN_FLATNESS:
        notes.append(f"warning: every reconstruction came out flat (best {best[0]:.2f}); the result is a relief, try rephrasing")
    image_path, mesh_path = best[1], best[2]

    t = time.time()
    obj = voxelize_mesh(load_mesh(mesh_path), height=brief.height, yaw_deg=yaw_deg)
    grid = to_grid(obj, allowed=brief.allowed_blocks, seed=seed, max_types=max_types)
    timings["voxel_s"] = round(time.time() - t, 1)
    notes.extend(grid.report.get("notes", []))

    result = ObjectResult(brief=brief, grid=grid, work_dir=work, image=image_path, mesh=mesh_path,
                          blocks=obj.block_count, size=obj.size, timings=timings, notes=notes)
    if preview:
        from craftpilot.preview.render import render

        t = time.time()
        result.preview = render(grid, work / "preview.png")
        timings["preview_s"] = round(time.time() - t, 1)
    if schematic:
        from craftpilot.export import litematic

        t = time.time()
        path = SETTINGS.schematics_dir / f"{brief.label}_{stamp}.litematic"
        result.litematic = litematic.save(grid, path, brief.label, "craftpilot", text, SETTINGS.mc_data_version)
        timings["export_s"] = round(time.time() - t, 1)
    (work / "report.json").write_text(_json(result.to_dict()))
    return result


def _json(d: dict) -> str:
    import json

    return json.dumps(d, indent=1, default=str)
