"""build_object: request -> brief -> reference image -> mesh -> voxels -> SemanticGrid (+ preview, litematic).

Every stage writes its artefact under `out_dir/<label>_<stamp>/` so a bad result can be inspected: the
image the reconstructor saw, the mesh, the preview and the schematic. Timings per stage go in the report.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from craftpilot.config import SETTINGS
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.objects import imagegen, recon
from craftpilot.objects.brief import ObjectBrief, compose_brief
from craftpilot.objects.voxelize import (
    flatness,
    load_mesh,
    relief_like,
    to_grid,
    voxelize_mesh,
)


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


# A pancake mesh (thin along up, square footprint) means the view confused the reconstructor; see
# voxelize.relief_like. Plain flatness is not the test: a horse in profile is 0.31 and perfectly fine.


def build_object(text: str, out_dir: Path | None = None, height: int | None = None, use_llm: bool = True,
                 preview: bool = True, schematic: bool = True, resolution: int = 256, seed: int = 0,
                 yaw_deg: float = 0.0, max_types: int | None = None, max_recon_tries: int = 2,
                 on_stage: Callable[[str, str], None] | None = None, engine: str | None = None) -> ObjectResult:
    """The whole object path. Raises `recon.ReconUnavailable` when the local 3D worker is missing.
    `on_stage(name, text)` is called as each stage completes (brief, image, mesh, voxel) for live progress."""
    timings: dict = {}
    notes: list[str] = []

    def stage(name: str, msg: str) -> None:
        if on_stage is not None:
            on_stage(name, msg)

    t = time.time()
    brief, meta = compose_brief(text, use_llm=use_llm, height=height)
    timings["brief_s"] = round(time.time() - t, 1)
    if meta.get("error"):
        notes.append(f"brief: LLM failed ({meta['error']}); used the keyword fallback")
    stage("brief", f"{brief.label}: {brief.subject} ({brief.height} tall, {brief.palette}{', plinth' if brief.plinth else ''})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = (out_dir or SETTINGS.schematics_dir / "objects") / f"{brief.label}_{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "brief.json").write_text(_json(brief.to_dict()))

    # image -> mesh, with a degeneracy gate: a relief-like mesh (flatness < MIN_FLATNESS) means the view
    # confused the reconstructor; try again with the next camera phrasing and keep the best of the tries.
    best: tuple[tuple[int, float], Path, Path, dict] | None = None
    timings["image_s"] = 0.0
    timings["mesh_s"] = 0.0
    for cam in range(max_recon_tries):
        t = time.time()
        attempts = imagegen.prompt_attempts(brief.subject, brief.plinth, brief.style, request=text, camera=cam,
                                            vivid=brief.palette == "colorful")
        img_path = work / ("reference.png" if cam == 0 else f"reference_{cam}.png")
        mesh_path = work / ("mesh.ply" if cam == 0 else f"mesh_{cam}.ply")
        img = imagegen.generate(attempts[0], img_path, attempts=attempts[1:])
        timings["image_s"] = round(timings["image_s"] + img["seconds"], 1)
        stage("image", "reference image drawn" + (" (retry with another camera)" if cam else ""))
        if cam == 0:
            (work / "prompt.txt").write_text(img["prompt"])
            if img.get("attempt"):
                notes.append(f"image: prompt {img['attempt']} of {len(attempts)} passed the content filter (earlier ones were rejected)")
        t = time.time()
        rep = recon.reconstruct(img_path, mesh_path, resolution=resolution, engine=engine)
        timings["mesh_s"] = round(timings["mesh_s"] + rep["wall_seconds"], 1)
        mesh_obj = load_mesh(mesh_path)
        flat = flatness(mesh_obj)
        up_axis = str(rep.get("up_axis", "z"))
        relief = relief_like(mesh_obj, up_axis="xyz".index(up_axis[-1]))
        notes.append(f"mesh{'' if cam == 0 else ' ' + str(cam)}: {rep.get('engine', 'triposr')}, {rep.get('vertices')} vertices, infer {rep.get('infer_s')}s, flatness {flat:.2f}" + (" relief" if relief else ""))
        score = (0 if relief else 1, flat)
        if best is None or score > best[0]:
            best = (score, img_path, mesh_path, rep)
        stage("mesh", f"{rep.get('faces', '?')} faces reconstructed" + (" — came out flat, trying another view" if relief else ""))
        if not relief:
            break
    assert best is not None
    if best[0][0] == 0:
        notes.append("warning: every reconstruction came out as a flat relief; try rephrasing (a compact pose helps)")
    image_path, mesh_path, rep = best[1], best[2], best[3]
    up_axis = str(rep.get("up_axis", "z"))
    front_axis = str(rep.get("front_axis", "+x"))

    t = time.time()
    obj = voxelize_mesh(load_mesh(mesh_path), height=brief.height, yaw_deg=yaw_deg, up=up_axis)
    grid = to_grid(obj, allowed=brief.allowed_blocks, seed=seed, max_types=max_types)
    grid.report["object_front"] = front_axis  # which side faced the camera; placement turns it toward the player
    timings["voxel_s"] = round(time.time() - t, 1)
    notes.extend(grid.report.get("notes", []))
    stage("voxel", f"{obj.size[0]}x{obj.size[1]}x{obj.size[2]}, {obj.block_count} blocks, {len(grid.palette)} block types")

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
