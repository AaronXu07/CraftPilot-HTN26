"""Object requests (statues, creatures, vehicles, props) go to the image-to-3D path in `craftpilot.objects`
instead of the staged CAD pipeline: FLUX draws the subject, TripoSR reconstructs it, the voxels come back
as a block map that `placement.place_scene` animates into the world with the usual `/cp undo` bookkeeping.

Routing (see orchestrator.run_build): a request whose keyword guess is "object" comes here directly; one
the interpret stage classifies as an object comes here after the brief. Anything that makes the path
unavailable (package not installed, 3D worker missing, image rejected) falls back to the staged pipeline
with one chat line saying why, so the player always gets a build.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from ..jobs import check_cancel
from ..placement import place_scene
from .progress import Progress

PROGRESS_TAGS = {"brief": "plan", "image": "image", "mesh": "mesh", "voxel": "voxel"}


class ObjectPathUnavailable(RuntimeError):
    """The object path cannot run here; the caller should fall back to the staged pipeline."""


def enabled() -> bool:
    return os.environ.get("COPILOT_OBJECT_PATH", "1").lower() not in ("0", "false", "off", "no")


def available() -> Tuple[bool, str]:
    """(ok, reason): the craftpilot package imports and the local 3D worker exists."""
    if not enabled():
        return False, "COPILOT_OBJECT_PATH is off"
    try:
        from craftpilot.objects import recon
    except ImportError as e:  # the agent can run without the building package installed
        return False, f"craftpilot.objects not importable ({e})"
    if not recon.available():
        return False, "3D worker not installed (see tools/README.md)"
    return True, ""


def grid_to_block_map(grid: Any) -> Dict[Tuple[int, int, int], str]:
    """SemanticGrid -> scene-space BlockMap (centred, front on +z) for placement.place_scene."""
    from craftpilot.objects.place import grid_blocks

    return {(x, y, z): s for x, y, z, s in grid_blocks(grid)}


def run_object_build(ctx: Any, request: str, place: bool = True, height: Optional[int] = None) -> Dict[str, Any]:
    """Build and (optionally) place an object. Returns a dict for the ChatResult: reply, brief, placed, data.
    Raises ObjectPathUnavailable when the path cannot produce a result (caller falls back)."""
    ok, why = available()
    if not ok:
        raise ObjectPathUnavailable(why)
    from craftpilot.objects.imagegen import ImageRejected
    from craftpilot.objects.pipeline import build_object
    from craftpilot.objects.recon import ReconUnavailable

    session = ctx.session
    progress: Progress = getattr(ctx, "progress", None) or Progress(ctx)
    t0 = time.time()

    def on_stage(name: str, text: str) -> None:
        check_cancel(ctx)
        progress.say(PROGRESS_TAGS.get(name, name), text)

    check_cancel(ctx)
    try:
        result = build_object(request, height=height, preview=False, schematic=True, on_stage=on_stage)
    except ReconUnavailable as e:
        raise ObjectPathUnavailable(str(e)) from e
    except ImageRejected as e:
        raise ObjectPathUnavailable(str(e)) from e
    brief = result.brief
    session.brief = {"kind": "object", "name": brief.label, "build_type": brief.label.replace("_", " "), "style": brief.style,
                     "footprint": [result.size[0], result.size[2]], "height": result.size[1], "facing": "south",
                     "key_features": [], "silhouette_plan": brief.subject, "stages": [], "request": request,
                     "object": result.to_dict()}
    session.scene.name = brief.label
    session.scene.meta["brief"] = {k: v for k, v in session.brief.items() if k not in ("request", "object")}

    placed = False
    place_text = ""
    if place:
        check_cancel(ctx)
        bridge = getattr(ctx, "bridge", None)
        if bridge is None:
            place_text = "no bridge on the context"
        else:
            if session.world.is_placed():
                # a new object is a new build: it gets its own spot in front of the player instead of
                # reusing the last anchor (which, in "full" mode, would erase the previous object)
                from ..session import WorldState

                session.snapshot(f"before_{brief.label}_{session.turn}")
                session.world = WorldState()
            block_map = grid_to_block_map(result.grid)

            def on_progress(pct: int) -> None:
                progress.say("build", pct=pct)

            place_text = place_scene(session, bridge, block_map, mode="full", animate=True, on_progress=on_progress)
            placed = not place_text.startswith("ERROR") and "nothing to place" not in place_text
    secs = time.time() - t0
    mins, s = divmod(int(secs), 60)
    lines = [
        f"Built {brief.label}: {brief.subject}",
        f"{result.size[0]}×{result.size[2]} footprint, {result.size[1]} tall, {result.blocks:,} blocks, "
        f"{len(result.grid.palette)} block types, in {mins}:{s:02d}"
        + (f" ({', '.join(n for n in result.notes if n.startswith('warning'))})" if any(n.startswith("warning") for n in result.notes) else ""),
    ]
    if result.litematic:
        lines.append(f"Schematic: {result.litematic.name} (Litematica → Load Schematics → craftpilot)")
    lines.append("Say `undo` to remove it, or describe another object.")
    data = {"kind": "object", "object": result.to_dict(), "place": place_text, "seconds": round(secs, 1)}
    return {"reply": "\n".join(lines), "brief": session.brief, "placed": placed, "data": data}
