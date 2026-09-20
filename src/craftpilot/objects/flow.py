"""The objects path as the service and the CLI drive it: draw, describe, hologram, place.

`draw()` runs objects.pipeline.build_object with chat-friendly error mapping and hands back an ObjectResult
whose grid faces south, so `place_object()` can rotate a copy toward the player and hand it to the same
`place/placer.place_grid` a building uses (terrain seating, outline, one /setblocks). `ghost_payload()` and
`result_payload()` shape the answers exactly like the building handlers in serve.py, so the Fabric mod needs
no new fields. Imports of the heavy object modules (trimesh, scipy) happen inside the functions.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from craftpilot.config import SETTINGS

if TYPE_CHECKING:
    from craftpilot.objects.brief import ObjectBrief
    from craftpilot.objects.pipeline import ObjectResult
    from craftpilot.place.placer import PlaceContext


class ObjectError(RuntimeError):
    """The objects path could not produce a result; `status` is the HTTP status the service answers with."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def unavailable_reason() -> str | None:
    """Why the objects path cannot run on this machine at all (before spending anything), or None."""
    from craftpilot.objects import recon

    if not SETTINGS.azure_configured:
        return "image generation needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
    if not recon.available():
        return (f"3D worker not installed: expected {recon.worker_python()} and {recon.WORKER.name} "
                "(see tools/README.md)")
    return None


def draw(text: str, *, brief: ObjectBrief | None = None, seed: int = 0, use_llm: bool = True,
         height: int | None = None, say: Callable[[str], None] | None = None, preview: bool = True,
         out_dir: Path | None = None, engine: str | None = None, max_types: int | None = None,
         resolution: int = 256) -> ObjectResult:
    """Text (or a precomposed brief) -> ObjectResult with a south-facing grid. Raises ObjectError."""
    reason = unavailable_reason()
    if reason:
        raise ObjectError(503, reason + " - or /build building <text> for the procedural generator")
    from craftpilot.objects.imagegen import ImageRejected
    from craftpilot.objects.orient import face_south
    from craftpilot.objects.pipeline import build_object
    from craftpilot.objects.recon import ReconUnavailable

    last = {"stage": "brief"}

    def on_stage(name: str, msg: str) -> None:
        last["stage"] = name
        if say is not None:
            say(f"object · {name}: {msg}")

    try:
        result = build_object(text, out_dir=out_dir, height=height, use_llm=use_llm, preview=preview, schematic=True,
                              seed=seed, max_types=max_types, resolution=resolution, on_stage=on_stage, engine=engine,
                              brief=brief)
    except ReconUnavailable as exc:
        raise ObjectError(503, f"3D worker not available: {exc} (see tools/README.md) - or /build building <text> "
                               "for the procedural generator") from exc
    except ImageRejected as exc:
        if getattr(exc, "kind", "") == "blocklist":
            raise ObjectError(422, "the image service refuses that name (protected or trademarked); describe the look "
                                   f"instead, e.g. 'a chubby yellow mouse-like creature with red cheeks' ({exc})") from exc
        raise ObjectError(422, f"the image was rejected by Azure's content filter: {exc}; try a museum-replica "
                               "phrasing") from exc
    except ObjectError:
        raise
    except Exception as exc:  # noqa: BLE001 - one chat line, not a traceback in the game
        raise ObjectError(500, f"object pipeline failed at {last['stage']}: {type(exc).__name__}: {str(exc)[:160]}") from exc
    face_south(result.grid)  # idempotent; keeps a monkeypatched build_object honest too
    result.size = (result.grid.W, result.grid.H, result.grid.D)
    return result


def describe_brief(brief: ObjectBrief, grid: Any | None = None) -> list[str]:
    """Chat lines for a brief, the object-path counterpart of program/describe.describe()."""
    if grid is not None:
        size = f"{grid.W} wide x {grid.H} tall x {grid.D} deep"
    else:
        size = f"about {brief.height} blocks in its largest dimension"
    lines = [
        f"{brief.label.replace('_', ' ')} - {size} (object: reference image -> 3D mesh -> voxels)",
        f"subject: {brief.subject}",
    ]
    if brief.style:
        lines.append(f"style: {brief.style}")
    lines.append(f"palette: {brief.palette}" + (f" ({len(grid.palette)} block types)" if grid is not None else ""))
    lines.append("on a plinth" if brief.plinth else "no plinth (stands on the ground)")
    lines.append("brief via " + ("the model" if brief.source == "llm" else "keywords (no model)"))
    return lines


def ghost_payload(result: ObjectResult, notes: list[str], source: str = "object", seed: int | None = None) -> dict:
    """The hologram answer the mod's armGhost() expects; `seed` is the token it echoes back on commit."""
    from craftpilot.preview.ghost import ghost_cloud

    grid = result.grid
    flat, height, n = ghost_cloud(grid)
    token = result.seed if seed is None else seed
    return {
        "kind": "object",
        "ghost": {"width": grid.W, "height": height, "depth": grid.D, "blocks": flat},
        "seed": token,
        "bounds": {"width": grid.W, "height": height, "depth": grid.D},
        "blocks": n,
        "generate_seconds": result.seconds,
        "plan": describe_brief(result.brief, grid),
        "source": source,
        "notes": list(notes),
        "summary": f"{result.brief.label} #{token}: {n} blocks, {grid.W}x{height}x{grid.D} - preview ready "
                   f"({result.seconds:.0f} s)",
        "work_dir": str(result.work_dir),
        "preview": str(result.preview) if result.preview else None,
        "schematic": str(result.litematic) if result.litematic else None,
        "timings": result.timings,
    }


def result_payload(result: ObjectResult, facing: str, source: str, notes: list[str], seed: int | None = None) -> dict:
    """A run_build-shaped result for an object (what serve._summary and the mod's report() read)."""
    grid = result.grid
    return {
        "kind": "object",
        "schematic": str(result.litematic) if result.litematic else None,
        "program": None,
        "work_dir": str(result.work_dir),
        "preview": str(result.preview) if result.preview else None,
        "image": str(result.image),
        "blocks": grid.block_count(),
        "bounds": (grid.W, grid.H, grid.D),
        "seed": result.seed if seed is None else seed,
        "facing": facing,
        "generate_seconds": result.seconds,
        "export_seconds": float(result.timings.get("export_s", 0.0)),
        "notes": list(notes),
        "attachments": [],
        "source": source,
        "timings": result.timings,
    }


def place_object(result: ObjectResult, ctx: PlaceContext, facing: str, undo_path: Path | None = None) -> dict:
    """Rotate a copy of the cached south-facing grid toward the player and queue it through the terrain placer.
    Returns place_grid's placement dict. Raises ValueError (cannot fit on the site) and BridgeError."""
    from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
    from craftpilot.place.placer import place_grid

    grid = copy.deepcopy(result.grid)  # rotate_cw mutates in place; the cache stays south-facing
    if facing in ("north", "east", "west"):
        rotate_cw(grid, quarter_turns_for_facing(facing))
    # A plinth stands like a foundation ring; a creature on legs gets a level pad under its whole footprint.
    ctx.opts.base = "row0" if result.brief.plinth else "footprint"
    if undo_path is None:
        undo_path = Path(result.work_dir) / "placed.json"
    return place_grid(grid, ctx, result.brief.label, undo_path=undo_path)
