"""Command line interface."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from craftpilot.config import SETTINGS
from craftpilot.program.model import Bounds, BuildProgram

if TYPE_CHECKING:
    from craftpilot.place.placer import PlaceContext

app = typer.Typer(add_completion=False, help="Procedural Minecraft buildings from natural language.")


def _parse_bounds(text: str | None) -> Bounds | None:
    if not text:
        return None
    parts = [int(p) for p in re.split(r"[x, ]+", text.strip().lower()) if p]
    if len(parts) != 3:
        raise typer.BadParameter("bounds must look like 25x30x25 (width x height x depth)")
    return Bounds(width=parts[0], height=parts[1], depth=parts[2])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "build"


def run_build(program: BuildProgram, bounds: Bounds | None, seed: int, out: Path | None, preview: bool,
              author: str, source_text: str, extra_notes: list[str], facing: str = "south",
              place_ctx: PlaceContext | None = None) -> dict:
    """Generate, write the schematic, and (with ``place_ctx``) stream the blocks into the running game."""
    from craftpilot.engine.pipeline import generate
    from craftpilot.export.litematic import save
    from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
    from craftpilot.program.validate import clamp_bounds, repair

    program, notes = repair(program, SETTINGS.safety_limit)
    notes = extra_notes + notes
    if bounds is None:
        bounds = program.bounds
    bounds, err = clamp_bounds(bounds, SETTINGS.safety_limit)
    if err:
        raise typer.BadParameter(err)
    if place_ctx is not None:
        notes += _outline_bounds(place_ctx, bounds, facing)
    t0 = time.time()
    grid = generate(program, bounds, seed)
    if facing in ("north", "east", "west"):
        rotate_cw(grid, quarter_turns_for_facing(facing))
    gen_s = time.time() - t0
    label = _slug(program.label)
    if out is None:
        out = SETTINGS.schematics_dir / f"{label}_{seed}_{int(time.time()) % 100000}.litematic"
    desc = json.dumps({"text": source_text, "seed": seed, "bounds": bounds.as_tuple(), "label": program.label})
    t1 = time.time()
    save(grid, out, name=f"{program.label} #{seed}", author=author, description=desc,
         mc_version=SETTINGS.mc_data_version)
    program_path = out.with_suffix(".program.json")
    program_path.write_text(program.model_dump_json(indent=2))
    export_s = time.time() - t1
    result = {
        "schematic": str(out),
        "program": str(program_path),
        "blocks": grid.block_count(),
        "bounds": bounds.as_tuple(),
        "seed": seed,
        "facing": facing,
        "generate_seconds": round(gen_s, 3),
        "export_seconds": round(export_s, 3),
        "notes": notes + grid.report["notes"],
        "attachments": [s for s in grid.report["stages"] if s["stage"].startswith("attachments")],
    }
    if place_ctx is not None:
        from craftpilot.place.placer import place_grid

        placement = place_grid(grid, place_ctx, program.label)
        result["placement"] = placement
        result["notes"] = result["notes"] + placement["warnings"]
    if preview:
        from craftpilot.preview.render import render
        png = out.with_suffix(".png")
        render(grid, png)
        result["preview"] = str(png)
    return result


def _outline_bounds(place_ctx: PlaceContext, bounds: Bounds, facing: str) -> list[str]:
    """Show the box the build will occupy as soon as the bounds are known, before generation.

    The engine builds facing south and is rotated afterwards, so an east or west facing swaps
    width and depth. The trimmed box replaces this one when the blocks are queued."""
    from craftpilot.place.anchor import plan_origin
    from craftpilot.place.placer import show_outline

    w, h, d = bounds.width, bounds.height, bounds.depth
    if facing in ("east", "west"):
        w, d = d, w
    bbox = (0, 0, 0, w - 1, h - 1, d - 1)
    ox, oy, oz = plan_origin(place_ctx.player, bbox, gap=place_ctx.opts.gap, sink=place_ctx.opts.sink)
    return show_outline(place_ctx.bridge, (ox, oy, oz), (ox + w - 1, oy + h - 1, oz + d - 1), "generating")


@app.command()
def build(
    text: str = typer.Argument(..., help="What to build, in plain language"),
    bounds: str = typer.Option(None, "--bounds", "-b", help="Bounding box WxHxD, overrides the program's suggestion"),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed (default: time based)"),
    out: Path = typer.Option(None, "--out", "-o", help="Output .litematic path"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Also write an isometric PNG next to the schematic"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Azure OpenAI and use the nearest exemplar"),
    exemplar: str = typer.Option(None, "--exemplar", "-e", help="Use this exemplar's program directly"),
    author: str = typer.Option("craftpilot", "--author"),
    facing: str = typer.Option("south", "--facing", "-f", help="Which way the front door faces: north, east, south, west"),
    show_program: bool = typer.Option(False, "--show-program", help="Print the composed program"),
    place: bool = typer.Option(False, "--place", "-P", help="Also place the blocks in the running game via the mod"),
    gap: int = typer.Option(None, "--gap", help="Air blocks between you and the building (place only)"),
    sink: int = typer.Option(None, "--sink", help="Blocks the plinth row sinks below your feet (place only)"),
    no_clear: bool = typer.Option(False, "--no-clear", help="Do not clear terrain inside the bounding box"),
    delay_ms: int = typer.Option(None, "--delay-ms", help="Pause between placed chunks (place only)"),
    chunk: int = typer.Option(None, "--chunk", help="Blocks per game tick (place only)"),
    flags: int = typer.Option(None, "--flags", help="setBlockState flags; default 18 = force state, no updates"),
    kind: str = typer.Option("auto", "--kind", "-k", help="auto (route the text), building, or object"),
    height: int = typer.Option(None, "--height", "-H", help="Objects: largest dimension in blocks"),
) -> None:
    """Build TEXT: the router sends architecture to the procedural generator and statues, creatures, vehicles
    and named landmarks to the image->3D objects path; write a Litematica schematic either way."""
    from craftpilot.llm.compose import compose
    from craftpilot.program.exemplars import load_one
    from craftpilot.route import classify

    if kind not in ("auto", "building", "object"):
        raise typer.BadParameter("kind must be auto, building, or object")
    b = _parse_bounds(bounds)
    if exemplar:
        kind = "building"
    r = classify(text, override=None if kind == "auto" else kind, use_llm=not no_llm, has_bounds=b is not None)
    typer.echo(r.note, err=True)
    if r.kind == "object":
        _object_build(text, height=height, out_dir=None, use_llm=not no_llm, preview=preview, seed=seed, place=place,
                      gap=gap, sink=sink, clear=not no_clear, delay_ms=delay_ms, chunk=chunk, flags=flags, facing=facing)
        return

    place_ctx = None
    if place:
        place_ctx = _place_context(gap, sink, not no_clear, delay_ms, chunk, flags)
        from craftpilot.place.anchor import facing_from_yaw

        facing = facing_from_yaw(place_ctx.player["yaw"])
        typer.echo(f"placing in front of {place_ctx.player['name']}, building faces {facing}", err=True)

    if seed is None:
        seed = int(time.time()) % 1_000_000
    notes: list[str] = [r.note]
    t_start = time.time()
    if exemplar:
        ex = load_one(SETTINGS.exemplars_dir, exemplar)
        if ex is None:
            raise typer.BadParameter(f"No exemplar named '{exemplar}'")
        program = ex.program
        source = "compose:exemplar"
    else:
        program, source, notes = compose(text, use_llm=not no_llm, bounds_hint=b)
    if show_program:
        typer.echo(program.model_dump_json(indent=2))
    if facing not in ("north", "east", "south", "west"):
        raise typer.BadParameter("facing must be north, east, south, or west")
    compose_s = time.time() - t_start
    result = run_build(program, b, seed, out, preview, author, text, notes, facing, place_ctx)
    result["source"] = source
    result["compose_seconds"] = round(compose_s, 3)
    result["total_seconds"] = round(time.time() - t_start, 3)
    typer.echo(json.dumps(result, indent=2))
    if source == "fallback":
        typer.echo("WARNING: the model was not used; this build came from the offline fallback. " + " ".join(notes),
                   err=True)
    typer.echo(f"Built in {result['total_seconds']:.1f}s: compose {compose_s:.1f}s, generate {result['generate_seconds']:.2f}s, "
               f"export {result['export_seconds']:.2f}s", err=True)


def _place_context(gap: int | None, sink: int | None, clear: bool, delay_ms: int | None, chunk: int | None,
                   flags: int | None):
    from craftpilot.place.bridge import BridgeError, HttpBridge
    from craftpilot.place.placer import PlaceContext, PlaceOptions

    bridge = HttpBridge()
    try:
        player = bridge.player()
    except BridgeError as exc:
        raise typer.BadParameter(
            f"mod not reachable at {bridge.base_url} ({exc}); is Minecraft running with the CraftPilot mod "
            "and a world open?") from exc
    opts = PlaceOptions(clear=clear)
    if gap is not None:
        opts.gap = gap
    if sink is not None:
        opts.sink = sink
    if delay_ms is not None:
        opts.delay_ms = delay_ms
    if chunk is not None:
        opts.chunk_blocks = chunk
    if flags is not None:
        opts.flags = flags
    return PlaceContext(bridge=bridge, player=player, opts=opts)


@app.command()
def plan(
    text: str = typer.Argument(..., help="What to build, in plain language"),
    bounds: str = typer.Option(None, "--bounds", "-b", help="Bounding box WxHxD"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Azure OpenAI and use the nearest exemplar"),
    kind: str = typer.Option("auto", "--kind", "-k", help="auto (route the text), building, or object"),
) -> None:
    """Describe what TEXT would build (a building program, or an object brief), without generating anything."""
    from craftpilot.llm.compose import compose
    from craftpilot.program.describe import describe
    from craftpilot.program.validate import repair
    from craftpilot.route import classify

    b = _parse_bounds(bounds)
    r = classify(text, override=None if kind == "auto" else kind, use_llm=not no_llm, has_bounds=b is not None)
    typer.echo(r.note, err=True)
    if r.kind == "object":
        from craftpilot.objects.brief import compose_brief
        from craftpilot.objects.flow import describe_brief

        brief, meta = compose_brief(text, use_llm=not no_llm)
        for line in describe_brief(brief):
            typer.echo(line)
        if meta.get("error"):
            typer.echo(f"  note: brief: LLM failed ({meta['error']}); used the keyword fallback", err=True)
        typer.echo(f"  source: object:{brief.source}", err=True)
        return
    program, source, notes = compose(text, use_llm=not no_llm, bounds_hint=_parse_bounds(bounds))
    program, r_notes = repair(program, SETTINGS.safety_limit)
    for line in describe(program, _parse_bounds(bounds) or program.bounds):
        typer.echo(line)
    for note in notes + r_notes:
        typer.echo(f"  note: {note}", err=True)
    typer.echo(f"  source: {source}", err=True)


@app.command("place-status")
def place_status() -> None:
    """Show the mod's health and placement queue."""
    from craftpilot.place.bridge import HttpBridge

    bridge = HttpBridge()
    typer.echo(json.dumps({"health": bridge.health(), "status": bridge.status()}, indent=2))


@app.command("place-cancel")
def place_cancel() -> None:
    """Drop everything still queued for placement in the game."""
    from craftpilot.place.bridge import HttpBridge

    typer.echo(json.dumps(HttpBridge().cancel()))


@app.command()
def render(
    names: list[str] = typer.Argument(None, help="Exemplar names (default: all)"),
    out_dir: Path = typer.Option(Path("out/exemplars"), "--out-dir"),
    seed: int = typer.Option(1, "--seed"),
) -> None:
    """Generate exemplars and write previews plus schematics into OUT_DIR."""
    from craftpilot.engine.pipeline import generate
    from craftpilot.export.litematic import save
    from craftpilot.preview.render import render as render_png
    from craftpilot.program.exemplars import load_all
    from craftpilot.program.validate import repair

    exs = load_all(SETTINGS.exemplars_dir)
    if names:
        exs = [e for e in exs if e.name in names]
    for e in exs:
        program, notes = repair(e.program, SETTINGS.safety_limit)
        grid = generate(program, program.bounds, seed)
        png = render_png(grid, out_dir / f"{e.name}.png")
        save(grid, out_dir / f"{e.name}.litematic", e.name, "craftpilot", e.description, SETTINGS.mc_data_version)
        typer.echo(f"{e.name}: {grid.block_count()} blocks -> {png}  {notes + grid.report['notes']}")


@app.command()
def serve(port: int = typer.Option(None, "--port")) -> None:
    """Run the local HTTP service used by the Fabric mod."""
    import uvicorn

    from craftpilot.serve import app as fastapi_app

    uvicorn.run(fastapi_app, host="127.0.0.1", port=port or SETTINGS.service_port, log_level="info")


def _object_build(text: str, *, height: int | None, out_dir: Path | None, use_llm: bool, preview: bool,
                  seed: int | None, place: bool, gap: int | None = None, sink: int | None = None, clear: bool = True,
                  delay_ms: int | None = None, chunk: int | None = None, flags: int | None = None,
                  facing: str = "south", types: int | None = None, resolution: int = 256,
                  engine: str | None = None) -> None:
    """The objects path from the CLI: draw, report, and with ``place`` seat it in front of the player through the
    same terrain placer a building uses (the mod's /heightmap survey, outline, one /setblocks)."""
    from craftpilot.objects import flow

    place_ctx = None
    if place:
        place_ctx = _place_context(gap, sink, clear, delay_ms, chunk, flags)
        from craftpilot.place.anchor import facing_from_yaw

        facing = facing_from_yaw(place_ctx.player["yaw"])
        typer.echo(f"placing in front of {place_ctx.player['name']}, object faces {facing}", err=True)
    try:
        r = flow.draw(text, height=height, out_dir=out_dir, use_llm=use_llm, preview=preview,
                      seed=seed if seed is not None else int(time.time()) % 1_000_000,
                      say=lambda line: typer.echo(f"  {line}", err=True), max_types=types, resolution=resolution,
                      engine=engine)
    except flow.ObjectError as exc:
        typer.echo(f"error: {exc.detail}", err=True)
        raise typer.Exit(2) from exc
    b = r.brief
    typer.echo(f"{b.label}: {b.subject}")
    typer.echo(f"  {r.size[0]}x{r.size[1]}x{r.size[2]}, {r.blocks} blocks, palette {b.palette}, plinth {b.plinth} "
               f"(brief via {b.source}, {r.engine or 'recon'})")
    typer.echo("  " + "  ".join(f"{k} {v}s" for k, v in r.timings.items()) + f"  total {r.seconds}s")
    for n in r.notes:
        typer.echo(f"  note: {n}")
    if r.preview:
        typer.echo(f"  preview   {r.preview}")
    if r.litematic:
        typer.echo(f"  litematic {r.litematic}")
    typer.echo(f"  work dir  {r.work_dir}")
    if place_ctx is not None:
        from craftpilot.place.bridge import BridgeError

        try:
            pl = flow.place_object(r, place_ctx, facing)
        except (ValueError, BridgeError) as exc:
            typer.echo(f"  not placed: {exc}", err=True)
            raise typer.Exit(2) from exc
        ox, oy, oz = pl["origin"]
        typer.echo(f"  placed    {pl['blocks']} blocks in {pl['chunks']} layers at ({ox}, {oy}, {oz}), ~{pl['estimated_seconds']}s")
        for w in pl["warnings"]:
            typer.echo(f"  note: {w}")
        if pl.get("undo_file"):
            typer.echo(f"  undo      craftpilot object-undo \"{pl['undo_file']}\"")


@app.command("object")
def object_(
    text: str = typer.Argument(..., help="The object: a statue, creature, character, vehicle, landmark or prop"),
    height: int = typer.Option(None, "--height", "-H", help="Largest dimension in blocks (default: the brief's choice)"),
    out_dir: Path = typer.Option(None, "--out-dir", help="Where the working folder goes (default: <schematics>/objects)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM brief; use the request text as the image prompt"),
    no_preview: bool = typer.Option(False, "--no-preview"),
    seed: int = typer.Option(None, "--seed", "-s", help="Voxel palette seed (the image itself is never deterministic)"),
    types: int = typer.Option(None, "--types", help="Max distinct block types on the surface (default: 6 for grey subjects, 10 for colourful)"),
    resolution: int = typer.Option(256, "--resolution", help="Reconstruction marching-cubes resolution"),
    place_now: bool = typer.Option(False, "--place", "-P", help="Also place it in the running game, in front of the player (needs the Fabric mod)"),
    gap: int = typer.Option(None, "--gap", help="Air blocks between you and the object (place only)"),
    sink: int = typer.Option(None, "--sink", help="Blocks the base sinks below your feet (place only)"),
    no_clear: bool = typer.Option(False, "--no-clear", help="Do not clear terrain inside the bounding box"),
    engine: str = typer.Option(None, "--engine", help="Reconstructor: hunyuan (default; real 3D, ~13 s) or triposr (~2 s, shallow)"),
) -> None:
    """Text -> reference image (FLUX) -> mesh (Hunyuan3D / TripoSR, local) -> coloured voxels -> .litematic (+ preview).
    The same as `craftpilot build --kind object`."""
    _object_build(text, height=height, out_dir=out_dir, use_llm=not no_llm, preview=not no_preview, seed=seed,
                  place=place_now, gap=gap, sink=sink, clear=not no_clear, types=types, resolution=resolution,
                  engine=engine)


@app.command("object-undo")
def object_undo(undo_file: Path = typer.Argument(..., help="placed.json written next to a placed object's artefacts")) -> None:
    """Remove a placed object from the world (sets every recorded position to air)."""
    from craftpilot.place.bridge import BridgeError, HttpBridge
    from craftpilot.place.undo import undo

    try:
        n = undo(undo_file, HttpBridge())
    except BridgeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"removed {n} blocks")


@app.command()
def route(
    text: str = typer.Argument(..., help="A build request"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Word rules only; never ask the model"),
    as_json: bool = typer.Option(False, "--json", help="Print the full Route as JSON"),
) -> None:
    """Show which path a request takes (building generator or image->3D objects) and why."""
    from craftpilot.route import classify

    r = classify(text, use_llm=False if no_llm else None)
    if as_json:
        typer.echo(json.dumps(r.to_dict(), indent=2))
        return
    typer.echo(f"{r.kind}  ({r.source}, confidence {r.confidence:.2f})")
    typer.echo(f"  reason:  {r.reason}")
    typer.echo(f"  matched: {', '.join(r.matched) or '-'}")
    typer.echo(f"  scores:  building {r.scores[0]}, object {r.scores[1]}")


@app.command()
def catalog() -> None:
    """Print the block family catalog summary given to the LLM."""
    from craftpilot.blocks.catalog import catalog_summary

    typer.echo(catalog_summary())


def main() -> None:
    app()
