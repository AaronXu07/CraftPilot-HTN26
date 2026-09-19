"""Command line interface."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import typer

from craftpilot.config import SETTINGS
from craftpilot.program.model import Bounds, BuildProgram

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
              author: str, source_text: str, extra_notes: list[str], facing: str = "south") -> dict:
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
    if preview:
        from craftpilot.preview.render import render
        png = out.with_suffix(".png")
        render(grid, png)
        result["preview"] = str(png)
    return result


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
) -> None:
    """Compose a build program from TEXT, generate it, and write a Litematica schematic."""
    from craftpilot.llm.compose import compose
    from craftpilot.program.exemplars import load_one

    if seed is None:
        seed = int(time.time()) % 1_000_000
    b = _parse_bounds(bounds)
    notes: list[str] = []
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
    result = run_build(program, b, seed, out, preview, author, text, notes, facing)
    result["source"] = source
    result["compose_seconds"] = round(compose_s, 3)
    result["total_seconds"] = round(time.time() - t_start, 3)
    typer.echo(json.dumps(result, indent=2))
    typer.echo(f"Built in {result['total_seconds']:.1f}s: compose {compose_s:.1f}s, generate {result['generate_seconds']:.2f}s, "
               f"export {result['export_seconds']:.2f}s", err=True)


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


@app.command()
def catalog() -> None:
    """Print the block family catalog summary given to the LLM."""
    from craftpilot.blocks.catalog import catalog_summary

    typer.echo(catalog_summary())


def main() -> None:
    app()
