"""Run the engine stages in order."""

from __future__ import annotations

import numpy as np

from craftpilot.engine import attachments  # noqa: F401  (registers kinds)
from craftpilot.engine.attachments.base import auto_budget, run_level
from craftpilot.engine.attic import attic
from craftpilot.engine.depth import depth
from craftpilot.engine.detail import detail
from craftpilot.engine.facade import facade
from craftpilot.engine.interior import interior
from craftpilot.engine.layout import layout
from craftpilot.engine.massing import massing
from craftpilot.engine.materials import materials
from craftpilot.engine.postprocess import postprocess
from craftpilot.engine.roof import roof
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import Bounds, BuildProgram


def generate(program: BuildProgram, bounds: Bounds, seed: int) -> SemanticGrid:
    program = program.model_copy(deep=True)
    rng = np.random.default_rng(seed)
    grid = SemanticGrid(bounds.width, bounds.height, bounds.depth, rng)
    grid.report["seed"] = seed
    grid.report["bounds"] = bounds.as_tuple()
    grid.report["label"] = program.label
    layout(grid, program)
    auto_budget(grid, program)
    run_level(grid, program, "mass")
    massing(grid, program)
    roof(grid, program)
    attic(grid, program)
    run_level(grid, program, "roof")
    facade(grid, program)
    run_level(grid, program, "facade")
    interior(grid, program)
    depth(grid, program)
    detail(grid, program)
    materials(grid, program)
    postprocess(grid, program)
    return grid
