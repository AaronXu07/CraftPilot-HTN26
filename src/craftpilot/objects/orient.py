"""Turn an object grid to the engine's authoring convention: the presented side faces south (+z).

The reconstructor reports which axis faced the camera (`grid.report["object_front"]`, set by
objects.pipeline). Once the grid faces south it is indistinguishable from an engine-built building for the
hologram cloud, `grid/ops.rotate_cw(grid, quarter_turns_for_facing(...))` at placement, and `place_grid`.
"""

from __future__ import annotations

import os

from craftpilot.grid.enums import Dir
from craftpilot.grid.ops import rotate_cw
from craftpilot.grid.semantic import SemanticGrid

# Clockwise quarter turns (seen from above, x east / z south) that bring each reported front to +z. Same
# handedness as grid/ops.rotate_cw and the mod's hologram rotation: (x, z) -> (D-1-z, x).
PRESENTATION_TURNS = {"+x": 3, "+z": 0, "-x": 1, "-z": 2}
DEFAULT_FRONT = "+x"  # TripoSR's frame; Hunyuan3D reports "+z"


def face_south(grid: SemanticGrid) -> SemanticGrid:
    """Rotate `grid` in place so its presented side faces +z. Idempotent: a grid that already faces south (or
    has no front recorded) is left alone. CRAFTPILOT_OBJECT_TURNS=<n> adds n extra quarter turns as a
    calibration knob for a reconstructor whose reported front comes out wrong in-game."""
    front = str(grid.report.get("object_front", DEFAULT_FRONT))
    turns = PRESENTATION_TURNS.get(front, PRESENTATION_TURNS[DEFAULT_FRONT])
    if "object_front" not in grid.report:
        turns = 0  # not an object grid, or already normalised by a caller that dropped the key
    try:
        turns += int(os.environ.get("CRAFTPILOT_OBJECT_TURNS", "0"))
    except ValueError:
        pass
    if turns % 4:
        rotate_cw(grid, turns % 4)
    grid.report["object_front"] = "+z"
    grid.front = Dir.SOUTH
    return grid
