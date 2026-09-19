"""Unit tests for the engine stages."""

from __future__ import annotations

import numpy as np
import pytest

from craftpilot.blocks import catalog
from craftpilot.blocks.circles import circle_mask, ngon_mask, perimeter
from craftpilot.blocks.state import BlockRef, rotate_ref_cw
from craftpilot.engine.pipeline import generate
from craftpilot.engine.roof import height_map
from craftpilot.grid.enums import BShape, Dir, Role
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.grid.semantic import LayoutPart
from craftpilot.program.model import (
    AttachmentKind,
    AttachmentRequest,
    Bounds,
    BuildProgram,
    FamilyWeight,
    PaletteSpec,
    PartSpec,
    RolePalette,
    RoofSpec,
    RoofType,
    Size,
)
from craftpilot.program.validate import repair

SAFETY = (256, 256, 256)


def simple_program(roof: RoofType = RoofType.gable, floors: int = 2, **kw) -> BuildProgram:
    return BuildProgram(
        label="test",
        bounds=Bounds(width=19, height=20, depth=15),
        parts=[PartSpec(name="main", size=Size(width=1.0, depth=1.0), floors=floors,
                        roof=RoofSpec(type=roof, pitch=1.0, overhang=1), **kw)],
        palette=PaletteSpec(primary=RolePalette(families=[FamilyWeight(family="oak")]),
                            roof=RolePalette(families=[FamilyWeight(family="dark_oak")])),
    )


def test_circle_mask_is_symmetric_and_odd():
    m = circle_mask(11, 11, 5, 5, 5)
    assert m.sum() > 60
    assert np.array_equal(m, m[::-1, :]) and np.array_equal(m, m[:, ::-1])


def test_ngon_mask_inside_circle():
    n = ngon_mask(15, 15, 7, 7, 6, 8)
    c = circle_mask(15, 15, 7, 7, 6.5)
    assert (n & ~c).sum() == 0
    assert n.sum() > 60


def test_perimeter_of_rect():
    m = np.zeros((7, 7), dtype=bool)
    m[1:6, 1:6] = True
    assert perimeter(m).sum() == 16


def test_hip_roof_height_map_is_chessboard_distance():
    spec = simple_program(RoofType.hip).parts[0]
    mask = np.zeros((11, 11), dtype=bool)
    mask[2:9, 2:9] = True
    part = LayoutPart(0, spec, mask, 2, 2, 8, 8)
    h, dil, closed, top_role, fill_role, top_shape = height_map(part, spec.roof, mask)
    assert dil.sum() == 81            # 7x7 dilated by 1
    assert h[5, 5] == 5               # centre: distance 5 from the dilated edge
    assert h[1, 1] == 1 and closed[1, 1]


def test_gable_ridge_along_long_axis():
    spec = simple_program(RoofType.gable).parts[0]
    mask = np.zeros((15, 9), dtype=bool)
    mask[1:14, 2:7] = True           # 13 wide, 5 deep -> ridge along x
    part = LayoutPart(0, spec, mask, 1, 2, 13, 6)
    h, dil, closed, *_ = height_map(part, spec.roof, mask)
    assert h[7, 4] == 4               # centre row is the ridge: (7 dilated rows + 1) / 2
    assert h[7, 1] == 1 and h[7, 7] == 1
    assert h[2, 4] == h[12, 4]        # same profile along the ridge


@pytest.mark.parametrize("roof", list(RoofType))
def test_every_roof_type_renders(roof):
    prog, _ = repair(simple_program(roof), SAFETY)
    grid = generate(prog, prog.bounds, 1)
    assert grid.block_count() > 100
    assert grid.door is not None


def test_stairs_face_uphill_on_gable():
    prog, _ = repair(simple_program(RoofType.gable), SAFETY)
    grid = generate(prog, prog.bounds, 1)
    stairs = [(x, y, z) for x, y, z in zip(*np.nonzero((grid.role == Role.ROOF) & (grid.shape == BShape.STAIR)))]
    assert stairs
    for x, y, z in stairs:
        n = int(grid.normal[x, y, z])
        vx, _, vz = {Dir.NORTH: (0, 0, -1), Dir.SOUTH: (0, 0, 1), Dir.EAST: (1, 0, 0), Dir.WEST: (-1, 0, 0)}[n]
        assert grid.roof_surface[x + vx, z + vz] >= grid.roof_surface[x, z]


def test_every_block_state_is_valid_for_the_target_version():
    prog, _ = repair(simple_program(RoofType.hip), SAFETY)
    prog.attachments = [AttachmentRequest(kind=k) for k in AttachmentKind]
    grid = generate(prog, prog.bounds, 3)
    known = catalog.KNOWN_BLOCKS
    if known is None:
        pytest.skip("no generated block list")
    import json
    import os
    from pathlib import Path
    version = os.environ.get("CRAFTPILOT_MC_VERSION", "26.2")
    data = json.loads((Path(__file__).resolve().parents[1] / "data" / f"blocks_{version}.json").read_text())["blocks"]
    for ref in grid.palette:
        assert ref.block_id in known, ref
        props = data[ref.block_id]["properties"]
        for k, v in ref.props:
            assert k in props, (ref, k)
            assert v in props[k], (ref, k, v)


def test_rotation_maps_north_marker_to_east():
    prog, _ = repair(simple_program(RoofType.gable), SAFETY)
    grid = generate(prog, prog.bounds, 1)
    door_before = grid.door
    rotate_cw(grid, quarter_turns_for_facing("east"))
    assert grid.front == Dir.EAST
    x, y, z = grid.door
    ref = grid.palette[int(grid.block[x, y, z])]
    assert ref.block_id.endswith("_door") and ref.prop("facing") == "west"   # door faces inward
    assert door_before != grid.door


def test_rotate_ref_cw_rotates_connections():
    ref = BlockRef.make("minecraft:oak_fence", north="true", east="false", south="false", west="false")
    r = rotate_ref_cw(ref, 1)
    assert r.prop("east") == "true" and r.prop("north") == "false"


def test_repair_fixes_broken_programs():
    prog = simple_program(RoofType.gable)
    prog.parts.append(PartSpec(name="wing", size=Size(width=0.5, depth=0.5), roof=RoofSpec(type=RoofType.hip)))
    prog.parts[1].attach = None       # two roots
    prog.palette.primary.families.append(FamilyWeight(family="not_a_family"))
    prog.parts[0].floors = 99
    fixed, notes = repair(prog, SAFETY)
    assert len([p for p in fixed.parts if p.attach is None]) == 1
    assert all(catalog.family(f.family) for f in fixed.palette.primary.families)
    assert fixed.parts[0].floors == 12
    assert notes


def test_every_part_is_reachable_from_the_door():
    from craftpilot.engine.interior import reachable
    from craftpilot.program.exemplars import load_all
    from pathlib import Path
    for ex in load_all(Path(__file__).resolve().parents[1] / "exemplars"):
        prog, _ = repair(ex.program, SAFETY)
        grid = generate(prog, prog.bounds, 2)
        seen = reachable(grid)
        for part in grid.parts:
            if part.spec.attach is not None and part.spec.attach.side.value == "top":
                continue   # stacked parts connect through stairs, checked separately later
            y = part.floor_block_y(0) + 1
            cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(part.mask))
                     if grid.role[x, y, z] == Role.INTERIOR and grid.part_id[x, y, z] == part.index]
            if not cells:
                continue
            assert any(seen[x, y, z] for (x, z) in cells), (ex.name, part.spec.name)


def test_palette_repair_rules():
    from craftpilot.program.palette import check_and_repair
    # Cobblestone as the wall of a modern villa is too busy; mossy stone and prismarine do not belong on a cottage.
    pal = PaletteSpec(primary=RolePalette(families=[FamilyWeight(family="cobblestone")]),
                      roof=RolePalette(families=[FamilyWeight(family="prismarine")]),
                      accent=RolePalette(families=[FamilyWeight(family="purpur")]))
    notes = check_and_repair(pal, "modern villa")
    assert pal.primary.families[0].family != "cobblestone"
    assert catalog.family(pal.roof.families[0].family).has("stairs")
    assert notes
    # A classic combination passes untouched.
    pal = PaletteSpec(primary=RolePalette(families=[FamilyWeight(family="bricks")]),
                      roof=RolePalette(families=[FamilyWeight(family="deepslate_tiles")]),
                      framing=RolePalette(families=[FamilyWeight(family="stone_bricks")]),
                      foundation=RolePalette(families=[FamilyWeight(family="stone_bricks")]))
    assert check_and_repair(pal, "townhouse") == []
    # Exemplar palettes are all acceptable as written.
    from craftpilot.program.exemplars import load_all
    from pathlib import Path
    for ex in load_all(Path(__file__).resolve().parents[1] / "exemplars"):
        assert check_and_repair(ex.program.palette, ex.program.label) == [], ex.name


def _roof_holes(grid) -> int:
    """Air cells under a roof, inside the footprint, that touch outside air sideways."""
    from craftpilot.grid.enums import DIR_VEC, HORIZONTAL
    holes = 0
    for part in grid.parts:
        if part.roof_surface is None:
            continue
        for x, z in zip(*np.nonzero(part.mask)):
            s = grid.roof_surface[x, z]
            if not np.isfinite(s):
                continue
            top = int(np.ceil(s)) - 1
            for y in range(part.eave_y, top):
                if grid.role[x, y, z] not in (Role.EMPTY, Role.INTERIOR):
                    continue
                for d in HORIZONTAL:
                    vx, _, vz = DIR_VEC[d]
                    nx, nz = x + vx, z + vz
                    if grid.in_bounds(nx, y, nz) and grid.role[nx, y, nz] == Role.EMPTY and not grid.footprint[nx, nz]:
                        holes += 1
                        break
    return holes


@pytest.mark.parametrize("roof", [RoofType.cone, RoofType.dome, RoofType.spire, RoofType.hip, RoofType.gable])
@pytest.mark.parametrize("pitch", [0.5, 1.0, 1.5])
def test_round_and_sloped_roofs_have_no_holes(roof, pitch):
    from craftpilot.program.model import Shape
    prog = simple_program(roof, shape=Shape.circle if roof in (RoofType.cone, RoofType.dome, RoofType.spire) else Shape.rect)
    prog.parts[0].roof.pitch = pitch
    prog.attachments = []
    prog.budget.dominant = prog.budget.medium_min = 0
    prog, _ = repair(prog, SAFETY)
    grid = generate(prog, prog.bounds, 1)
    assert _roof_holes(grid) == 0


def test_exemplar_roofs_have_no_holes():
    from craftpilot.program.exemplars import load_all
    from pathlib import Path
    for ex in load_all(Path(__file__).resolve().parents[1] / "exemplars"):
        prog, _ = repair(ex.program, SAFETY)
        grid = generate(prog, prog.bounds, 4)
        assert _roof_holes(grid) == 0, ex.name


def _staircases_walkable(grid) -> list[str]:
    """Every interior staircase must be climbable: steps rise one per cell, each step faces the
    direction you arrive from (low half toward the previous step), three clear cells above each
    step, a floor cell to start from, and a solid landing with clear head room."""
    from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, Dir
    problems = []
    is_step = lambda x, y, z: grid.in_bounds(x, y, z) and grid.role[x, y, z] == Role.STAIRCASE and grid.shape[x, y, z] == BShape.STAIR
    seen = set()
    for x, y, z in zip(*np.nonzero((grid.role == Role.STAIRCASE) & (grid.shape == BShape.STAIR))):
        x, y, z = int(x), int(y), int(z)
        # A bottom step has no step one below it in any neighbouring cell.
        if any(is_step(x + DIR_VEC[d][0], y - 1, z + DIR_VEC[d][2]) for d in HORIZONTAL) or (x, y, z) in seen:
            continue
        n = int(grid.normal[x, y, z])
        vx, _, vz = DIR_VEC[n]
        ax, az = x - vx, z - vz
        if grid.role[ax, y - 1, az] not in (Role.FLOOR, Role.FOUNDATION, Role.PARTITION) or grid.role[ax, y, az] != Role.INTERIOR:
            problems.append(f"no floor cell to start from before {(x, y, z)}")
        cx, cy, cz = x, y, z
        steps = 0
        while True:
            seen.add((cx, cy, cz))
            steps += 1
            for yy in (cy + 1, cy + 2, cy + 3):
                if grid.in_bounds(cx, yy, cz) and grid.role[cx, yy, cz] not in (Role.INTERIOR, Role.EMPTY, Role.LIGHT, Role.STAIRCASE):
                    problems.append(f"head hit above step at {(cx, cy, cz)}: {Role(int(grid.role[cx, yy, cz])).name} at y={yy}")
            nxt = None
            for d in HORIZONTAL:
                dx, _, dz = DIR_VEC[d]
                if is_step(cx + dx, cy + 1, cz + dz):
                    nxt = (cx + dx, cy + 1, cz + dz, d)
                    break
            if nxt is None:
                break
            nx, ny, nz, d = nxt
            if int(grid.normal[nx, ny, nz]) != d:
                problems.append(f"step at {(nx, ny, nz)} faces {Dir(int(grid.normal[nx, ny, nz])).name}, arrival is {Dir(d).name}: a jump")
            cx, cy, cz = nx, ny, nz
        # Landing: a solid floor cell next to the top step at the same level as its top.
        landing = None
        for d in HORIZONTAL:
            dx, _, dz = DIR_VEC[d]
            lx, lz = cx + dx, cz + dz
            if grid.in_bounds(lx, cy, lz) and grid.role[lx, cy, lz] in (Role.FLOOR, Role.PARTITION, Role.WALL) \
                    and all(grid.role[lx, yy, lz] in (Role.INTERIOR, Role.EMPTY, Role.LIGHT) for yy in (cy + 1, cy + 2)):
                landing = (lx, lz)
                break
        if landing is None:
            problems.append(f"no landing after run ending at {(cx, cy, cz)}")
        if steps < 3:
            problems.append(f"run of {steps} steps at {(x, y, z)} is too short")
    return problems


def test_interior_staircases_are_walkable():
    from craftpilot.program.exemplars import load_all
    from pathlib import Path
    for ex in load_all(Path(__file__).resolve().parents[1] / "exemplars"):
        prog, _ = repair(ex.program, SAFETY)
        grid = generate(prog, prog.bounds, 2)
        assert _staircases_walkable(grid) == [], ex.name
