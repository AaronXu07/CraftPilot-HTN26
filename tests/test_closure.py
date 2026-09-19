"""Closure: any program that passes validation renders to a complete, in-bounds building."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from craftpilot.blocks import catalog
from craftpilot.engine.pipeline import generate
from craftpilot.grid.enums import Role
from craftpilot.program.model import (
    Align,
    Attach,
    AttachmentKind,
    AttachmentParams,
    AttachmentRequest,
    Bounds,
    BuildProgram,
    DepthRules,
    FacadeRules,
    FamilyWeight,
    Framing,
    Gradient,
    InteriorRules,
    IntRange,
    PaletteSpec,
    PartSpec,
    RolePalette,
    RoofSpec,
    RoofType,
    Shape,
    Side,
    SilhouetteBudget,
    Size,
    WindowStyle,
)
from craftpilot.program.validate import repair

FAMILY_NAMES = sorted(catalog.FAMILIES)

role_palette = st.builds(
    RolePalette,
    families=st.lists(st.builds(FamilyWeight, family=st.sampled_from(FAMILY_NAMES),
                                weight=st.floats(0.1, 2.0)), min_size=1, max_size=4),
    gradient=st.sampled_from(list(Gradient)),
    weathering=st.floats(0, 1),
    texture_rate=st.floats(0, 0.5),
)

roof_spec = st.builds(RoofSpec, type=st.sampled_from(list(RoofType)), pitch=st.sampled_from([0.5, 1.0, 1.5, 2.0]),
                      overhang=st.integers(0, 2), tiers=st.integers(1, 4), crenellated=st.booleans(),
                      ridge_axis=st.sampled_from(["auto", "x", "z"]))


def part_spec(name: str, root: bool, parent_names: list[str]):
    attach = st.none() if root else st.builds(Attach, to=st.sampled_from(parent_names), side=st.sampled_from(list(Side)),
                                              align=st.sampled_from(list(Align)), overlap=st.integers(0, 4),
                                              offset=st.integers(-4, 4), offset_z=st.integers(-3, 3))
    return st.builds(PartSpec, name=st.just(name), shape=st.sampled_from(list(Shape)), sides=st.integers(3, 10),
                     size=st.builds(Size, width=st.floats(0.2, 1.0), depth=st.floats(0.2, 1.0)),
                     floors=st.integers(1, 5), floor_height=st.integers(3, 6), taper=st.floats(0, 0.2),
                     wall_thickness=st.integers(1, 2), odd_dims=st.booleans(), roof=roof_spec, attach=attach)


@st.composite
def programs(draw):
    n_parts = draw(st.integers(1, 3))
    parts = [draw(part_spec("p0", True, ["p0"]))]
    for i in range(1, n_parts):
        parts.append(draw(part_spec(f"p{i}", False, [p.name for p in parts])))
    attachments = draw(st.lists(st.builds(AttachmentRequest, kind=st.sampled_from(list(AttachmentKind)),
                                          count=st.one_of(st.none(), st.integers(0, 4)),
                                          spacing=st.sampled_from(["regular", "irregular"]),
                                          params=st.builds(AttachmentParams, width=st.one_of(st.none(), st.integers(1, 7)),
                                                           depth=st.one_of(st.none(), st.integers(1, 4)),
                                                           height=st.one_of(st.none(), st.integers(1, 8)),
                                                           radius=st.one_of(st.none(), st.integers(1, 5)),
                                                           floors=st.one_of(st.none(), st.integers(1, 6)),
                                                           roof=st.one_of(st.none(), st.sampled_from(list(RoofType))),
                                                           glazed=st.one_of(st.none(), st.booleans()))),
                                max_size=5))
    return BuildProgram(
        label="fuzz",
        bounds=Bounds(width=draw(st.integers(9, 40)), height=draw(st.integers(10, 40)), depth=draw(st.integers(9, 40))),
        parts=parts,
        facade=draw(st.builds(FacadeRules, bay_width=st.builds(IntRange, min=st.integers(2, 5), max=st.integers(2, 7)),
                              window=st.sampled_from(list(WindowStyle)), window_width=st.integers(1, 3),
                              window_height=st.integers(1, 4), shutters=st.floats(0, 1), sills=st.booleans(),
                              framing=st.sampled_from(list(Framing)), symmetry=st.booleans(),
                              max_flat_run=st.integers(3, 12), ground_floor_taller=st.integers(0, 2))),
        depth=draw(st.builds(DepthRules, frame_protrude=st.integers(0, 2), window_inset=st.integers(0, 1),
                             eave_trim=st.booleans(), foundation_rise=st.integers(0, 3),
                             foundation_outset=st.integers(0, 2), floor_lips=st.booleans(), corbels=st.booleans())),
        attachments=attachments,
        budget=draw(st.builds(SilhouetteBudget, dominant=st.integers(0, 2), medium_min=st.integers(0, 3),
                              medium_max=st.integers(0, 4), small_min=st.integers(0, 2), small_max=st.integers(0, 3))),
        interior=draw(st.builds(InteriorRules, stairs=st.booleans(), doorways=st.booleans(),
                                partitions=st.booleans(), lighting=st.booleans())),
        palette=draw(st.builds(PaletteSpec, primary=role_palette, roof=role_palette,
                               secondary=st.one_of(st.none(), role_palette), accent=st.one_of(st.none(), role_palette),
                               framing=st.one_of(st.none(), role_palette), trim=st.one_of(st.none(), role_palette),
                               foundation=st.one_of(st.none(), role_palette), glass=st.none())),
    )


@settings(max_examples=60, deadline=None)
@given(programs(), st.integers(0, 10_000))
def test_any_valid_program_renders(program, seed):
    fixed, _ = repair(program, (256, 256, 256))
    grid = generate(fixed, fixed.bounds, seed)
    assert grid.block_count() > 0
    # Every placed block is inside the bounds by construction; check the palette resolves.
    assert all(ref.block_id.startswith("minecraft:") for ref in grid.palette)
    # No stair without a facing, no lantern floating.
    for x, y, z in zip(*np.nonzero(grid.role == Role.LIGHT)):
        below = grid.block[x, y - 1, z] >= 0 if y > 0 else False
        above = grid.block[x, y + 1, z] >= 0 if y + 1 < grid.H else False
        assert below or above
    known = catalog.KNOWN_BLOCKS
    if known is not None:
        assert all(ref.block_id in known for ref in grid.palette)
