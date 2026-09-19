"""Object path: voxelisation and block matching on synthetic meshes, the offline brief, prompt cascade."""
from __future__ import annotations

import numpy as np
import trimesh

from craftpilot.objects.brief import compose_brief, fallback_brief
from craftpilot.objects.imagegen import CAMERAS, compose_prompt, prompt_attempts
from craftpilot.objects.voxelize import (
    PALETTE,
    _srgb_to_lab,
    clamp_chroma,
    flatness,
    match_blocks,
    to_grid,
    voxelize_mesh,
)


def _colored_box(size=(2.0, 6.0, 2.0), rgb=(200, 30, 30)) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.visual.vertex_colors = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(m.vertices), 1))
    return m


def test_voxelize_scales_to_height_and_fills_interior():
    # a 2x6x2 y-up box asked to be 12 tall -> 4x12x4 voxels, solid inside
    obj = voxelize_mesh(_colored_box(), height=12, up="y")
    assert obj.size[1] == 12 and obj.size[0] in (4, 5) and obj.size[2] in (4, 5)
    assert obj.block_count >= 4 * 12 * 4 * 0.9
    assert obj.occupancy[obj.size[0] // 2, 6, obj.size[2] // 2]  # interior filled


def test_voxelize_z_up_default_rotates_triposr_frame():
    # a box that is tall along z (TripoSR's up) must stand up along y after the default orientation
    m = _colored_box(size=(2.0, 2.0, 6.0))
    obj = voxelize_mesh(m, height=12)  # up="z" default
    assert obj.size[1] == 12


def test_colors_map_to_nearest_block_in_palette_family():
    obj = voxelize_mesh(_colored_box(rgb=(200, 30, 30)), height=8, up="y")
    grid = to_grid(obj)
    names = {r.block_id.split(":")[1] for r in grid.palette}
    assert names & {"red_concrete", "red_wool", "red_terracotta"}, names
    # a stone-only allowed list keeps a red box grey
    grid2 = to_grid(obj, allowed=["stone", "andesite", "deepslate"])
    assert {r.block_id.split(":")[1] for r in grid2.palette} <= {"stone", "andesite", "deepslate"}


def test_match_blocks_grey_stays_grey_and_red_stays_red():
    greys = np.array([[120, 120, 120]] * 95 + [[200, 120, 200]] * 5)  # 5% magenta hallucination on a grey statue
    out = match_blocks(greys, max_types=3)
    assert set(out.tolist()) <= {"stone", "stone_bricks", "cobblestone", "andesite", "polished_andesite", "light_gray_concrete", "tuff"}
    car = np.array([[20, 20, 20]] * 60 + [[190, 30, 30]] * 40)  # black tyres/glass are the majority, body is red
    out = match_blocks(car, max_types=4)
    assert any(n.startswith("red") or n == "nether_bricks" for n in out.tolist())


def test_clamp_chroma_uses_high_percentile():
    lab = _srgb_to_lab(np.array([[20, 20, 20]] * 60 + [[190, 30, 30]] * 40, dtype=float))
    out = clamp_chroma(lab)
    red = np.hypot(out[60:, 1], out[60:, 2])
    assert red.min() > 40  # the red 40% keeps its chroma


def test_flatness_detects_relief_meshes():
    assert flatness(_colored_box(size=(1, 1, 1))) == 1.0
    assert flatness(_colored_box(size=(1, 1, 0.2))) < 0.32


def test_palette_only_has_full_opaque_blocks():
    bad = [n for n in PALETTE if any(s in n for s in ("glass", "leaves", "slab", "stairs", "fence", "pane", "door"))]
    assert not bad


def test_fallback_brief_reads_height_and_plinth():
    b = fallback_brief("build a dragon statue 40 blocks tall")
    assert b.height == 40 and b.plinth and b.palette == "stone" and b.label == "dragon_statue" and b.source == "fallback"
    c = fallback_brief("a red sports car")
    assert not c.plinth and c.height == 24 and c.palette == "auto" and c.subject == "red sports car"
    brief, meta = compose_brief("a wooden rowing boat", use_llm=False, height=10)
    assert brief.height == 10 and brief.palette == "wood" and meta["seconds"] == 0.0


def test_prompt_attempts_cascade_and_cameras():
    attempts = prompt_attempts("A sleek red car with a spoiler", plinth=False, style="glossy paint", request="a red car")
    assert len(attempts) == 4
    assert attempts[0].startswith(CAMERAS[0]) and "glossy paint" in attempts[0]
    assert "glossy paint" not in attempts[1]  # style dropped
    assert "a red car. Whole object" in attempts[2]  # the player's own words
    assert "museum exhibit" in attempts[3].lower() and "plinth" in attempts[3]
    p1 = compose_prompt("a cat", plinth=True, camera=1)
    assert p1.startswith(CAMERAS[1]) and "plinth" in p1
    assert len({compose_prompt("x", False, camera=i) for i in range(len(CAMERAS))}) == len(CAMERAS)
