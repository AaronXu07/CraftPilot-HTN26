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


def test_relief_gate_distinguishes_pancakes_from_narrow_objects():
    from craftpilot.objects.voxelize import relief_like

    assert flatness(_colored_box(size=(1, 1, 1))) == 1.0
    assert relief_like(_colored_box(size=(0.96, 0.93, 0.30)))  # the failed car: thin along up (z), square footprint
    assert not relief_like(_colored_box(size=(0.31, 0.71, 1.0)))  # a rearing horse: thin sideways
    assert not relief_like(_colored_box(size=(1.0, 0.45, 0.30)))  # a real low car: thin up but 2:1 footprint
    assert not relief_like(_colored_box(size=(0.5, 0.5, 1.0)))  # an upright statue


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


def test_placement_geometry_faces_the_player():
    from craftpilot.objects import place as P
    from craftpilot.objects.voxelize import to_grid

    # an L-shaped object: a tall column plus a foot sticking out toward +x in the mesh frame (the presented side)
    m = trimesh.util.concatenate([
        _colored_box(size=(2, 2, 8)).apply_translation([0, 0, 4]),          # column (z-up frame)
        _colored_box(size=(4, 2, 2)).apply_translation([3, 0, 1]),          # foot toward +x
    ])
    grid = to_grid(voxelize_mesh(m, height=8))
    blocks = P.grid_blocks(grid)
    xs = [b[0] for b in blocks]; zs = [b[2] for b in blocks]
    # after the presentation turn the foot (mesh +x) points to +z (south, toward the player)
    foot = [b for b in blocks if b[1] == 0]
    assert max(z for _, _, z, _ in foot) > max(zs) - 2 and min(xs) < 0 < max(xs)
    # player standing north of the origin looking south: the object goes south of them, foot pointing back at them
    anchor, k = P.plan_anchor({"pos": [10.5, 64.0, 10.5], "yaw": 0.0}, blocks, gap=2)
    assert k == P.FACING_TURNS["south"]
    world = P.to_world(blocks, anchor, k)
    assert min(w[2] for w in world) == 10 + 1 + 2  # 2 air blocks in front of the player
    assert min(w[1] for w in world) == 64  # ground at the player's feet
    ws = sorted(world, key=lambda b: b[2])
    assert ws[0][1] == 0 + 64 and any(b[1] == 64 for b in ws[:4])  # the foot is the nearest part
    chunks = P.layer_chunks(world, 5)
    assert all(c[0][1] <= c[-1][1] for c in chunks) and chunks[0][0][1] == 64
