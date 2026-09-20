"""Prompt files exist, mention every tool, and fit the token budget."""
from __future__ import annotations

import os
import re

from copilot.pipeline import STAGES, load_prompt, system_prompt
from copilot.pipeline.stages import PROMPT_DIR
from tests.fakes import FakeCtx

PROMPTS = ["system_core", "stage_interpret", "stage_blocking", "stage_detailing", "stage_materials", "stage_decoration", "critic", "router"]

CONTRACT_TOOLS = [
    "add", "delete", "duplicate", "rename", "move", "move_to", "rotate", "scale", "align", "stack", "mirror_copy",
    "set_shape", "set_op", "set_material", "set_anchor", "set_visible", "reorder", "add_modifier", "remove_modifier",
    "set_modifier", "group", "ungroup", "select", "describe", "bbox", "measure", "top_of", "side_of", "define_material",
    "list_materials", "paint", "undo", "redo", "snapshot", "restore", "run_script", "render", "lint", "place",
    "undo_world", "export_schematic", "materials_list", "get_player", "say", "search_blocks", "nearest_block",
    "set_brief", "finish",
]


def _tool_names():
    try:
        from copilot.tools.schemas import TOOL_SCHEMAS

        return [t["function"]["name"] if "function" in t else t["name"] for t in TOOL_SCHEMAS]
    except Exception:  # noqa: BLE001
        return CONTRACT_TOOLS


def test_all_prompt_files_exist():
    for p in PROMPTS:
        assert os.path.exists(os.path.join(PROMPT_DIR, p + ".md")), p
        assert len(load_prompt(p)) > 500


def test_system_core_mentions_every_tool():
    core = load_prompt("system_core")
    missing = [n for n in _tool_names() if f"`{n}(" not in core and f"`{n}`" not in core and f" {n}(" not in core]
    assert not missing, f"system_core.md does not mention: {missing}"


def test_placeholder_substitution_and_budget():
    ctx = FakeCtx()
    for stage in STAGES:
        sp = system_prompt(ctx, stage)
        assert not re.search(r"\{\{\w+\}\}", sp), "unfilled placeholder"
        assert "Materials catalog (fake)" in sp
        assert load_prompt(stage.prompt_file).splitlines()[0] in sp
        assert len(sp) + 10_000 < 24_000, f"{stage.name} system prompt too long with a 10k-char catalog: {len(sp)}"


def test_prompts_carry_the_core_content():
    core = load_prompt("system_core")
    for needle in ("x east", "y up", "z south", "bottom_center", "snake_case", "P8", "run_script", "describe(", "scene castle_v3"):
        assert needle in core, needle
    critic = load_prompt("critic")
    assert '"top_3_fixes"' in critic and "P1" in critic and "P8" in critic
    router = load_prompt("router")
    assert '"stages"' in router and '"selection"' in router and '"direct_ops"' in router
    for stage in STAGES:
        txt = load_prompt(stage.prompt_file)
        assert "## Checklist" in txt and "Worked example" in txt, stage.name
