"""Stage definitions and the stage runner (plan §7.2).

Every stage uses the same tools and sees the same scene; they differ by system prompt, checklist and
which ops are encouraged. `run_stage` composes system_core + the stage prompt, runs the tool loop,
then renders and lints so the caller (orchestrator/critic) has fresh feedback.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from ..llm import DEFAULT_OP_CAP, LoopResult, run_tool_loop
from .budget import get_budget, profile_row
from .common import call_tool, env_int, get_dispatch, get_tools, outline

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

# tools the model must not use inside a stage (the orchestrator owns placement/export)
STAGE_TOOL_EXCLUDE = {"place", "undo_world", "export_schematic", "set_brief"}

FALLBACK_CATALOG = (
    "## Materials catalog (abridged)\n"
    "wood (oak, spruce, birch, jungle, acacia, dark_oak, mangrove, cherry, bamboo, crimson, warped): "
    "<w>_planks/_log/_stripped_log/_stairs/_slab/_fence/_door/_trapdoor. "
    "stone: stone, stone_bricks, cracked/mossy/chiseled_stone_bricks, cobblestone, mossy_cobblestone, andesite, "
    "diorite, granite (+polished_), deepslate, deepslate_bricks/tiles (+cracked_), tuff, blackstone, "
    "polished_blackstone_bricks, sandstone, red_sandstone (+smooth_, cut_, chiseled_), nether_bricks, "
    "red_nether_bricks, prismarine(_bricks), dark_prismarine, quartz_block, smooth_quartz, end_stone_bricks, "
    "mud_bricks, bricks, purpur_block; each has _stairs/_slab and most have _wall. "
    "16 colours × wool/concrete/concrete_powder/terracotta/stained_glass/glazed_terracotta. "
    "copper: copper_block, cut_copper (+exposed_/weathered_/oxidized_, waxed_), cut_copper_stairs/slab. "
    "glass, glass_pane, iron_bars, chain, lantern, soul_lantern, torch, wall_torch, glowstone, sea_lantern, "
    "campfire, banners (<colour>_banner/_wall_banner), flower_pot, potted_*, leaves, moss_block, vines, "
    "hay_block, barrel, chest, crafting_table, bookshelf, anvil, bell, gravel, dirt_path, coarse_dirt, "
    "grass_block, water, lava, ice, snow_block, bone_block, obsidian, gold_block, iron_block."
)


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Read prompts/<name>.md (cached)."""
    path = os.path.join(PROMPT_DIR, name if name.endswith(".md") else name + ".md")
    with open(path, encoding="utf-8") as f:
        return f.read()


def catalog_text(ctx: Any) -> str:
    reg = getattr(ctx, "registry", None)
    if reg is not None and hasattr(reg, "catalog_text"):
        try:
            txt = reg.catalog_text()
            if txt:
                return str(txt)
        except Exception:  # noqa: BLE001
            pass
    return FALLBACK_CATALOG


def system_prompt(ctx: Any, stage: Optional["Stage"] = None, extra_sections: Optional[List[str]] = None) -> str:
    """system_core with the catalog placeholder filled, plus the stage prompt and extra sections."""
    core = load_prompt("system_core").replace("{{materials_catalog}}", catalog_text(ctx))
    parts = [core]
    if stage is not None:
        parts.append(load_prompt(stage.prompt_file))
    for s in extra_sections or []:
        if s:
            parts.append(s)
    return "\n\n".join(parts)


@dataclass
class Stage:
    name: str
    prompt_file: str
    encouraged_tools: List[str] = field(default_factory=list)
    temperature: float = 0.2
    max_calls: int = 40
    critic_focus: str = ""

    def __str__(self) -> str:
        return self.name


STAGES: List[Stage] = [
    Stage(
        "blocking",
        "stage_blocking",
        ["add", "stack", "align", "mirror_copy", "group", "run_script", "render", "set_shape", "move"],
        critic_focus="silhouette and proportions vs the brief",
    ),
    Stage(
        "detailing",
        "stage_detailing",
        ["add", "add_modifier", "reorder", "run_script", "lint", "render", "set_shape"],
        critic_focus="depth, rhythm, entrance, blank faces",
    ),
    Stage(
        "materials",
        "stage_materials",
        ["define_material", "set_material", "paint", "select", "search_blocks", "nearest_block", "list_materials", "render"],
        critic_focus="contrast, grounding gradient, palette variation, fit",
    ),
    Stage(
        "decoration",
        "stage_decoration",
        ["add", "top_of", "side_of", "run_script", "search_blocks", "render", "lint"],
        critic_focus="lighting, entrance framing, prop scale",
    ),
]
STAGE_BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}
FIX_STAGE = Stage("fix", "stage_detailing", ["set_shape", "move", "add", "add_modifier", "set_material", "define_material"], max_calls=15, critic_focus="applying the critic's fixes")


def stage_for_rule(rule: str) -> Stage:
    """Which stage prompt best applies a critic fix rule (P4/P5 → materials, P7 → decoration, else detailing)."""
    r = (rule or "").upper()
    if r in ("P4", "P5"):
        return STAGE_BY_NAME["materials"]
    if r == "P7":
        return STAGE_BY_NAME["decoration"]
    if r in ("P1", "F"):
        return STAGE_BY_NAME["blocking"]
    return STAGE_BY_NAME["detailing"]


@dataclass
class StageResult:
    stage: str
    text: str
    calls: List[Any]
    lint_text: str = ""
    images: List[Any] = field(default_factory=list)
    render_text: str = ""
    loop: Optional[LoopResult] = None
    stopped_reason: str = "text"
    seconds: float = 0.0

    @property
    def tool_calls(self) -> int:
        return len(self.calls)

    def call_names(self) -> List[str]:
        return [c[0] for c in self.calls]


def stage_user_message(stage: Stage, request: str, brief: Optional[Dict[str, Any]], scene_outline: str, selection: Optional[str] = None, extra: Optional[str] = None, fix_round: bool = False, budget_s: Optional[float] = None) -> str:
    parts = [f"## Player request\n{request.strip()}"]
    if brief:
        parts.append("## Brief\n```json\n" + json.dumps(brief, indent=1) + "\n```")
    parts.append("## Current scene\n```\n" + scene_outline + "\n```")
    if selection:
        parts.append(f"## Scope\nOnly edit objects matching the selection `{selection}` (use select(query=\"{selection}\") to list them). You may add new objects that serve this request; do not touch unrelated parts.")
    if extra:
        parts.append("## Critic findings to address\n" + extra.strip())
    if fix_round:
        parts.append(f"This is a FIX round for the {stage.name} stage: apply the findings above with the fewest ops, render once to confirm, then call finish.")
    else:
        parts.append(
            f"Begin the **{stage.name}** stage now. Work through the checklist with as few round trips as possible: "
            "put all the geometry/material changes for this stage in ONE run_script call, then render and lint together, "
            "fix what is wrong in one more call if needed, and call finish with a one-line summary."
        )
    if budget_s:
        parts.append(f"Time budget for this stage: about {int(budget_s)} s of wall clock; the stage is cut off after that.")
    return "\n\n".join(parts)


def run_stage(
    llm: Any,
    ctx: Any,
    stage: Stage,
    request: str,
    brief: Optional[Dict[str, Any]],
    selection: Optional[str] = None,
    extra: Optional[str] = None,
    max_calls: Optional[int] = None,
    temperature: Optional[float] = None,
    fix_round: bool = False,
    render_after: bool = True,
    lint_after: bool = True,
    deadline: Optional[float] = None,
    op_cap: Optional[int] = None,
) -> StageResult:
    """Run one pipeline stage: compose prompts, run the tool loop, then render + lint for feedback.

    `deadline` (absolute time; defaults to the stage deadline of `ctx.budget`) ends the loop after the
    current tool call; `op_cap` (default COPILOT_OP_CAP / 12) bounds individual op calls per stage.
    """
    session = getattr(ctx, "session", None)
    t0 = time.time()
    if session is not None:
        session.stage = stage.name
        session.tool_calls_this_stage = 0
    cap = env_int("COPILOT_MAX_CALLS_PER_STAGE", 40)
    limit = min(max_calls or stage.max_calls, cap) if max_calls else min(stage.max_calls, cap)
    budget = get_budget(ctx)
    if deadline is None and budget is not None:
        deadline = budget.stage_deadline
    if op_cap is None:
        op_cap = env_int("COPILOT_OP_CAP", DEFAULT_OP_CAP)
    budget_s = (deadline - t0) if deadline else None
    tools = get_tools(ctx, exclude=STAGE_TOOL_EXCLUDE)
    sys_prompt = system_prompt(ctx, stage)
    if stage.encouraged_tools:
        sys_prompt += "\n\nEncouraged tools in this stage: " + ", ".join(stage.encouraged_tools) + "."
    user = stage_user_message(stage, request, brief, outline(ctx), selection, extra, fix_round, budget_s)
    dispatch = get_dispatch(ctx)
    progress = getattr(ctx, "progress", None)
    on_tool = progress.op if progress is not None and getattr(progress, "verbose", False) and hasattr(progress, "op") else None
    loop = run_tool_loop(
        llm,
        ctx,
        sys_prompt,
        [{"role": "user", "content": user}],
        tools,
        dispatch,
        max_calls=limit,
        temperature=stage.temperature if temperature is None else temperature,
        on_tool=on_tool,
        log=getattr(ctx, "log", None),
        deadline=deadline,
        op_cap=op_cap,
    )
    result = StageResult(stage=stage.name, text=loop.text, calls=loop.calls, loop=loop, stopped_reason=loop.stopped_reason)
    t_engine = time.time()
    if render_after:
        r = call_tool(ctx, "render", {"views": ["iso", "front"]})
        result.images = list(getattr(r, "images", None) or [])
        result.render_text = getattr(r, "text", "") or ""
    if lint_after:
        r = call_tool(ctx, "lint", {})
        result.lint_text = getattr(r, "text", "") or ""
    result.seconds = time.time() - t0
    profile_row(
        ctx,
        ("fix:" if fix_round else "") + stage.name,
        wall_ms=int(result.seconds * 1000),
        llm_ms=loop.llm_ms,
        engine_ms=loop.tool_ms + int((time.time() - t_engine) * 1000),
        llm_calls=loop.llm_calls,
        tool_calls=loop.tool_calls,
        stopped=loop.stopped_reason,
        note=f"ops {loop.ops}",
    )
    if session is not None:
        session.stage = None
    return result
