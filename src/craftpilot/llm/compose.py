"""Natural language -> BuildProgram, with exemplar retrieval and an offline fallback."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from craftpilot.blocks.catalog import catalog_summary
from craftpilot.config import SETTINGS
from craftpilot.llm.fallback import fallback_program
from craftpilot.llm.schema import response_format
from craftpilot.program.exemplars import Exemplar, load_all, retrieve
from craftpilot.program.model import Bounds, BuildProgram

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def system_prompt() -> str:
    from craftpilot.program.palette import library_summary

    return _PROMPT_PATH.read_text().replace("{catalog}", catalog_summary()).replace("{palettes}", library_summary())


def _few_shot(exemplars: list[Exemplar]) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    for e in exemplars:
        msgs.append({"role": "user", "content": e.description})
        msgs.append({"role": "assistant", "content": e.program.model_dump_json()})
    return msgs


def _user_message(text: str, bounds_hint: Bounds | None, footprint_hint: tuple[int, int] | None = None) -> str:
    if bounds_hint is not None:
        return (f"{text}\n\nThe player selected a bounding box of {bounds_hint.width} wide, "
                f"{bounds_hint.height} tall, {bounds_hint.depth} deep. Set bounds to exactly that and size the parts for it.")
    if footprint_hint is not None:
        w, d = footprint_hint
        return (f"{text}\n\nThe player marked the base area on the ground: {w} wide along the front and {d} deep. "
                f"Set bounds.width to {w} and bounds.depth to {d} exactly, size the parts to fill that footprint, "
                f"and choose whatever height suits the building.")
    return text


def compose(text: str, use_llm: bool = True, bounds_hint: Bounds | None = None,
            k: int = 4, footprint_hint: tuple[int, int] | None = None) -> tuple[BuildProgram, str, list[str]]:
    """Returns (program, source, notes). source is 'llm', 'llm-cache', or 'fallback'.

    ``bounds_hint`` fixes the whole box; ``footprint_hint`` (width, depth) fixes the base area and leaves
    the height to the model (or the exemplar's own height in the fallback)."""
    exemplars = load_all(SETTINGS.exemplars_dir)
    notes: list[str] = []
    if use_llm and SETTINGS.llm_configured:
        try:
            from craftpilot.llm.azure import DeploymentUnavailable, structured_call

            shots = retrieve(exemplars, text, k)
            messages = _few_shot(shots) + [{"role": "user", "content": _user_message(text, bounds_hint, footprint_hint)}]
            deployments = [SETTINGS.compose_deployment]
            if SETTINGS.edit_deployment and SETTINGS.edit_deployment != SETTINGS.compose_deployment:
                deployments.append(SETTINGS.edit_deployment)
            out = meta = None
            used = None
            for dep in deployments:
                try:
                    out, meta = structured_call(dep, system_prompt(), messages, response_format(),
                                                effort="medium", tag="compose")
                    used = dep
                    break
                except DeploymentUnavailable as exc:
                    notes.append(f"{exc}; trying the next deployment.")
            if out is None:
                raise RuntimeError("no usable Azure deployment")
            program = BuildProgram.model_validate_json(out)
            source = "llm-cache" if meta.get("cached") else "llm"
            if meta.get("seconds"):
                notes.append(f"Composed by {used} in {meta['seconds']}s.")
            return program, source, notes
        except (ValidationError, json.JSONDecodeError) as exc:
            notes.append(f"LLM output did not validate ({type(exc).__name__}); used the offline fallback.")
        except Exception as exc:  # network, auth, timeout
            notes.append(f"LLM call failed ({type(exc).__name__}: {str(exc)[:120]}); used the offline fallback.")
    elif use_llm:
        notes.append("Azure OpenAI is not configured; used the offline fallback.")
    program, fb_notes = fallback_program(text, exemplars)
    if bounds_hint is not None:
        program.bounds = bounds_hint
    elif footprint_hint is not None:
        program.bounds = Bounds(width=footprint_hint[0], height=program.bounds.height, depth=footprint_hint[1])
    return program, "fallback", notes + fb_notes


def edit(previous: BuildProgram, text: str) -> tuple[BuildProgram, str, list[str]]:
    """Patch a previous program with an edit request such as 'make it taller'."""
    notes: list[str] = []
    deployment = SETTINGS.edit_deployment or SETTINGS.compose_deployment
    if not SETTINGS.llm_configured or not deployment:
        return previous, "fallback", ["Azure OpenAI is not configured; edit ignored."]
    try:
        from craftpilot.llm.azure import structured_call

        messages = [
            {"role": "user", "content": "Here is the current BuildProgram:\n" + previous.model_dump_json()},
            {"role": "assistant", "content": "Understood. Tell me the change."},
            {"role": "user", "content": f"Apply this change and return the full updated program: {text}"},
        ]
        out, meta = structured_call(deployment, system_prompt(), messages, response_format(), effort="low",
                                    cache=False, tag="edit")
        return BuildProgram.model_validate_json(out), "llm", notes
    except Exception as exc:
        return previous, "fallback", [f"Edit failed ({type(exc).__name__}); kept the previous program."]
