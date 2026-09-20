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


_EFFORTS = ("minimal", "low", "medium", "high")


def _efforts(first: str) -> list[str]:
    """The configured effort, then one level up as the retry for an answer that failed validation."""
    if first not in _EFFORTS:
        first = "low"
    i = _EFFORTS.index(first)
    return list(dict.fromkeys([first, _EFFORTS[min(i + 1, len(_EFFORTS) - 1)]]))


def _user_message(text: str, bounds_hint: Bounds | None) -> str:
    if bounds_hint is not None:
        return (f"{text}\n\nThe player selected a bounding box of {bounds_hint.width} wide, "
                f"{bounds_hint.height} tall, {bounds_hint.depth} deep. Set bounds to exactly that and size the parts for it.")
    return text


def compose(text: str, use_llm: bool = True, bounds_hint: Bounds | None = None,
            k: int = 4) -> tuple[BuildProgram, str, list[str]]:
    """Returns (program, source, notes). source is 'llm', 'llm-cache', or 'fallback'."""
    exemplars = load_all(SETTINGS.exemplars_dir)
    notes: list[str] = []
    if use_llm and SETTINGS.llm_configured:
        try:
            from craftpilot.llm.azure import DeploymentUnavailable, structured_call

            shots = retrieve(exemplars, text, k)
            messages = _few_shot(shots) + [{"role": "user", "content": _user_message(text, bounds_hint)}]
            deployments = [SETTINGS.compose_deployment]
            if SETTINGS.edit_deployment and SETTINGS.edit_deployment != SETTINGS.compose_deployment:
                deployments.append(SETTINGS.edit_deployment)
            program = None
            for effort in _efforts(SETTINGS.compose_effort):
                out = meta = None
                used = None
                for dep in deployments:
                    try:
                        out, meta = structured_call(dep, system_prompt(), messages, response_format(),
                                                    effort=effort, tag="compose")
                        used = dep
                        break
                    except DeploymentUnavailable as exc:
                        notes.append(f"{exc}; trying the next deployment.")
                if out is None:
                    raise RuntimeError("no usable Azure deployment")
                try:
                    program = BuildProgram.model_validate_json(out)
                except (ValidationError, json.JSONDecodeError) as exc:
                    # a cut-off or malformed answer: think harder once before giving up on the model
                    notes.append(f"LLM output at effort {effort} did not validate ({type(exc).__name__}).")
                    continue
                break
            if program is None:
                notes.append("Used the offline fallback.")
            else:
                source = "llm-cache" if meta.get("cached") else "llm"
                if meta.get("seconds"):
                    notes.append(f"Composed by {used} in {meta['seconds']}s (effort {effort}).")
                return program, source, notes
        except Exception as exc:  # network, auth, timeout
            notes.append(f"LLM call failed ({type(exc).__name__}: {str(exc)[:120]}); used the offline fallback.")
    elif use_llm:
        notes.append("Azure OpenAI is not configured; used the offline fallback.")
    program, fb_notes = fallback_program(text, exemplars)
    if bounds_hint is not None:
        program.bounds = bounds_hint
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
