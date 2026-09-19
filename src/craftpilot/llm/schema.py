"""Turn the pydantic schema into the strict JSON schema OpenAI structured outputs accept."""

from __future__ import annotations

import copy
from typing import Any

from craftpilot.program.model import BuildProgram

_DROP_KEYS = {"default", "title", "examples"}


def _strictify(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_strictify(n, defs) for n in node]
    if not isinstance(node, dict):
        return node
    # A $ref may not carry sibling keywords in strict mode. Inline the definition so the
    # field description is kept for the model.
    if "$ref" in node and len([k for k in node if k not in _DROP_KEYS]) > 1:
        name = node["$ref"].rsplit("/", 1)[-1]
        target = copy.deepcopy(defs.get(name, {}))
        for k, v in node.items():
            if k != "$ref" and k not in _DROP_KEYS:
                target[k] = v
        node = target
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in _DROP_KEYS:
            continue
        out[k] = _strictify(v, defs)
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


def build_program_schema() -> dict[str, Any]:
    raw = BuildProgram.model_json_schema()
    defs = raw.get("$defs", {})
    return _strictify(raw, defs)


def response_format(name: str = "build_program") -> dict[str, Any]:
    return {"type": "json_schema", "name": name, "schema": build_program_schema(), "strict": True}
