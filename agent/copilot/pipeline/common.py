"""Helpers shared by the pipeline modules: tool dispatch, tool schemas, scene outline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..jobs import JobCancelled, check_cancel


@dataclass
class ToolResultLike:
    """Duck-typed stand-in for tools.dispatch.ToolResult (used when a call fails or dispatch is absent)."""

    text: str
    images: List[Any] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True


def get_dispatch(ctx: Any) -> Callable[[Any, str, Dict[str, Any]], Any]:
    """`ctx.dispatch` if the context carries an override (tests), else tools.dispatch.dispatch."""
    d = getattr(ctx, "dispatch", None)
    if callable(d):
        return d
    from ..tools.dispatch import dispatch  # lazy: Track 2a

    return dispatch


def call_tool(ctx: Any, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
    """Dispatch one tool call from pipeline code (not the model). Never raises; errors become text."""
    args = args or {}
    check_cancel(ctx)
    try:
        fn = get_dispatch(ctx)
    except Exception as e:  # noqa: BLE001
        return ToolResultLike(text=f"ERROR: tool dispatch unavailable ({e})", ok=False)
    try:
        r = fn(ctx, name, args)
    except JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001
        return ToolResultLike(text=f"ERROR in {name}: {e}", ok=False)
    if r is None:
        return ToolResultLike(text="")
    if not hasattr(r, "text"):
        return ToolResultLike(text=str(r))
    return r


def get_tools(ctx: Any, names: Optional[List[str]] = None, exclude: Optional[set] = None) -> Optional[List[Dict[str, Any]]]:
    """Tool schemas (OpenAI function format). `ctx.tools` overrides; None if schemas are unavailable."""
    schemas = getattr(ctx, "tools", None)
    if schemas is None:
        try:
            from ..tools import schemas as mod  # lazy: Track 2a

            if names and hasattr(mod, "tool_subset"):
                schemas = mod.tool_subset(names)
            else:
                schemas = list(mod.TOOL_SCHEMAS)
        except Exception:  # noqa: BLE001
            return None
    out = []
    for t in schemas:
        n = _tool_name(t)
        if names and n not in names:
            continue
        if exclude and n in exclude:
            continue
        out.append(t)
    return out


def _tool_name(t: Dict[str, Any]) -> str:
    if "function" in t and isinstance(t["function"], dict):
        return str(t["function"].get("name"))
    return str(t.get("name"))


def tool_names(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [_tool_name(t) for t in (tools or [])]


def outline(ctx: Any, ids: Any = None, detail: str = "outline") -> str:
    """Current scene outline (what the model reads)."""
    session = getattr(ctx, "session", None)
    scene = getattr(session, "scene", None)
    if scene is None:
        return "(no scene)"
    try:
        return scene.describe(ids, detail)
    except Exception as e:  # noqa: BLE001
        return f"(outline unavailable: {e})"


def fast_llm(llm: Any) -> Any:
    """The LLM's cheaper variant (`llm.fast()`, e.g. MODEL_FAST) for router/describe/fix passes; else the LLM itself."""
    f = getattr(llm, "fast", None)
    if callable(f):
        try:
            return f() or llm
        except Exception:  # noqa: BLE001
            return llm
    return llm


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")
