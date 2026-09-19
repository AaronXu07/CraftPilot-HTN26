"""Staged build pipeline: interpret → blocking → detailing → materials → decoration → critic.

Public entry point: `orchestrator.handle_chat(ctx, text, llm=None) -> ChatResult`.
"""
from .common import ToolResultLike, call_tool, get_dispatch, get_tools, outline  # noqa: F401
from .stages import STAGES, STAGE_BY_NAME, Stage, StageResult, run_stage, system_prompt, load_prompt  # noqa: F401
from .interpret import interpret, normalize_brief, brief_to_text  # noqa: F401
from .critic import Critique, critique, fixes_text  # noqa: F401
from .router import Route, route  # noqa: F401
from .orchestrator import ChatResult, handle_chat  # noqa: F401
