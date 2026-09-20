"""Staged build pipeline: interpret → blocking → detailing → materials → decoration → critic.

Public entry point: `orchestrator.handle_chat(ctx, text, llm=None) -> ChatResult`.
"""
from .common import ToolResultLike, call_tool, get_dispatch, get_tools, outline  # noqa: F401
from .critic import Critique, critique, fixes_text  # noqa: F401
from .interpret import brief_to_text, interpret, normalize_brief  # noqa: F401
from .orchestrator import ChatResult, handle_chat  # noqa: F401
from .router import Route, route  # noqa: F401
from .stages import STAGE_BY_NAME, STAGES, Stage, StageResult, load_prompt, run_stage, system_prompt  # noqa: F401
