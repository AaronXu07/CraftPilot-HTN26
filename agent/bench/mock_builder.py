"""A scripted "builder" LLM that plays the whole pipeline without a real model.

Used by `bench/run.py --mock` and by the pipeline tests. It inspects the system prompt to decide which
role it is playing (interpret / stage / critic / router / question) and emits plausible tool calls.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from copilot.llm import LLMResponse, MockLLM, ToolCall, _messages_have_images

DEFAULT_CRITIC = {"score": 9, "top_3_fixes": [], "summary": "Solid massing and materials; nothing blocking."}
DEFAULT_ROUTE = {"intent": "edit", "stages": ["detailing"], "selection": "all", "direct_ops": [], "needs_place": True}


class ScriptedBuilderLLM(MockLLM):
    """MockLLM whose responses depend on the prompt role rather than a fixed script.

    `critic_responses`: queue of critic JSON dicts (popped per critic call; default score 9).
    `routes`: queue of router JSON dicts (popped per router call).
    `stage_ops`: optional override {stage_name: [tool_call dicts]}.
    """

    def __init__(self, critic_responses: Optional[List[Dict[str, Any]]] = None, routes: Optional[List[Dict[str, Any]]] = None, stage_ops: Optional[Dict[str, List[Dict[str, Any]]]] = None, supports_vision: bool = True):
        super().__init__([], supports_vision=supports_vision)
        self.critic_responses = list(critic_responses or [])
        self.routes = list(routes or [])
        self.stage_ops = dict(stage_ops or {})
        self.roles: List[str] = []

    # ------------------------------------------------------------------------------------
    def chat(self, messages, tools=None, temperature=0.2, tool_choice="auto", max_tokens=4000) -> LLMResponse:
        self.calls += 1
        names = [t.get("function", {}).get("name", t.get("name")) for t in (tools or [])]
        self.requests.append({"messages": [dict(m) for m in messages], "tools": names, "temperature": temperature, "tool_choice": tool_choice, "has_images": _messages_have_images(messages)})
        system = _text(messages[0].get("content")) if messages and messages[0].get("role") == "system" else ""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = _text(user_msgs[-1].get("content")) if user_msgs else ""
        if "set_brief" in names and (tool_choice != "auto" or "# Stage 0" in system):
            self.roles.append("interpret")
            return self._calls([{"name": "set_brief", "args": {"brief": self._brief(last_user)}}])
        if system.lstrip().startswith("# Critic"):
            self.roles.append("critic")
            body = self.critic_responses.pop(0) if self.critic_responses else DEFAULT_CRITIC
            return LLMResponse(text=json.dumps(body), tool_calls=[], usage={})
        if system.lstrip().startswith("# Router"):
            self.roles.append("router")
            body = self.routes.pop(0) if self.routes else DEFAULT_ROUTE
            return LLMResponse(text=json.dumps(body), tool_calls=[], usage={})
        if "Answer the player's question" in system:
            self.roles.append("question")
            return LLMResponse(text="The keep is 8 blocks tall and 10 wide.", tool_calls=[], usage={})
        stage = _stage_of(system)
        self.roles.append(stage or "unknown")
        if any(m.get("role") == "tool" for m in messages) or "FIX round" in last_user:
            return self._calls([{"name": "finish", "args": {"summary": f"{stage or 'stage'} done"}}])
        ops = self.stage_ops.get(stage or "") or self._default_ops(stage or "", last_user)
        if not ops:
            return self._calls([{"name": "finish", "args": {"summary": "nothing to do"}}])
        return self._calls(ops)

    # ------------------------------------------------------------------------------------
    def _calls(self, calls: List[Dict[str, Any]]) -> LLMResponse:
        tcs = [ToolCall(id=f"mock_{self.calls}_{i}", name=c["name"], args=dict(c.get("args", {}))) for i, c in enumerate(calls)]
        return LLMResponse(text=None, tool_calls=tcs, usage={})

    @staticmethod
    def _brief(request: str) -> Dict[str, Any]:
        m = re.search(r"Player request:\s*(.*)", request)
        req = (m.group(1) if m else request).strip()
        bt = "structure"
        for w in ("castle", "villa", "pagoda", "lighthouse", "bridge", "cathedral", "temple", "treehouse", "tower", "greenhouse", "house", "hall"):
            if w in req.lower():
                bt = w
                break
        return {
            "name": f"{bt}_v1",
            "build_type": bt,
            "style": "medieval stone",
            "footprint": [16, 12],
            "height": 14,
            "facing": "south",
            "key_features": ["hall", "hip roof", "door"],
            "constraints": ["flat ground"],
            "materials_intent": "stone bricks, dark oak roof",
            "silhouette_plan": "Hall 16x12 8 tall with a hip roof pyramid 18x14 6 tall; door on the south side.",
            "stages": ["blocking", "detailing", "materials", "decoration"],
        }

    @staticmethod
    def _default_ops(stage: str, last_user: str) -> List[Dict[str, Any]]:
        if stage == "blocking":
            return [
                {"name": "define_material", "args": {"name": "stone_wall", "spec": {"base": "stone_bricks"}}},
                {"name": "add", "args": {"id": "hall", "shape": {"type": "box", "size": [16, 8, 12]}, "pos": [0, 0, 0], "material": "stone_wall", "modifiers": [{"type": "shell", "thickness": 1}]}},
                {"name": "add", "args": {"id": "hall_roof", "shape": {"type": "pyramid", "base": [18, 14], "height": 6}, "pos": [0, 8, 0], "material": "roof_main"}},
                {"name": "stack", "args": {"id": "hall_roof", "on": "hall"}},
            ]
        if stage == "detailing":
            return [
                {"name": "add", "args": {"id": "hall_door_cut", "shape": {"type": "box", "size": [2, 3, 3]}, "pos": [0, 0, 6], "op": "subtract"}},
                {"name": "add", "args": {"id": "hall_win_cut", "shape": {"type": "box", "size": [1, 2, 3]}, "pos": [-5, 3, 6], "op": "subtract", "modifiers": [{"type": "array", "count": 3, "offset": [5, 0, 0]}]}},
            ]
        if stage == "materials":
            return [
                {"name": "define_material", "args": {"name": "castle_wall", "spec": {"base": "stone_bricks", "palette": [["stone_bricks", 0.7], ["cracked_stone_bricks", 0.3]], "fit": "stairs+slab"}}},
                {"name": "define_material", "args": {"name": "roof_dark", "spec": {"base": "dark_oak_planks", "fit": "stairs+slab"}}},
                {"name": "set_material", "args": {"ids": "hall", "material": "castle_wall"}},
                {"name": "set_material", "args": {"ids": "hall_roof", "material": "roof_dark"}},
            ]
        if stage == "decoration":
            return [
                {"name": "add", "args": {"id": "door_lantern", "shape": {"type": "block", "state": "lantern[hanging=false]"}, "pos": [-2, 3, 6], "modifiers": [{"type": "array", "count": 2, "offset": [4, 0, 0]}]}},
            ]
        return []


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def _stage_of(system: str) -> Optional[str]:
    for name in ("blocking", "detailing", "materials", "decoration"):
        if re.search(rf"# Stage \d — {name.capitalize()}", system):
            return name
    return None
