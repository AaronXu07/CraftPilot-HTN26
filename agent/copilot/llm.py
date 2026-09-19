"""LLM layer: Azure OpenAI client, scripted mock, tool-calling loop, JSON helpers, run logging.

See plan.md §7.6. The model is an OpenAI chat model on Azure with function calling; vision is used
by the critic when the deployment supports it (`LLM.supports_vision`).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .jobs import JobCancelled, check_cancel

try:  # optional: .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass


# ----------------------------------------------------------------------------------------------
# Data types
# ----------------------------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class LLMResponse:
    text: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    """Anything with a `chat` method and a `supports_vision` flag."""

    supports_vision: bool

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        tool_choice: Any = "auto",
        max_tokens: int = 4000,
    ) -> LLMResponse: ...


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
IMAGE_MAX_PX = 640  # T2: every image sent to the model is at most 640 px on its long side
IMAGES_PER_CALL = 2  # ...and no call carries more than two images
JPEG_QUALITY = 82


def image_content(pil_image, detail: str = "high", max_px: int = IMAGE_MAX_PX, fmt: str = "jpeg") -> Dict[str, Any]:
    """A chat-completions `image_url` content part: JPEG (default) or PNG data URI, downscaled to <= max_px."""
    img = pil_image
    w, h = img.size
    if max(w, h) > max_px:
        s = max_px / float(max(w, h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO()
    if fmt.lower() in ("jpg", "jpeg"):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        mime = "image/jpeg"
    else:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}


def image_parts(images: Sequence[Any], limit: int = IMAGES_PER_CALL) -> List[Dict[str, Any]]:
    """Content parts for the last `limit` images (the most recent renders are the ones that matter)."""
    keep = list(images)[-limit:] if limit else []
    return [image_content(im) for im in keep]


def text_content(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_args(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments as a dict; tolerant of bad JSON from the model."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"value": v}
        except json.JSONDecodeError:
            v = extract_json(raw)
            return v if isinstance(v, dict) else {"_raw": raw}
    return {"value": raw}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: Any) -> Optional[Any]:
    """Best-effort JSON extraction from model text: raw JSON, fenced block, or first balanced {...}/[...]."""
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    s = str(text).strip()
    if not s:
        return None
    candidates: List[str] = [s]
    candidates += [m.strip() for m in _FENCE_RE.findall(s)]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(s[start : i + 1])
                        break
            start = s.find(opener, start + 1)
            if len(candidates) > 12:
                break
    for c in candidates:
        for attempt in (c, _clean_json(c), _balance(_clean_json(c))):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _balance(s: str) -> str:
    """Append missing closers for truncated JSON (e.g. a cut-off tool argument string)."""
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    return s + "".join(reversed(stack))


def _clean_json(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"(?m)^\s*//.*$", "", s)  # line comments
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return s


class JsonlLogger:
    """Append-only JSON-lines logger (one file per turn: runs/<session>/<turn>.jsonl)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, **rec: Any) -> None:
        rec.setdefault("t", time.time())
        self.records.append(rec)
        if not self.path:
            return
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        except OSError:
            pass

    def event(self, name: str, **fields: Any) -> None:
        self.log(event=name, **fields)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)[:200]


def _preview(s: Any, n: int = 200) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


# ----------------------------------------------------------------------------------------------
# Azure OpenAI
# ----------------------------------------------------------------------------------------------
class AzureLLM:
    """Azure OpenAI chat completions with function calling and (optional) vision.

    Env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_VISION_DEPLOYMENT (defaults to the chat deployment), AZURE_OPENAI_API_VERSION
    (default 2024-10-21), AZURE_OPENAI_VISION ("0" disables image input), AZURE_OPENAI_TIMEOUT_S
    (per call, default 120) and AZURE_OPENAI_MAX_ATTEMPTS (default 2 = one retry on timeout/429/5xx).
    The SDK's own retries are disabled so one `chat()` never exceeds attempts x timeout.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        vision_deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("MODEL", "")
        self.vision_deployment = vision_deployment or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT") or self.deployment
        # T2: a cheaper deployment for the router / describe / fix passes (falls back to the main one)
        self.fast_deployment = os.environ.get("MODEL_FAST") or os.environ.get("AZURE_OPENAI_FAST_DEPLOYMENT") or self.deployment
        self.is_fast = False
        self.supports_vision = os.environ.get("AZURE_OPENAI_VISION", "1").lower() not in ("0", "false", "no")
        # "chat" (chat.completions), "responses" (Responses API; required by some newer deployments such as
        # gpt-5.x on Azure AI Foundry) or "auto": responses when the endpoint URL ends in /responses, else chat,
        # switching to responses automatically if the deployment rejects chat completions.
        self.api = (os.environ.get("AZURE_OPENAI_API", "auto") or "auto").lower()
        if self.api == "auto":
            self.api = "responses" if self.endpoint.rstrip("/").endswith("/responses") else "chat"
        self.reasoning_effort = os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "low")
        self.timeout = float(timeout if timeout is not None else os.environ.get("AZURE_OPENAI_TIMEOUT_S", 120.0))
        self.max_attempts = max(1, int(max_attempts if max_attempts is not None else os.environ.get("AZURE_OPENAI_MAX_ATTEMPTS", 2)))
        # 429s are short and frequent on small deployments: allow a few more (cheap) attempts than for timeouts
        self.rate_limit_attempts = max(self.max_attempts, int(os.environ.get("AZURE_OPENAI_RATE_LIMIT_ATTEMPTS", 4)))
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0
        self._client = None
        if not (self.endpoint and self.api_key and self.deployment):
            raise RuntimeError(
                "Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_DEPLOYMENT (or COPILOT_LLM=mock for tests)"
            )

    @property
    def client(self):
        """Lazily build the client. Accepts either a bare resource endpoint
        (https://<res>.openai.azure.com/ → classic AzureOpenAI client with api-version) or any URL containing
        /openai/v1 (e.g. a pasted .../openai/v1/responses URL → Azure's v1 API via the plain OpenAI client,
        no api-version, deployment name as `model`). AZURE_OPENAI_USE_V1=1 forces the v1 form."""
        if self._client is None:
            from urllib.parse import urlparse

            from openai import AzureOpenAI, OpenAI

            u = urlparse(self.endpoint if "://" in self.endpoint else "https://" + self.endpoint)
            root = f"{u.scheme}://{u.netloc}"
            use_v1 = "/openai/v1" in (u.path or "") or os.environ.get("AZURE_OPENAI_USE_V1", "").lower() in ("1", "true", "yes")
            if use_v1:
                self._client = OpenAI(base_url=f"{root}/openai/v1/", api_key=self.api_key, timeout=self.timeout, max_retries=0)
            else:
                self._client = AzureOpenAI(
                    azure_endpoint=root + "/", api_key=self.api_key, api_version=self.api_version, timeout=self.timeout, max_retries=0
                )
        return self._client

    def fast(self) -> "AzureLLM":
        """A view of this client that uses `fast_deployment` (shares the HTTP client and usage counters)."""
        if self.is_fast or self.fast_deployment == self.deployment:
            return self
        import copy

        _ = self.client  # build once so both views share it
        f = copy.copy(self)
        f.deployment = self.fast_deployment
        f.is_fast = True
        return f

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """gpt-5.x and o-series deployments reject `temperature` and want `max_completion_tokens`."""
        m = (model or "").lower()
        return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        tool_choice: Any = "auto",
        max_tokens: int = 4000,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """One completion. `timeout` (s) overrides the client default for this call (T2: the pipeline
        passes what is left of the turn's hard budget)."""
        import openai

        has_images = _messages_have_images(messages)
        if has_images and not self.supports_vision:
            messages = strip_images(messages)
            has_images = False
        model = self.vision_deployment if has_images else self.deployment
        if self.api == "responses":
            return self._chat_via_responses(messages, tools, temperature, tool_choice, max_tokens, model, has_images, timeout)
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if timeout is not None:
            kwargs["timeout"] = float(timeout)
        if self._is_reasoning_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = True
        delay = 2.0
        last_err: Optional[Exception] = None
        attempts = self.max_attempts
        attempt = 0
        while attempt < attempts:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                return self._parse(resp)
            except openai.BadRequestError as e:
                if has_images and "image" in str(e).lower():
                    # deployment without vision: degrade gracefully and remember it
                    self.supports_vision = False
                    kwargs["messages"] = strip_images(messages)
                    kwargs["model"] = self.deployment
                    has_images = False
                    continue
                if "parallel_tool_calls" in str(e) and "parallel_tool_calls" in kwargs:
                    kwargs.pop("parallel_tool_calls")
                    continue
                msg = str(e)
                if "not allowed in this deployment" in msg.lower() or "operation is not allowed" in msg.lower():
                    # Responses-only deployment (e.g. gpt-5.x on Foundry): switch APIs and remember it
                    self.api = "responses"
                    return self._chat_via_responses(messages, tools, temperature, tool_choice, max_tokens, model, has_images, timeout)
                if "max_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
                delay = max(delay, _retry_after_seconds(e))
                if isinstance(e, openai.RateLimitError):
                    attempts = self.rate_limit_attempts
            except openai.APIStatusError as e:
                if e.status_code and e.status_code >= 500:
                    last_err = e
                else:
                    raise
            if attempt < attempts:
                time.sleep(min(delay, timeout or 60.0))
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"Azure OpenAI failed after {attempt} attempts: {last_err}")

    # -- Responses API ------------------------------------------------------------------------
    def _chat_via_responses(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        tool_choice: Any,
        max_tokens: int,
        model: str,
        has_images: bool,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Same contract as chat(), over `client.responses.create` (chat-style history converted)."""
        import openai

        instructions, items = messages_to_responses_input(messages)
        kwargs: Dict[str, Any] = {"model": model, "input": items, "max_output_tokens": max_tokens}
        if timeout is not None:
            kwargs["timeout"] = float(timeout)
        if instructions:
            kwargs["instructions"] = instructions
        if self._is_reasoning_model(model):
            if self.reasoning_effort and self.reasoning_effort != "none":
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools_to_responses(tools)
            kwargs["tool_choice"] = tool_choice_to_responses(tool_choice)
            kwargs["parallel_tool_calls"] = True
        delay = 2.0
        last_err: Optional[Exception] = None
        attempts = self.max_attempts
        attempt = 0
        while attempt < attempts:
            attempt += 1
            try:
                resp = self.client.responses.create(**kwargs)
                self.calls += 1
                return self._parse_responses(resp)
            except openai.BadRequestError as e:
                msg = str(e)
                if has_images and "image" in msg.lower():
                    self.supports_vision = False
                    _, kwargs["input"] = messages_to_responses_input(strip_images(messages))
                    kwargs["model"] = self.deployment
                    has_images = False
                    continue
                for key in ("parallel_tool_calls", "reasoning", "temperature"):
                    if key in msg and key in kwargs:
                        kwargs.pop(key)
                        break
                else:
                    raise
                continue
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
                delay = max(delay, _retry_after_seconds(e))
                if isinstance(e, openai.RateLimitError):
                    attempts = self.rate_limit_attempts
            except openai.APIStatusError as e:
                if e.status_code and e.status_code >= 500:
                    last_err = e
                else:
                    raise
            if attempt < attempts:
                time.sleep(min(delay, timeout or 60.0))
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"Azure OpenAI (responses) failed after {attempt} attempts: {last_err}")

    def _parse_responses(self, resp: Any) -> LLMResponse:
        calls: List[ToolCall] = []
        texts: List[str] = []
        for i, item in enumerate(getattr(resp, "output", None) or []):
            t = getattr(item, "type", None)
            if t == "function_call":
                calls.append(ToolCall(id=getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{i}", name=item.name, args=parse_args(item.arguments)))
            elif t == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                        texts.append(part.text)
        usage: Dict[str, int] = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v
        text = "\n".join(texts) if texts else None
        return LLMResponse(text=text, tool_calls=calls, usage=usage, raw=resp)

    def _parse(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        calls: List[ToolCall] = []
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=fn.name, args=parse_args(fn.arguments)))
        usage: Dict[str, int] = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v
        return LLMResponse(text=msg.content, tool_calls=calls, usage=usage, raw=resp)


def _retry_after_seconds(e: Any) -> float:
    """Seconds to wait from a Retry-After header (or the seconds mentioned in the message), else 0."""
    try:
        resp = getattr(e, "response", None)
        ra = resp.headers.get("retry-after") if resp is not None else None
        if ra:
            return float(ra)
    except Exception:  # noqa: BLE001
        pass
    import re as _re

    m = _re.search(r"retry after (\d+) second", str(e), _re.I)
    return float(m.group(1)) if m else 0.0


def messages_to_responses_input(messages: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert chat-completions style history into (instructions, Responses API input items).

    system → instructions; user/assistant text → role messages (image_url parts → input_image);
    assistant tool_calls → function_call items; tool results → function_call_output items.
    """
    instructions: List[str] = []
    items: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                instructions.append(content)
            elif isinstance(content, list):
                instructions.append("\n".join(p.get("text", "") for p in content if isinstance(p, dict)))
            continue
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": str(m.get("tool_call_id", "")), "output": _content_to_text(content)})
            continue
        if role == "assistant":
            if content:
                items.append({"role": "assistant", "content": _content_to_text(content)})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, default=_json_default)
                items.append({"type": "function_call", "call_id": str(tc.get("id", "")), "name": fn.get("name", ""), "arguments": args})
            continue
        # user (and anything else) --------------------------------------------------------
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "image_url":
                    iu = p.get("image_url") or {}
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    part: Dict[str, Any] = {"type": "input_image", "image_url": url}
                    if isinstance(iu, dict) and iu.get("detail"):
                        part["detail"] = iu["detail"]
                    parts.append(part)
                elif p.get("type") in ("text", "input_text"):
                    parts.append({"type": "input_text", "text": p.get("text", "")})
            items.append({"role": "user", "content": parts})
        else:
            items.append({"role": "user", "content": str(content or "")})
    return "\n\n".join(instructions), items


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def tools_to_responses(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chat-completions function tools → Responses API flat function tools."""
    out: List[Dict[str, Any]] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                "strict": bool(fn.get("strict", False)),
            })
        else:
            out.append(t)
    return out


def tool_choice_to_responses(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict) and "function" in tool_choice:
        return {"type": "function", "name": tool_choice["function"].get("name")}
    return tool_choice


def _messages_have_images(messages: Sequence[Dict[str, Any]]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            return True
    return False


def strip_images(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace image parts with a text note (for deployments without vision)."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            n_img = 0
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    n_img += 1
                else:
                    parts.append(p)
            if n_img:
                parts.append(text_content(f"[{n_img} image(s) omitted: this model has no vision; rely on the text summaries]"))
            m = dict(m)
            m["content"] = parts
        out.append(m)
    return out


# ----------------------------------------------------------------------------------------------
# Scripted mock
# ----------------------------------------------------------------------------------------------
class MockLLM:
    """Returns scripted responses in order and records every request.

    Script steps: `{"tool_calls": [{"name": ..., "args": {...}}, ...]}`, `{"text": "..."}`, a step with
    both, or a callable `(messages, tools) -> step`. When the script is exhausted the mock returns a
    plain text response (so loops terminate).
    """

    def __init__(self, script: Optional[List[Any]] = None, supports_vision: bool = True, exhausted_text: str = "(mock script exhausted)"):
        self.script: List[Any] = list(script or [])
        self.supports_vision = supports_vision
        self.exhausted_text = exhausted_text
        self.requests: List[Dict[str, Any]] = []
        self.calls = 0
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._i = 0

    def extend(self, steps: List[Any]) -> None:
        self.script.extend(steps)

    @property
    def remaining(self) -> int:
        return max(0, len(self.script) - self._i)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        tool_choice: Any = "auto",
        max_tokens: int = 4000,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        self.calls += 1
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": [t.get("function", {}).get("name", t.get("name")) for t in (tools or [])],
                "temperature": temperature,
                "tool_choice": tool_choice,
                "has_images": _messages_have_images(messages),
                "timeout": timeout,
                "deployment": getattr(self, "deployment", None),
            }
        )
        if self._i >= len(self.script):
            return LLMResponse(text=self.exhausted_text, tool_calls=[], usage={})
        step = self.script[self._i]
        self._i += 1
        if callable(step):
            step = step(messages, tools)
        if isinstance(step, str):
            step = {"text": step}
        calls = []
        for j, tc in enumerate(step.get("tool_calls", []) or []):
            calls.append(ToolCall(id=tc.get("id") or f"mock_call_{self.calls}_{j}", name=tc["name"], args=dict(tc.get("args", {}))))
        usage = {"prompt_tokens": sum(approx_tokens(str(m.get("content", ""))) for m in messages), "completion_tokens": 50}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return LLMResponse(text=step.get("text"), tool_calls=calls, usage=usage, raw=step)


# ----------------------------------------------------------------------------------------------
# Tool loop
# ----------------------------------------------------------------------------------------------
@dataclass
class LoopResult:
    text: str
    calls: List[Tuple[str, Dict[str, Any], str]] = field(default_factory=list)
    images_shown: int = 0
    stopped_reason: str = "text"  # text | finish | max_calls | budget | error
    messages: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    llm_ms: int = 0
    tool_ms: int = 0
    ops: int = 0  # individual (non-script) scene-mutating calls

    @property
    def tool_calls(self) -> int:
        return len(self.calls)

    def call_names(self) -> List[str]:
        return [c[0] for c in self.calls]


# tools that never change the scene: a run of these in one response is executed concurrently
READ_ONLY_TOOLS = frozenset({"select", "describe", "bbox", "measure", "top_of", "side_of", "list_materials", "render", "lint", "search_blocks", "nearest_block", "get_player", "materials_list"})
# calls that do not count against the per-stage op cap (batching, feedback, history, chat)
OP_CAP_EXEMPT = READ_ONLY_TOOLS | frozenset({"run_script", "finish", "say", "set_brief", "undo", "redo", "snapshot", "restore"})
DEFAULT_OP_CAP = 12
PARALLEL_WORKERS = 4


def get_logger(ctx: Any) -> Any:
    """ctx.log if it has .log(**rec), else a JsonlLogger under runs/<session>/<turn>.jsonl."""
    log = getattr(ctx, "log", None)
    if log is not None and (hasattr(log, "log") or callable(log)):
        return log
    session = getattr(ctx, "session", None)
    run_dir = getattr(ctx, "run_dir", None) or getattr(session, "run_dir", None) or "runs/anon"
    turn = getattr(session, "turn", 0)
    logger = JsonlLogger(os.path.join(run_dir, f"{turn}.jsonl"))
    try:
        ctx.log = logger
    except Exception:  # noqa: BLE001
        pass
    return logger


def _emit(log: Any, **rec: Any) -> None:
    if log is None:
        return
    try:
        if hasattr(log, "log"):
            log.log(**rec)
        elif callable(log):
            log(rec)
    except Exception:  # noqa: BLE001
        pass


_TIMEOUT_OK: Dict[type, bool] = {}


def _accepts_timeout(llm: Any) -> bool:
    t = type(llm)
    if t not in _TIMEOUT_OK:
        try:
            import inspect

            _TIMEOUT_OK[t] = "timeout" in inspect.signature(llm.chat).parameters
        except (TypeError, ValueError):
            _TIMEOUT_OK[t] = False
    return _TIMEOUT_OK[t]


def llm_call(llm: Any, ctx: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.2, tool_choice: Any = "auto", max_tokens: int = 4000) -> LLMResponse:
    """`llm.chat` with the per-call timeout capped to what is left of the turn's budget (ctx.budget)."""
    budget = getattr(ctx, "budget", None)
    if budget is not None and hasattr(budget, "llm_timeout") and _accepts_timeout(llm):
        default = getattr(llm, "timeout", None) or 120.0
        return llm.chat(messages, tools, temperature, tool_choice, max_tokens, timeout=budget.llm_timeout(default))
    return llm.chat(messages, tools, temperature, tool_choice, max_tokens)


def _exec_tool(ctx: Any, dispatch_fn: Callable[[Any, str, Dict[str, Any]], Any], tc: ToolCall) -> Tuple[str, List[Any], int]:
    """Run one tool call; never raises except JobCancelled. Returns (text, images, ms)."""
    check_cancel(ctx)  # a cancel lands before the next tool call, even mid-batch
    t1 = time.time()
    try:
        tr = dispatch_fn(ctx, tc.name, tc.args)
        text = getattr(tr, "text", None)
        if text is None:
            text = str(tr)
        images = list(getattr(tr, "images", None) or [])
    except JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001
        text = f"ERROR in {tc.name}: {e}"
        images = []
    return str(text), images, int((time.time() - t1) * 1000)


def _exec_batch(ctx: Any, dispatch_fn: Callable[[Any, str, Dict[str, Any]], Any], calls: List[ToolCall]) -> List[Tuple[str, List[Any], int]]:
    """Execute the tool calls of one response in order; runs of read-only calls go concurrently (T2)."""
    out: List[Tuple[str, List[Any], int]] = []
    i = 0
    while i < len(calls):
        j = i
        while j < len(calls) and calls[j].name in READ_ONLY_TOOLS:
            j += 1
        if j - i >= 2:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, j - i), thread_name_prefix="copilot-tool") as pool:
                out.extend(pool.map(lambda tc: _exec_tool(ctx, dispatch_fn, tc), calls[i:j]))
            i = j
        else:
            out.append(_exec_tool(ctx, dispatch_fn, calls[i]))
            i += 1
    return out


def run_tool_loop(
    llm: Any,
    ctx: Any,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    dispatch_fn: Callable[[Any, str, Dict[str, Any]], Any],
    max_calls: int = 40,
    temperature: float = 0.2,
    on_tool: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
    log: Any = None,
    tool_choice: Any = "auto",
    max_llm_calls: Optional[int] = None,
    deadline: Optional[float] = None,
    op_cap: Optional[int] = DEFAULT_OP_CAP,
) -> LoopResult:
    """OpenAI tool-calling loop.

    * All tool calls in one response are executed in order (runs of read-only calls concurrently);
      results are appended as `tool` messages.
    * Images returned by tools (ToolResult.images) are attached to the next user message when the
      model supports vision (at most IMAGES_PER_CALL, 640 px JPEG), otherwise only the tool text is used.
    * Stops on a plain text response, on a `finish` tool call, when `max_calls` tool calls have
      been made (a system nudge is inserted when 5 calls remain), or — after the current tool
      call — once `deadline` (absolute time) has passed (stopped_reason "budget").
    * `op_cap` individual scene-mutating calls per loop; beyond it the model is told to batch the
      rest in one `run_script`.
    * Every call is logged with timing as {t, stage, name, args, result_preview, ms}.
    """
    log = log or get_logger(ctx)
    session = getattr(ctx, "session", None)
    stage = getattr(session, "stage", None)
    full: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}] + list(messages)
    result = LoopResult(text="", messages=full)
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pending_images: List[Any] = []
    nudged = False
    time_nudged = False
    n_calls = 0
    max_llm = max_llm_calls or (max_calls * 2 + 4)
    supports_vision = bool(getattr(llm, "supports_vision", False))
    last_text = ""
    loop_t0 = time.time()

    def over_deadline() -> bool:
        return deadline is not None and time.time() >= deadline

    while True:
        if result.llm_calls >= max_llm:
            result.stopped_reason = "max_calls"
            break
        if over_deadline():
            result.stopped_reason = "budget"
            result.text = last_text or "stopped: stage time budget reached"
            break
        check_cancel(ctx)
        if deadline is not None and not time_nudged:
            left = deadline - time.time()
            if left <= max(10.0, 0.25 * (deadline - loop_t0)):
                time_nudged = True
                full.append({"role": "system", "content": f"About {int(left)} s left in this stage. Make the remaining essential changes in ONE call (a run_script if several ops), then call finish."})
        t0 = time.time()
        try:
            resp: LLMResponse = llm_call(llm, ctx, full, tools, temperature, tool_choice)
        except JobCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            _emit(log, stage=stage, name="__llm_error__", args={}, result_preview=_preview(e), ms=int((time.time() - t0) * 1000))
            result.stopped_reason = "error"
            result.text = last_text or f"LLM error: {e}"
            break
        ms = int((time.time() - t0) * 1000)
        result.llm_calls += 1
        result.llm_ms += ms
        for k, v in (resp.usage or {}).items():
            usage[k] = usage.get(k, 0) + int(v or 0)
        _emit(log, stage=stage, name="__llm__", args={"tools": len(tools or []), "model": getattr(llm, "deployment", None)}, result_preview=_preview(resp.text or f"{len(resp.tool_calls)} tool calls"), ms=ms)
        if resp.text:
            last_text = resp.text
        if not resp.tool_calls:
            result.text = resp.text or ""
            result.stopped_reason = "text"
            full.append({"role": "assistant", "content": resp.text or ""})
            break
        # assistant message with tool calls (OpenAI format)
        full.append(
            {
                "role": "assistant",
                "content": resp.text,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.args, default=_json_default)}}
                    for tc in resp.tool_calls
                ],
            }
        )
        finished = False
        to_run: List[ToolCall] = []
        results: Dict[str, Tuple[str, List[Any], int]] = {}
        for tc in resp.tool_calls:
            n_calls += 1
            if session is not None:
                try:
                    session.tool_calls_this_stage = getattr(session, "tool_calls_this_stage", 0) + 1
                except Exception:  # noqa: BLE001
                    pass
            if tc.name == "finish":
                text = str(tc.args.get("summary") or tc.args.get("text") or "done")
                results[tc.id] = (text, [], 0)
                result.text = text
                finished = True
                continue
            if tc.name not in OP_CAP_EXEMPT:
                result.ops += 1
                if op_cap and result.ops > op_cap:
                    results[tc.id] = (f"ERROR: op cap reached ({op_cap} individual ops this stage). Batch the remaining edits in ONE run_script(python=...) call (scene.add / scene.set_shape / ... inside the script).", [], 0)
                    continue
            to_run.append(tc)
        check_cancel(ctx)
        for tc, res in zip(to_run, _exec_batch(ctx, dispatch_fn, to_run)):
            results[tc.id] = res
        for tc in resp.tool_calls:
            text, images, ms = results.get(tc.id, ("", [], 0))
            if tc.name == "finish":
                full.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                result.calls.append((tc.name, tc.args, text))
                _emit(log, stage=stage, name="finish", args=tc.args, result_preview=_preview(text), ms=0)
            else:
                result.tool_ms += ms
                _emit(log, stage=stage, name=tc.name, args=tc.args, result_preview=_preview(text), ms=ms)
                full.append({"role": "tool", "tool_call_id": tc.id, "content": str(text)})
                result.calls.append((tc.name, tc.args, str(text)))
                if images:
                    pending_images.extend(images)
            if on_tool:
                on_tool(tc.name, tc.args, text)
        if finished:
            result.stopped_reason = "finish"
            break
        if pending_images:
            if supports_vision:
                shown = pending_images[-IMAGES_PER_CALL:]
                parts: List[Dict[str, Any]] = [text_content(f"Rendered view(s) from your last render call ({len(shown)} image(s)). Check silhouette, proportions and detail before continuing.")]
                parts += image_parts(shown)
                full.append({"role": "user", "content": parts})
                result.images_shown += len(shown)
            pending_images = []
        if over_deadline():
            result.stopped_reason = "budget"
            result.text = last_text or f"stopped after {n_calls} tool calls: stage time budget reached"
            break
        remaining = max_calls - n_calls
        if remaining <= 0:
            result.stopped_reason = "max_calls"
            result.text = last_text or f"stopped after {n_calls} tool calls"
            break
        if remaining <= 5 and not nudged:
            nudged = True
            full.append({"role": "system", "content": f"You have {remaining} tool calls left in this stage. Finish the essentials and call finish with a one-line summary."})
    result.usage = usage
    result.messages = full
    return result


def single_call(llm: Any, system_prompt: str, user_content: Any, temperature: float = 0.2, tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Any = "auto", max_tokens: int = 2000, ctx: Any = None) -> LLMResponse:
    """One chat call with a system prompt and a user message (string or content parts).
    With `ctx`, the call's timeout is capped to the turn's remaining budget."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    if ctx is not None:
        return llm_call(llm, ctx, messages, tools, temperature, tool_choice, max_tokens)
    return llm.chat(messages, tools, temperature, tool_choice, max_tokens)


def make_llm(kind: Optional[str] = None) -> Any:
    """Factory: COPILOT_LLM=mock → MockLLM([]), else AzureLLM()."""
    kind = (kind or os.environ.get("COPILOT_LLM", "azure")).lower()
    if kind == "mock":
        try:
            from bench.mock_builder import ScriptedBuilderLLM  # role-aware scripted builder

            return ScriptedBuilderLLM()
        except Exception:  # noqa: BLE001
            return MockLLM([])
    return AzureLLM()
