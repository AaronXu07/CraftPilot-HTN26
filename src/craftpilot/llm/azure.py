"""Azure OpenAI client and a thin call wrapper with logging."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from craftpilot.config import PROJECT_ROOT, SETTINGS

LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = LOG_DIR / "cache"


def endpoint_root(endpoint: str) -> str:
    """Accept the resource root, or any full URL under it, and return the root."""
    root = endpoint.strip().rstrip("/")
    if "/openai" in root:
        root = root.split("/openai")[0]
    return root


def client(timeout: float | None = None, max_retries: int = 1):
    from openai import OpenAI

    if not SETTINGS.azure_configured:
        raise RuntimeError("Azure OpenAI is not configured; set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                           "(and a deployment name) in the environment or .env")
    # Azure's v1 surface speaks the plain OpenAI protocol at <root>/openai/v1/ with the key as bearer token.
    base = endpoint_root(SETTINGS.azure_endpoint) + "/openai/v1/"
    return OpenAI(api_key=SETTINGS.azure_api_key, base_url=base,
                  timeout=SETTINGS.llm_timeout if timeout is None else timeout, max_retries=max_retries)


class DeploymentUnavailable(RuntimeError):
    pass


def _cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


# One model call per cache key at a time: a second caller for the same prompt (the mod's /prepare warm-up
# followed by the real /build a few seconds later) waits for the first and reads its cached answer.
_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()


def _read_cache(cache_path: Path, waited: bool = False) -> tuple[str, dict[str, Any]]:
    data = json.loads(cache_path.read_text())
    return data["output"], {"cached": True, "waited": waited, "request_id": data.get("request_id"), "seconds": 0.0}


def structured_call(deployment: str, instructions: str, messages: list[dict[str, str]], fmt: dict[str, Any],
                    effort: str, cache: bool = True, tag: str = "compose", timeout: float | None = None,
                    max_retries: int | None = None, max_output_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    """Returns (output_text, meta). Caches by (deployment, instructions, messages, effort[, max_output_tokens]).

    ``timeout`` / ``max_retries`` override the client defaults (SETTINGS.llm_timeout, 1 retry) for small calls
    that must not wait out a compose-sized timeout; ``max_output_tokens`` caps hidden reasoning plus the answer."""
    payload = {"deployment": deployment, "instructions": instructions, "messages": messages, "effort": effort,
               "schema": fmt.get("name")}
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    key = _cache_key(payload)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{tag}_{key}.json"
    if cache and cache_path.exists():
        return _read_cache(cache_path)
    leader = False
    if cache:
        with _inflight_lock:
            event = _inflight.get(cache_path.name)
            if event is None:
                event = _inflight[cache_path.name] = threading.Event()
                leader = True
        if not leader:
            event.wait(timeout=(SETTINGS.llm_timeout if timeout is None else timeout) + 30)
            if cache_path.exists():
                return _read_cache(cache_path, waited=True)
            # the first caller failed or timed out: make the call ourselves
    try:
        return _call(deployment, instructions, messages, fmt, effort, tag, timeout, max_retries, max_output_tokens,
                     cache_path)
    finally:
        if leader:
            with _inflight_lock:
                _inflight.pop(cache_path.name, None)
            event.set()


def _call(deployment: str, instructions: str, messages: list[dict[str, str]], fmt: dict[str, Any], effort: str,
          tag: str, timeout: float | None, max_retries: int | None, max_output_tokens: int | None,
          cache_path: Path) -> tuple[str, dict[str, Any]]:
    c = client(timeout=timeout, max_retries=1 if max_retries is None else max_retries)
    extra: dict[str, Any] = {} if max_output_tokens is None else {"max_output_tokens": max_output_tokens}
    t0 = time.time()
    try:
        resp = c.responses.create(
            model=deployment,
            instructions=instructions,
            input=messages,
            text={"format": fmt},
            reasoning={"effort": effort},
            **extra,
        )
    except Exception as exc:
        msg = str(exc)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "llm.jsonl").open("a") as f:
            f.write(json.dumps({"tag": tag, "time": time.time(), "deployment": deployment, "effort": effort,
                                "error": f"{type(exc).__name__}: {msg[:300]}",
                                "seconds": round(time.time() - t0, 2)}) + "\n")
        if "not allowed in this deployment" in msg or "DeploymentNotFound" in msg or "Error code: 404" in msg:
            raise DeploymentUnavailable(f"deployment '{deployment}' rejected the request: {msg[:160]}") from exc
        raise
    seconds = time.time() - t0
    text = resp.output_text
    meta = {"cached": False, "request_id": getattr(resp, "id", None), "seconds": round(seconds, 2),
            "usage": getattr(resp, "usage", None) and resp.usage.model_dump()}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "llm.jsonl").open("a") as f:
        f.write(json.dumps({"tag": tag, "time": time.time(), "deployment": deployment, "effort": effort,
                            "messages": messages, "output": text, "meta": meta}) + "\n")
    # Never cache an empty or cut-off answer (a token cap that ate the reasoning) - it would be returned forever.
    if text and text.strip() and getattr(resp, "status", "completed") in (None, "completed"):
        cache_path.write_text(json.dumps({"output": text, "request_id": meta["request_id"]}))
    else:
        meta["incomplete"] = True
    return text, meta
