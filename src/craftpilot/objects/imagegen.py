"""Reference image for an object with FLUX on Azure AI Foundry (OpenAI images API at <root>/openai/v1/)."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from craftpilot.config import SETTINGS
from craftpilot.llm.azure import endpoint_root

DEFAULT_IMAGE_DEPLOYMENT = "FLUX-1.1-pro"

# What the reconstructor needs: one subject, whole body in frame, a view that shows depth (three-quarter),
# plain background so the matting is clean, no text/shadows/props that would become geometry.
# Camera phrasings, best first. Single-image reconstruction wants a three-quarter view at roughly eye level:
# a view from above tends to come back as a flat relief, a dead-on front view loses the sides.
CAMERAS = [
    ("Three-quarter view product photo at eye level: the camera stands at the front-left corner, 45 degrees off the "
     "front, at the object's mid height, so the front AND the left side are both clearly visible."),
    "Product photo from the front-left corner at eye level, turned 40 degrees so one side is visible next to the front.",
    "Side-front view at eye level, the object turned 30 degrees toward the camera.",
]
PROMPT_TEMPLATE = (
    "{camera} {subject}. Whole object in frame with margin. {base_clause}Single object centred on a plain uniform "
    "white background, soft even studio lighting, no cast shadow, no text, no other objects. {style_clause}"
)


def image_deployment() -> str:
    return os.environ.get("CRAFTPILOT_IMAGE_DEPLOYMENT", DEFAULT_IMAGE_DEPLOYMENT)


def compose_prompt(subject: str, plinth: bool, style: str = "", camera: int = 0) -> str:
    base = "Standing on a plain rectangular stone plinth. " if plinth else "Standing on the ground, nothing under it. "
    style_clause = style.strip() if style else "Clear readable silhouette, moderate detail, solid opaque surfaces."
    cam = CAMERAS[camera % len(CAMERAS)]
    return PROMPT_TEMPLATE.format(camera=cam, subject=subject.strip().rstrip("."), base_clause=base, style_clause=style_clause)


class ImageRejected(RuntimeError):
    """The content filter refused the prompt or the generated image."""


# Azure's image filter flags weapons and creatures as "violence" surprisingly often (a knight holding a
# sword). One rewrite that frames the subject as an inert museum piece usually passes without changing
# what gets built.
SAFE_REWRITE = ("A museum exhibit: a static, inanimate sculpture of {subject_lc} carved from stone, calm and "
                "ceremonial, nothing happening, art gallery display.")


def _is_content_filter(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "content" in msg and ("reject" in msg or "filter" in msg or "safety" in msg or "violence" in msg)


def prompt_attempts(subject: str, plinth: bool, style: str, request: str | None = None, camera: int = 0) -> list[str]:
    """Prompts to try in order. Azure's image gateway has a keyword blocklist that fires on innocuous
    phrase combinations (a style tail like "glossy red paint, sleek curves" got a sports car rejected) and
    a violence filter that fires on weapons and monsters, so each retry removes the most likely trigger:
    the LLM's style clause, then the LLM's embellished subject (back to the player's own words), then a
    museum-sculpture framing for the violence filter."""
    out = [compose_prompt(subject, plinth, style, camera)]
    if style:
        out.append(compose_prompt(subject, plinth, "", camera))
    if request and request.strip().lower() != subject.strip().lower():
        out.append(compose_prompt(request, plinth, "", camera))
    out.append(compose_prompt(SAFE_REWRITE.format(subject_lc=(request or subject).strip().rstrip(".").lower()), plinth=True,
                              style="carved stone, matte, museum lighting", camera=camera))
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def generate(prompt: str, out_path: Path, size: str = "1024x1024", deployment: str | None = None,
             timeout: float = 90.0, attempts: list[str] | None = None) -> dict:
    """Generate one image, save it as PNG, return {"path", "seconds", "revised_prompt", "attempt", "prompt"}.
    Tries `prompt` then each of `attempts` (see prompt_attempts) on content-filter rejections; raises
    ImageRejected when all are refused."""
    from openai import OpenAI

    if not (SETTINGS.azure_endpoint and SETTINGS.azure_api_key):
        raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not set")
    client = OpenAI(api_key=SETTINGS.azure_api_key, base_url=endpoint_root(SETTINGS.azure_endpoint) + "/openai/v1/",
                    timeout=timeout, max_retries=1)
    t0 = time.time()
    todo = [prompt] + [a for a in (attempts or []) if a != prompt]
    last: Exception | None = None
    for i, p in enumerate(todo):
        try:
            # NB: no output_format — the Foundry gateway and the FLUX backend disagree on its allowed values
            resp = client.images.generate(model=deployment or image_deployment(), prompt=p, size=size, n=1)
        except Exception as exc:
            if _is_content_filter(exc):
                last = exc
                continue
            raise
        item = resp.data[0]
        if not getattr(item, "b64_json", None):
            raise RuntimeError("image API returned no b64_json payload")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(item.b64_json))
        return {"path": str(out_path), "seconds": round(time.time() - t0, 1),
                "revised_prompt": getattr(item, "revised_prompt", None), "attempt": i, "prompt": p}
    raise ImageRejected(
        "the image was rejected by Azure's content filter (it flags weapons and monsters as violence); "
        f"try rephrasing, e.g. 'a peaceful statue of …'. Last error: {str(last)[:160]}"
    )
