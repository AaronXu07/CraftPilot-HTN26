"""Request -> ObjectBrief: the one small LLM job in the object path (image prompt, height, plinth, palette).

Works without an LLM too (regex for the height, "statue" implies a plinth), so the path is testable offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass

from craftpilot.config import SETTINGS
from craftpilot.llm.azure import endpoint_root

PALETTES = ("auto", "stone", "wood", "metal", "colorful")
MIN_HEIGHT, MAX_HEIGHT, DEFAULT_HEIGHT = 6, 96, 40

# Families of full blocks per palette hint (names from objects.voxelize.PALETTE). "auto" = everything. Every
# family carries the neutrals: a wooden ship still has white sails, a bronze bust still has dark eye sockets.
_NEUTRALS = ["white_concrete", "white_wool", "quartz_block", "light_gray_concrete", "gray_concrete", "black_concrete", "black_wool"]
PALETTE_FAMILIES: dict[str, list[str]] = {
    "stone": ["stone", "smooth_stone", "stone_bricks", "cobblestone", "andesite", "polished_andesite", "diorite",
              "polished_diorite", "calcite", "deepslate", "polished_deepslate", "deepslate_bricks", "deepslate_tiles",
              "cobbled_deepslate", "tuff", "polished_tuff", "blackstone", *_NEUTRALS],
    "wood": ["oak_planks", "spruce_planks", "birch_planks", "dark_oak_planks", "jungle_planks", "acacia_planks",
             "mangrove_planks", "oak_log", "spruce_log", "stripped_oak_log", "stripped_spruce_log", "brown_terracotta",
             "brown_concrete", *_NEUTRALS],
    "metal": ["iron_block", "polished_deepslate", "gold_block", "waxed_copper_block", "waxed_exposed_copper",
              "waxed_oxidized_copper", "polished_blackstone", *_NEUTRALS],
}

SYSTEM = """You turn a Minecraft player's request for an OBJECT (statue, creature, character, vehicle, weapon, prop)
into a brief for an image model and a voxeliser. Reply with JSON only.

- subject: one vivid sentence describing exactly what to draw, in the style of an image prompt: the subject,
  its pose, its material/finish and colour, and 2-3 distinctive features. Keep every feature the player named.
  Prefer a static, compact, self-supporting pose (no thin outstretched limbs) and a solid, opaque material.
  The result is built from 1-metre blocks, so ask for BOLD, CHUNKY, simplified forms with smooth surfaces —
  never fine textures (feathers, scales, fur strands, filigree): they turn into noise.
- plinth: true for statues/monuments/busts (a plain rectangular plinth under the figure), false for vehicles,
  weapons, props, animals meant to stand on the ground.
- height: the object's LARGEST dimension in blocks (6-96) — height for a statue, length for a ship or car.
  Bigger reads better: default 40 for a statue, 32 for a creature/character, 36 for a vehicle or ship (its
  length), 30 for a weapon planted in the ground. "big" means 1.5x, "huge"/"giant"/"massive" 2x (cap 96).
  Respect any number the player gives.
- Thin parts do not survive: masts, rigging, ropes, wires, antennas, blades thinner than 1/15 of the object.
  Describe such subjects as a chunky toy-like model with thick simplified parts (a ship: solid thick masts,
  billowing solid sails, no rigging) so they come out as blocks.
- palette: "stone" ONLY when the player asks for a stone/marble/granite statue or monument; otherwise
  "colorful" (the subject has colours — keep them vivid in `subject`), "wood"/"metal" only if the player names
  that material, else "auto". A lightning dragon is "colorful" (electric blue, glowing white), not "metal".
- style: 3-8 words of rendering style for the image model, e.g. "carved granite, weathered", "glossy red paint,
  chrome trim", "smooth marble". Match the player's intent.
- label: 2-3 snake_case words naming the object, e.g. dragon_statue.
- NEVER write a trademarked or copyrighted name in `subject` — no Pokémon, Nintendo/Disney/Marvel/anime
  characters, video-game characters, car makes, brands, logos or real people. The image service rejects
  those words outright. Describe the appearance instead: "Pikachu" -> "a chubby yellow mouse-like creature
  with red circular cheeks, long ears with black tips, a zigzag lightning-bolt tail and big black eyes".
  The `label` may keep the name."""

SCHEMA = {
    "type": "json_schema",
    "name": "object_brief",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subject": {"type": "string"},
            "plinth": {"type": "boolean"},
            "height": {"type": "integer"},
            "palette": {"type": "string", "enum": list(PALETTES)},
            "style": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["subject", "plinth", "height", "palette", "style", "label"],
    },
}


@dataclass
class ObjectBrief:
    subject: str
    plinth: bool
    height: int
    palette: str
    style: str
    label: str
    source: str = "llm"  # llm | fallback
    request: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def allowed_blocks(self) -> list[str] | None:
        # Only a stone statue restricts the block set (its reconstruction colours are noisy greys anyway).
        # Everything else trusts the reference image's colours: a "metal" hint used to turn an electric-blue
        # dragon into grey iron.
        return PALETTE_FAMILIES.get(self.palette) if self.palette == "stone" else None


def object_deployment() -> str | None:
    for var in ("CRAFTPILOT_OBJECT_DEPLOYMENT", "AZURE_OPENAI_EDIT_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT",
                "AZURE_OPENAI_COMPOSE_DEPLOYMENT"):
        v = os.environ.get(var)
        if v:
            return v
    return None


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"a", "an", "the", "build", "make", "me", "of", "with", "and", "on", "in", "please", "big", "giant", "blocks", "tall"}
    words = [w for w in words if w not in stop and not w.isdigit()][:3]
    return "_".join(words) or "object"


def _height_from_text(text: str) -> int | None:
    m = re.search(r"(\d{1,3})\s*(?:blocks?\s*)?(?:tall|high)", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def fallback_brief(text: str) -> ObjectBrief:
    t = text.lower()
    plinth = any(w in t for w in ("statue", "monument", "bust", "sculpture", "memorial"))
    palette = "stone" if any(w in t for w in ("stone", "marble", "granite")) else "auto"
    if any(w in t for w in ("wooden", "wood ", "oak", "spruce")):
        palette = "wood"
    if any(w in t for w in ("iron", "steel", "metal", "chrome", "robot", "mech")):
        palette = "metal"
    height = _height_from_text(text) or (40 if plinth else 32)
    if not _height_from_text(text):
        if any(w in t for w in ("huge", "giant", "massive", "enormous", "colossal")):
            height *= 2
        elif any(w in t for w in ("big", "large", "tall")):
            height = int(height * 1.5)
    subject = re.sub(r"^\s*(please\s+)?((build|make|create|spawn)\s+(me\s+)?)?(a|an|the)\s+", "", text, flags=re.IGNORECASE).strip()
    return ObjectBrief(subject=subject or text, plinth=plinth, height=_clamp(height), palette=palette,
                       style="carved stone, matte" if palette == "stone" else "clear readable silhouette",
                       label=_slug(text), source="fallback", request=text)


def _clamp(h: int) -> int:
    return max(MIN_HEIGHT, min(MAX_HEIGHT, int(h)))


GENERIC_REWRITE = ("The image service rejected the previous subject because it contained a protected name. Rewrite the "
                   "brief so `subject` describes the character/object purely by appearance (shape, colours, features) "
                   "with no names, titles, franchises or brands at all. Keep everything else.")


def compose_brief(text: str, use_llm: bool = True, height: int | None = None, generic: bool = False) -> tuple[ObjectBrief, dict]:
    """Returns (brief, meta). `height` from the caller (CLI flag, wand selection) overrides the LLM. With
    `generic`, the model is told the subject must avoid protected names (retry after a blocklist rejection)."""
    meta: dict = {"seconds": 0.0}
    dep = object_deployment()
    brief: ObjectBrief | None = None
    if use_llm and dep and SETTINGS.azure_endpoint and SETTINGS.azure_api_key:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.azure_api_key, base_url=endpoint_root(SETTINGS.azure_endpoint) + "/openai/v1/",
                        timeout=SETTINGS.llm_timeout, max_retries=1)
        t0 = time.time()
        try:
            instructions = SYSTEM + ("\n\n" + GENERIC_REWRITE if generic else "")
            resp = client.responses.create(model=dep, instructions=instructions, input=[{"role": "user", "content": text}],
                                           text={"format": SCHEMA}, reasoning={"effort": "low"}, max_output_tokens=6000)
            data = json.loads(resp.output_text)
            brief = ObjectBrief(subject=str(data["subject"]), plinth=bool(data["plinth"]), height=_clamp(int(data["height"])),
                                palette=str(data["palette"]) if data.get("palette") in PALETTES else "auto",
                                style=str(data.get("style", "")), label=_slug(str(data.get("label") or text)),
                                source="llm", request=text)
            meta = {"seconds": round(time.time() - t0, 1), "deployment": dep}
        except Exception as exc:  # noqa: BLE001 — the fallback brief keeps the path alive
            meta = {"seconds": round(time.time() - t0, 1), "deployment": dep, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if brief is None:
        brief = fallback_brief(text)
    if height is not None:
        brief.height = _clamp(height)
    return brief, meta
