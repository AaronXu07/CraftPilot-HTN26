"""Building or object? Routes a build request to the procedural building generator or the image->3D objects path.

Two generators share one chat command. The building generator (llm/compose -> BuildProgram -> engine) is right
for architecture it can express: houses, castles, towers, churches, barns... and anything that comes with
storeys, dimensions or facade features. The objects path (objects/pipeline: reference image -> mesh -> voxels)
is right for anything whose identity is a silhouette: statues, creatures, vehicles, props, and named landmarks
- the Eiffel Tower is a landmark first and a tower second.

`classify()` decides in three layers: an explicit override (the mod's `/build object|building`), word rules
(landmarks, head noun, shape cues, structural cues), and - only when the rules are not confident - one small
cached LLM call. Any failure of that call falls back to the rule result, so the router never raises.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from craftpilot.config import SETTINGS

Kind = Literal["building", "object"]
Source = Literal["override", "rule", "llm", "fallback"]
KINDS = ("building", "object")

LLM_THRESHOLD = 0.8  # rule results below this ask the LLM (when configured)
LANDMARK_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Route:
    kind: Kind
    source: Source
    confidence: float
    reason: str
    matched: list[str] = field(default_factory=list)  # evidence tokens, e.g. landmark:eiffel tower, head:castle=building
    scores: tuple[int, int] = (0, 0)  # (building, object) rule points

    @property
    def note(self) -> str:
        """The one chat / CLI line that says where the request went and why."""
        where = "objects" if self.kind == "object" else "buildings"
        how = {"override": "", "rule": "", "llm": " (LLM)", "fallback": " (rules; LLM unavailable)"}[self.source]
        return f"routed to {where}{how}: {self.reason}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scores"] = list(self.scores)
        d["note"] = self.note
        return d


# ---------------------------------------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------------------------------------
# Named places and famous things: always the objects path, whatever generic words they contain. Longest
# phrase first. Entries that need an article ("the white house") avoid catching the generic phrase.
LANDMARKS = (
    "eiffel tower", "eiffel", "statue of liberty", "lady liberty", "big ben", "tower bridge", "golden gate bridge",
    "golden gate", "brooklyn bridge", "sydney harbour bridge", "taj mahal", "colosseum", "coliseum",
    "pyramids of giza", "great pyramid", "sphinx", "stonehenge", "space needle", "cn tower", "tokyo tower",
    "burj khalifa", "burj al arab", "empire state building", "empire state", "chrysler building", "flatiron",
    "sydney opera house", "mount rushmore", "leaning tower of pisa", "leaning tower", "tower of pisa",
    "arc de triomphe", "christ the redeemer", "parthenon", "acropolis", "pantheon", "notre dame", "notre-dame",
    "sagrada familia", "st basil", "saint basil", "kremlin", "buckingham palace", "the white house",
    "westminster abbey", "london eye", "tower of london", "the shard", "the gherkin", "hagia sophia", "blue mosque",
    "petra", "machu picchu", "chichen itza", "angkor wat", "great wall of china", "forbidden city",
    "neuschwanstein", "mont saint michel", "mont-saint-michel", "hollywood sign", "gateway arch",
    "washington monument", "lincoln memorial", "moai", "easter island head", "atomium", "petronas towers",
    "brandenburg gate", "louvre", "versailles", "st peter's basilica", "st peters basilica", "alhambra",
    "potala palace", "hoover dam", "himeji castle", "edinburgh castle", "windsor castle",
    # famous machines and fiction
    "titanic", "millennium falcon", "x-wing", "tie fighter", "death star", "star destroyer", "tardis", "delorean",
    "batmobile", "saturn v", "space shuttle", "apollo lander", "r2-d2", "r2d2", "bb-8", "wall-e", "optimus prime",
    "hogwarts", "minas tirith", "barad-dur", "barad dur", "helm's deep", "helms deep", "winterfell", "erebor",
    "rivendell", "castle black", "the burrow", "bag end",
)

# What the building generator's part grammar can express (masses with floors, roofs, windows, doors...).
BUILDING_WORDS = (
    "house", "home", "cottage", "cabin", "hut", "shack", "shed", "bungalow", "villa", "mansion", "manor", "estate",
    "palace", "chateau", "castle", "keep", "fortress", "fort", "stronghold", "citadel", "watchtower", "tower",
    "turret", "lookout", "outpost", "bell tower", "clock tower", "wizard tower", "lighthouse", "church", "chapel",
    "cathedral", "abbey", "monastery", "temple", "shrine", "pagoda", "mosque", "barn", "farmhouse", "stable",
    "silo", "granary", "mill", "factory", "factories", "warehouse", "foundry", "workshop", "forge", "smithy", "inn",
    "tavern", "pub", "shop", "store", "bakery", "market", "market hall", "library", "libraries", "school", "hall",
    "town hall", "guild hall", "townhouse", "terrace", "row house", "apartment", "apartment building",
    "office building", "building", "skyscraper", "office", "hotel", "hospital", "prison", "bunker", "garage",
    "barracks", "gatehouse", "guardhouse", "treehouse", "tree house", "greenhouse", "pavilion", "gazebo", "dojo",
    "tea house", "teahouse", "saltbox", "tomb", "mausoleum", "dungeon", "longhouse", "observatory", "bank",
    "station", "cinema", "theatre", "theater", "museum", "mall", "diner", "cafe", "restaurant",
)

# Anything whose identity is a silhouette (plan-1's object list minus the generic words, plus the structures
# the building grammar cannot express).
OBJECT_WORDS = (
    # sculpture
    "statue", "sculpture", "monument", "memorial", "bust", "figurine", "totem", "idol", "effigy", "gargoyle",
    "obelisk", "fountain",
    # structures without floors or roofs
    "bridge", "drawbridge", "footbridge", "suspension bridge", "cable-stayed bridge", "cable stayed bridge",
    "aqueduct", "viaduct", "windmill", "ferris wheel", "roller coaster", "carousel", "pyramid", "arch", "archway",
    "torii", "crane", "water tower", "radio tower", "cell tower", "oil rig", "wind turbine", "lighthouse lamp",
    # creatures and characters
    "creature", "monster", "beast", "dragon", "wyvern", "drake", "phoenix", "griffin", "hydra", "unicorn",
    "pegasus", "animal", "horse", "dog", "puppy", "cat", "kitten", "wolf", "fox", "bear", "lion", "tiger",
    "elephant", "giraffe", "deer", "stag", "moose", "cow", "pig", "sheep", "goat", "chicken", "duck", "rabbit",
    "bunny", "bird", "eagle", "owl", "parrot", "fish", "whale", "shark", "dolphin", "octopus", "squid", "kraken",
    "snake", "serpent", "spider", "scorpion", "turtle", "frog", "dinosaur", "t-rex", "trex", "raptor", "mammoth",
    "gorilla", "monkey", "panda", "penguin", "axolotl", "bee", "goose", "swan", "golem", "robot", "mech", "mecha",
    "android", "cyborg", "alien", "ghost", "skeleton", "zombie", "creeper", "enderman", "ghast", "warden",
    "wither", "blaze", "slime", "villager", "piglin", "allay", "sniffer", "knight", "warrior", "wizard", "witch",
    "king", "queen", "ninja", "samurai", "soldier", "astronaut", "character", "mascot", "pokemon", "pikachu",
    "charizard", "mario", "luigi", "sonic", "kirby", "yoshi", "batman", "superman", "spiderman", "spider-man",
    "iron man", "hulk", "goku", "totoro", "godzilla", "king kong", "steve", "alex", "herobrine", "snowman",
    "scarecrow", "gnome",
    # vehicles
    "vehicle", "car", "truck", "bus", "van", "jeep", "tractor", "tank", "train", "locomotive", "ship", "boat",
    "sailboat", "sailing ship", "pirate ship", "yacht", "galleon", "submarine", "canoe", "kayak", "raft", "plane",
    "airplane", "aeroplane", "jet", "helicopter", "rocket", "spaceship", "starship", "shuttle", "ufo", "airship",
    "blimp", "zeppelin", "hot air balloon", "bike", "bicycle", "motorcycle", "cart", "wagon", "carriage",
    "chariot", "sled", "sleigh",
    # weapons, props, food
    "sword", "axe", "hammer", "shield", "bow", "spear", "trident", "pickaxe", "dagger", "crown", "skull", "heart",
    "chess piece", "pawn", "rook", "mushroom", "flower", "tree", "cactus", "pumpkin", "apple", "pineapple",
    "banana", "strawberry", "cake", "burger", "hamburger", "pizza", "donut", "ice cream", "cupcake", "trophy",
    "logo", "emoji", "teddy bear", "toy", "guitar", "piano", "chair", "throne", "bottle", "mug", "cup", "coin",
    "diamond", "gem", "key", "ring", "dice", "controller", "phone", "rubber duck", "anchor", "cannon", "bell",
)

# A keyword followed by one of these is only a theme ("dragon-themed castle") and scores nothing.
_MODIFIERS = ("themed", "style", "styled", "inspired", "like", "esque")

# Explicit "looks like X" requests: the silhouette is the point, so the objects path (+4).
_SHAPE_RE = re.compile(r"\b(shaped like|in the shape of|shape of a|that looks like|looking like|look-?alike|"
                       r"replica of|model of)\b|\b\w+[- ]shaped\b", re.I)

# Structural evidence for the building generator.
_DIMS_RE = re.compile(r"\b\d{1,3}\s*[x×]\s*\d{1,3}(?:\s*[x×]\s*\d{1,3})?\b", re.I)
_SIZE_RE = re.compile(r"\b\d{1,3}\s*(?:blocks?\s*)?(?:wide|tall|high|deep|long)\b", re.I)
_STOREYS_RE = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[\s-]*"
                         r"(?:stor(?:e)?ys?|stories|floors?|levels?)\b", re.I)
_BOUNDS_WORDS_RE = re.compile(r"\b(?:dimensions?|footprint|bounds|bounding box)\b", re.I)
_FEATURES = (
    "roof", "roofs", "gable", "gabled", "hip roof", "dormer", "dormers", "chimney", "chimneys", "window", "windows",
    "door", "doors", "doorway", "porch", "balcony", "balconies", "facade", "façade", "attic", "basement", "cellar",
    "courtyard", "stairs", "staircase", "interior", "room", "rooms", "bedroom", "kitchen", "hallway", "arrow slit",
    "arrow slits", "battlements", "crenellation", "crenellations", "crenellated", "parapet", "buttress",
    "buttresses", "spire", "steeple", "pillar", "pillars", "column", "columns", "colonnade", "awning", "awnings",
    "shutters", "walls", "foundation", "veranda", "terrace roof",
)
_FEATURE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(_FEATURES, key=len, reverse=True)) + r")\b",
                         re.I)

_IMPERATIVE_RE = re.compile(r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
                            r"(?:build|make|create|spawn|generate|construct|design|place|give)\s+(?:me\s+)?"
                            r"(?:a|an|the|some)?\s*", re.I)
_BOUNDARY_RE = re.compile(r"\b(with|of|on|over|under|near|beside|next to|in|at|for|from|that|which|where|and|but|"
                          r"shaped|looking|like)\b|[,;:.]", re.I)
_PROPER_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b|\bthe\s+[A-Z][a-z]+\b")
_TAG_NOISE = {"red", "run", "down", "new", "bell", "sea", "harbour", "plant", "tea", "row", "wings", "glass",
              "curved", "round", "narrow", "tall", "large", "small", "american", "zen"}


def _phrase_re(words: tuple[str, ...]) -> re.Pattern[str]:
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"\b(" + alts + r")(?:e?s)?\b", re.I)


_LANDMARK_RE = _phrase_re(LANDMARKS)
_BUILDING_RE = _phrase_re(BUILDING_WORDS)
_OBJECT_RE = _phrase_re(OBJECT_WORDS)
_BUILDING_SET = set(BUILDING_WORDS)
_OBJECT_SET = set(OBJECT_WORDS)


def _keyword_kind(word: str) -> Kind | None:
    w = word.lower()
    for cand in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w):
        if cand in _OBJECT_SET:
            return "object"
        if cand in _BUILDING_SET:
            return "building"
    return None


def _keywords(text: str) -> list[tuple[int, int, str, Kind]]:
    """Every building/object keyword in `text` as (start, end, word, kind), longest match wins on overlap."""
    found: list[tuple[int, int, str, Kind]] = []
    for regex in (_OBJECT_RE, _BUILDING_RE):
        for m in regex.finditer(text):
            kind = _keyword_kind(m.group(1))
            if kind is not None:
                found.append((m.start(), m.end(), m.group(1).lower(), kind))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out: list[tuple[int, int, str, Kind]] = []
    last_end = -1
    for start, end, word, kind in found:
        if start < last_end:
            continue  # inside a longer phrase already taken ("suspension bridge" over "bridge")
        out.append((start, end, word, kind))
        last_end = end
    return out


@functools.lru_cache(maxsize=4)
def _exemplar_tags(directory: str) -> list[tuple[str, frozenset[str]]]:
    try:
        from pathlib import Path

        from craftpilot.program.exemplars import load_all

        return [(e.name, frozenset(set(e.tags) - _TAG_NOISE)) for e in load_all(Path(directory))]
    except Exception:  # noqa: BLE001 - evidence only; the router must never fail on the exemplar library
        return []


def _exemplar_evidence(text: str) -> str | None:
    from craftpilot.program.exemplars import tag_score

    words = set(re.findall(r"[a-z]+", text.lower()))
    best: tuple[int, str] | None = None
    for name, tags in _exemplar_tags(str(SETTINGS.exemplars_dir)):
        score = tag_score(words, set(tags))
        if best is None or score > best[0]:
            best = (score, name)
    return best[1] if best and best[0] >= 3 else None


def rule_route(text: str, has_bounds: bool = False) -> Route:
    """The word-rule layer: pure, no network."""
    raw = (text or "").strip()
    if not raw:
        return Route("building", "rule", 0.5, "empty request")
    lower = raw.lower()

    m = _LANDMARK_RE.search(lower)
    if m:
        name = m.group(1)
        return Route("object", "rule", LANDMARK_CONFIDENCE, f'landmark "{name}"', [f"landmark:{name}"], (0, 9))

    norm = _IMPERATIVE_RE.sub("", lower).strip()
    b = o = 0
    matched: list[str] = []
    reasons: list[str] = []

    keywords = _keywords(norm)
    bound = _BOUNDARY_RE.search(norm)
    head_end = bound.start() if bound else len(norm)

    # A keyword directly followed by a theme word ("dragon-themed") or by another keyword ("dragon statue",
    # "wizard tower") is attributive: it describes the head, it is not a second subject.
    def role(i: int) -> str:
        start, end, word, kind = keywords[i]
        tail = norm[end:end + 12]
        if re.match(r"[- ]?(" + "|".join(_MODIFIERS) + r")\b", tail):
            return "modifier"
        if i + 1 < len(keywords) and re.fullmatch(r"[-\s]*", norm[end:keywords[i + 1][0]]):
            return "attributive"
        return "noun"

    roles = [role(i) for i in range(len(keywords))]
    head_idx = None
    for i, (start, end, word, kind) in enumerate(keywords):
        if end <= head_end and roles[i] == "noun":
            head_idx = i
    head_kind: Kind | None = None
    extra_b = extra_o = 0
    for i, (start, end, word, kind) in enumerate(keywords):
        if roles[i] != "noun":
            matched.append(f"word:{word}={kind}({roles[i]})")
            if roles[i] == "attributive" and kind != keywords[i + 1][3]:
                # "pineapple house", "dog house", "wizard tower": mixed compounds are worth a second opinion
                if kind == "building":
                    extra_b += 1
                else:
                    extra_o += 1
            continue
        if i == head_idx:
            head_kind = kind
            matched.append(f"head:{word}={kind}")
            reasons.append(f'"{word}" is {"a building type" if kind == "building" else "an object"}')
            if kind == "building":
                b += 3
            else:
                o += 3
        else:
            matched.append(f"word:{word}={kind}")
            if kind == "building":
                extra_b += 1
            else:
                extra_o += 1
    b += min(2, extra_b)
    o += min(2, extra_o)

    shape = _SHAPE_RE.search(norm)
    if shape:
        o += 4
        matched.append(f"shape:{shape.group(0)}")
        reasons.append("shaped like something (a silhouette request)")

    proper = None
    if raw != lower:  # only when the player used capitals
        for pm in _PROPER_RE.finditer(raw):
            span = pm.group(0)
            if not _keywords(span.lower()) and not _IMPERATIVE_RE.match(span.lower()):
                proper = span
                break
    if proper:
        o += 2
        matched.append(f"proper:{proper}")
        reasons.append(f'"{proper}" reads like a name')

    if head_kind != "object":
        structural = 0
        cues: list[str] = []
        dims = _DIMS_RE.search(norm)
        size = _SIZE_RE.search(norm)
        if dims:
            structural += 2
            cues.append(f"dimensions {dims.group(0)}")
        elif size:
            structural += 1
            cues.append(f"size {size.group(0)}")
        if _BOUNDS_WORDS_RE.search(norm):
            structural += 1
            cues.append("explicit dimensions")
        storeys = _STOREYS_RE.search(norm)
        if storeys:
            structural += 2
            cues.append(f"storeys ({storeys.group(0)})")
        feats = []
        for fm in _FEATURE_RE.finditer(norm):
            f = fm.group(1).lower()
            if f not in feats:
                feats.append(f)
        if feats:
            structural += min(2, len(feats))
            cues.append("features " + ", ".join(feats[:3]))
        if has_bounds:
            structural += 1
            cues.append("a bounding box")
        ex = _exemplar_evidence(norm)
        if ex:
            structural += 1
            cues.append(f"like the {ex} exemplar")
        structural = min(5, structural)
        if structural:
            b += structural
            matched.extend(f"cue:{c}" for c in cues)
            reasons.append("+ " + "; ".join(cues[:2]))

    kind: Kind = "building" if b >= o else "object"
    margin = abs(b - o)
    confidence = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8}.get(margin, 0.9)
    if not reasons:
        reasons.append("no building or object words; the building generator is the default")
    return Route(kind, "rule", confidence, "; ".join(reasons), matched, (b, o))


# ---------------------------------------------------------------------------------------------------------
# LLM tie-break
# ---------------------------------------------------------------------------------------------------------
ROUTE_SYSTEM = """You route a Minecraft player's build request to one of two generators. Reply with JSON only.

building - a procedural ARCHITECTURE generator. It composes rectangular masses with floors, gable/hip/flat/cone/
dome/pagoda roofs, windows, doors, chimneys, dormers, porches, balconies, round corner towers, spires, parapets
with crenellations, and block palettes. Right for houses, cottages, cabins, castles, keeps, forts, churches,
temples, pagodas, barns, factories, warehouses, townhouses, mansions, towers, lighthouses, shops, inns, halls,
skyscrapers - and for any request that gives storeys, dimensions or architectural features. A theme or
decoration does not change this: a dragon-themed castle, a gingerbread house and a haunted mansion are buildings.

object - an image-to-3D SCULPTING path: it draws a picture and turns it into a voxel model, so it reproduces any
silhouette but has no floors, rooms or doors. Right for statues, monuments, creatures, characters, vehicles,
ships, weapons, props, food, plants, and for named real-world landmarks and famous things (Eiffel Tower, Statue
of Liberty, Big Ben, Tower Bridge, Titanic), bridges, windmills, pyramids, obelisks, arches, fountains, and for a
building whose defining feature is a non-architectural shape ("a house shaped like a shoe").

Rules: a named landmark or monument is always object, even when it is a tower, a bridge or a castle. A generic
building with a theme stays building. When the request is only a size or a vibe with no subject, choose
building. confidence is your certainty from 0.5 to 1.0; reason is under 12 words."""

ROUTE_SCHEMA = {
    "type": "json_schema",
    "name": "build_route",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["building", "object"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["kind", "confidence", "reason"],
    },
}


def route_deployment() -> str | None:
    return SETTINGS.route_deployment or SETTINGS.edit_deployment or SETTINGS.compose_deployment


def _llm_available() -> bool:
    return bool(SETTINGS.route_llm and SETTINGS.azure_configured and route_deployment())


def _llm_route(text: str, rule: Route) -> Route:
    try:
        from craftpilot.llm.azure import structured_call

        dep = route_deployment()
        assert dep
        out, meta = structured_call(dep, ROUTE_SYSTEM, [{"role": "user", "content": text}], ROUTE_SCHEMA,
                                    effort="low", cache=True, tag="route", timeout=SETTINGS.route_timeout,
                                    max_retries=0, max_output_tokens=3000)
        j = json.loads(out)
        kind = str(j.get("kind", "")).lower()
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}")
        conf = min(0.99, max(0.5, float(j.get("confidence", 0.7))))
        reason = str(j.get("reason") or "").strip() or "the model's call"
        return Route(kind, "llm", conf, reason, rule.matched + ["llm"], rule.scores)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - the router never raises; the rules answer instead
        return replace(rule, source="fallback", reason=f"{rule.reason}; LLM unavailable ({type(exc).__name__})")


def classify(text: str, override: str | None = None, use_llm: bool | None = None, has_bounds: bool = False) -> Route:
    """Decide the path for `text`. `override` ("building" | "object") wins outright; `use_llm=False` (or an
    unconfigured Azure) keeps it to the word rules; otherwise an unconfident rule result is confirmed by one
    small LLM call."""
    if override in KINDS:
        return Route(override, "override", 1.0, f"requested with /build {override}", ["override"])  # type: ignore[arg-type]
    rule = rule_route(text, has_bounds=has_bounds)
    if rule.confidence >= LLM_THRESHOLD or use_llm is False or not _llm_available():
        return rule
    return _llm_route(text, rule)
