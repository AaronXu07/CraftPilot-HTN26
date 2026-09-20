#!/usr/bin/env python3
"""Generate a machine-readable block list for a Minecraft Java Edition version.

Downloads the vanilla server jar for the requested version, runs the game's own
data generator (``--reports``), and writes ``data/blocks_<version>.json``::

    {
      "version": "26.2",
      "data_version": 4671,
      "blocks": {
        "minecraft:oak_stairs": {
          "properties": {"facing": ["north", "south", "west", "east"], ...}
        },
        ...
      }
    }

Property value lists keep the order the report gives them. Blocks without
properties get an empty ``properties`` object.

Each block additionally carries appearance data derived from the client jar's
block textures::

    "color": [r, g, b],       # mean RGB of the main texture (ints 0-255)
    "noise": 0.42,            # texture busyness, 0..1
    "texture": "minecraft:block/oak_planks"

All three are ``null`` for blocks whose blockstate/model/texture cannot be
resolved (air, some technical blocks).

Usage:
    uv run python scripts/gen_catalog.py            # defaults to 26.2
    uv run python scripts/gen_catalog.py 26.2
    uv run python scripts/gen_catalog.py 26.2 --force        # rerun the generator
    uv run python scripts/gen_catalog.py 26.2 --colors-only  # only redo colour/noise

Reruns are idempotent: the jars and generated reports are cached under
``.cache/`` and reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
DEFAULT_VERSION = "26.2"

# Texture keys tried, in order, to pick a block's "main" texture.
TEXTURE_KEY_ORDER = ["all", "side", "top", "texture", "pattern", "front", "up", "end", "particle"]
# Plains biome tints applied to grayscale, biome-tinted textures.
FOLIAGE_TINT = (0.475, 0.72, 0.30)
GRASS_TINT = (0.57, 0.74, 0.35)
FOLIAGE_NAME_HINTS = ("leaves", "vine", "fern", "lily")
GRASS_NAME_HINTS = ("grass",)
# Blocks with the sanity-check noise values reported after a run.
NOISE_SAMPLE = ["stone_bricks", "cobblestone", "smooth_stone", "white_concrete",
                "oak_planks", "mossy_cobblestone", "sandstone", "deepslate_tiles"]
JAVA_CANDIDATES = [
    os.environ.get("JAVA"),
    "/opt/homebrew/opt/openjdk/bin/java",
    shutil.which("java"),
]

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
DATA = ROOT / "data"


class GenError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[gen_catalog] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- net


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        raise GenError(f"failed to fetch {url}: {e}") from e


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise GenError(f"failed to download {url}: {e}") from e
    tmp.replace(dest)


def find_version_entry(version: str) -> dict:
    manifest = fetch_json(MANIFEST_URL)
    for entry in manifest.get("versions", []):
        if entry.get("id") == version:
            return entry
    latest = manifest.get("latest", {})
    raise GenError(
        f"version {version!r} not found in the Mojang manifest "
        f"(latest release: {latest.get('release')}, snapshot: {latest.get('snapshot')})"
    )


_VERSION_JSON: dict[str, dict] = {}


def version_json(version: str) -> dict:
    """Fetch (once per run) the per-version JSON that lists the download URLs."""
    if version not in _VERSION_JSON:
        log(f"looking up {version} in the version manifest")
        entry = find_version_entry(version)
        _VERSION_JSON[version] = fetch_json(entry["url"])
    return _VERSION_JSON[version]


def ensure_jar(version: str, kind: str) -> Path:
    """Return ``.cache/<kind>-<version>.jar``, downloading it if missing."""
    jar = CACHE / f"{kind}-{version}.jar"
    if jar.exists() and jar.stat().st_size > 0:
        log(f"using cached {jar.relative_to(ROOT)}")
        return jar
    vjson = version_json(version)
    try:
        url = vjson["downloads"][kind]["url"]
    except KeyError as e:
        raise GenError(f"version JSON for {version} has no downloads.{kind}.url") from e
    log(f"downloading {kind} jar -> {jar.relative_to(ROOT)}")
    download(url, jar)
    return jar


def ensure_server_jar(version: str) -> Path:
    return ensure_jar(version, "server")


def ensure_client_jar(version: str) -> Path:
    return ensure_jar(version, "client")


# -------------------------------------------------------------------------- java


def find_java() -> str:
    for cand in JAVA_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    raise GenError("no java binary found; set $JAVA or install openjdk (brew install openjdk)")


def java_major(java: str) -> int:
    out = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)
    text = out.stderr + out.stdout
    log("java -version: " + text.strip().splitlines()[0])
    # first line looks like: openjdk version "21.0.2" 2024-01-16
    m = re.search(r'version "(\d+)(?:\.(\d+))?', text)
    if not m:
        raise GenError(f"could not parse java version from: {text.strip()}")
    major = int(m.group(1))
    if major == 1:  # "1.8.0_xxx"
        major = int(m.group(2) or 0)
    return major


def run_data_generator(java: str, jar: Path, out_dir: Path, force: bool) -> Path:
    report = out_dir / "reports" / "blocks.json"
    if report.exists() and not force:
        log(f"using cached report {report.relative_to(ROOT)}")
        return report
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Work inside the cache dir so any stray files (libraries/, logs/) land there.
    attempts = [
        ("bundler main class", [java, "-DbundlerMainClass=net.minecraft.data.Main", "-jar", str(jar),
                                "--reports", "--output", str(out_dir)]),
        ("plain --reports", [java, "-jar", str(jar), "--reports", "--output", str(out_dir)]),
        ("legacy -cp data.Main", [java, "-cp", str(jar), "net.minecraft.data.Main",
                                  "--reports", "--output", str(out_dir)]),
    ]
    errors = []
    for label, cmd in attempts:
        log(f"running data generator ({label}): {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=CACHE, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and report.exists():
            log(f"data generator succeeded via {label}")
            return report
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        errors.append(f"  {label}: exit {proc.returncode}\n    " + "\n    ".join(tail))
    raise GenError("data generator failed with every invocation:\n" + "\n".join(errors))


def read_data_version(jar: Path) -> int:
    with zipfile.ZipFile(jar) as z:
        if "version.json" not in z.namelist():
            raise GenError(f"{jar.name} has no version.json inside")
        info = json.loads(z.read("version.json"))
    try:
        return int(info["world_version"])
    except (KeyError, ValueError, TypeError) as e:
        raise GenError(f"version.json in {jar.name} has no integer world_version") from e


# ---------------------------------------------------------------- textures/colour


class TextureError(RuntimeError):
    """A single block's appearance could not be resolved (never fatal)."""


def _strip_ns(rid: str) -> str:
    return rid.split(":", 1)[1] if ":" in rid else rid


def _first_dict(value: object) -> dict:
    """Blockstate variant/apply values are a dict or a list of dicts; take the first dict."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        raise TextureError("variant/apply entry is not a dict")
    return value


class TextureResolver:
    """Resolve block id -> model chain -> main texture -> colour/noise, from the client jar."""

    def __init__(self, client_jar: Path) -> None:
        self.zip = zipfile.ZipFile(client_jar)
        self.names = set(self.zip.namelist())
        self._models: dict[str, dict | None] = {}

    def close(self) -> None:
        self.zip.close()

    def _read_json(self, path: str) -> dict | None:
        if path not in self.names:
            return None
        return json.loads(self.zip.read(path))

    # -- blockstate -> model id

    def model_id_for(self, name: str) -> str:
        state = self._read_json(f"assets/minecraft/blockstates/{name}.json")
        if state is None:
            raise TextureError("no blockstate file")
        if "variants" in state:
            variants = state["variants"]
            if not variants:
                raise TextureError("empty variants")
            entry = _first_dict(next(iter(variants.values())))
        elif "multipart" in state:
            parts = state["multipart"]
            if not parts:
                raise TextureError("empty multipart")
            entry = _first_dict(parts[0].get("apply"))
        else:
            raise TextureError("blockstate has neither variants nor multipart")
        model = entry.get("model")
        if not isinstance(model, str):
            raise TextureError("variant has no model")
        return model

    # -- model chain

    def model(self, model_id: str) -> dict | None:
        if model_id not in self._models:
            self._models[model_id] = self._read_json(f"assets/minecraft/models/{_strip_ns(model_id)}.json")
        return self._models[model_id]

    def model_chain(self, model_id: str) -> list[dict]:
        """Model followed by its parents, child first. Missing/builtin parents end the chain."""
        chain: list[dict] = []
        seen: set[str] = set()
        cur: str | None = model_id
        while cur and cur not in seen:
            seen.add(cur)
            m = self.model(cur)
            if m is None:
                if not chain:
                    raise TextureError(f"model {cur} not found")
                break
            chain.append(m)
            cur = m.get("parent")
        return chain

    @staticmethod
    def merged_textures(chain: list[dict]) -> dict[str, str]:
        merged: dict[str, str] = {}
        for m in reversed(chain):  # parent first so the child overrides
            for key, val in (m.get("textures") or {}).items():
                # Newer versions allow {"sprite": "...", "force_translucent": true} (glass).
                if isinstance(val, dict):
                    val = val.get("sprite")
                if isinstance(val, str):
                    merged[key] = val
        # Resolve "#key" references.
        resolved: dict[str, str] = {}
        for key, val in merged.items():
            hops = 0
            while val.startswith("#") and hops < 16:
                val = merged.get(val[1:], "")
                hops += 1
            if val and not val.startswith("#"):
                resolved[key] = val
        return resolved

    @staticmethod
    def pick_texture(textures: dict[str, str]) -> str:
        for key in TEXTURE_KEY_ORDER:
            if key in textures:
                return textures[key]
        for val in textures.values():
            return val
        raise TextureError("model chain has no textures")

    @staticmethod
    def uses_tint(chain: list[dict]) -> bool:
        for m in chain:
            for el in m.get("elements") or []:
                for face in (el.get("faces") or {}).values():
                    if isinstance(face, dict) and "tintindex" in face:
                        return True
        return False

    # -- texture image -> colour/noise

    def frame(self, texture_id: str) -> np.ndarray:
        """First frame of the texture as an (h, w, 4) uint8 RGBA array."""
        path = f"assets/minecraft/textures/{_strip_ns(texture_id)}.png"
        if path not in self.names:
            raise TextureError(f"texture {texture_id} not found")
        with Image.open(io.BytesIO(self.zip.read(path))) as im:
            arr = np.asarray(im.convert("RGBA"))
        h, w = arr.shape[:2]
        if h > w:  # animated strip: keep the first square frame
            arr = arr[:w]
        return arr

    @staticmethod
    def mean_color(arr: np.ndarray) -> np.ndarray:
        rgb = arr[..., :3].astype(np.float64)
        mask = arr[..., 3] > 0
        if not mask.any():
            mask = np.ones(arr.shape[:2], dtype=bool)
        return rgb[mask].mean(axis=0)

    @staticmethod
    def noise(arr: np.ndarray) -> float:
        rgb = arr[..., :3].astype(np.float64)
        lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        std = float(lum.std())
        diffs = np.concatenate([np.abs(np.diff(lum, axis=1)).ravel(), np.abs(np.diff(lum, axis=0)).ravel()])
        hf = float(diffs.mean()) if diffs.size else 0.0
        return min(1.0, (std / 70.0) * 0.5 + (hf / 40.0) * 0.5)

    def appearance(self, block_id: str) -> dict:
        name = _strip_ns(block_id)
        chain = self.model_chain(self.model_id_for(name))
        texture_id = self.pick_texture(self.merged_textures(chain))
        arr = self.frame(texture_id)
        color = self.mean_color(arr)
        if any(h in name for h in GRASS_NAME_HINTS):
            color = color * GRASS_TINT
        elif any(h in name for h in FOLIAGE_NAME_HINTS) or self.uses_tint(chain):
            color = color * FOLIAGE_TINT
        color = np.clip(np.rint(color), 0, 255).astype(int)
        return {"color": [int(c) for c in color], "noise": round(self.noise(arr), 4), "texture": texture_id}


def add_appearance(catalog: dict, client_jar: Path) -> tuple[int, list[tuple[str, str]]]:
    """Fill ``color``/``noise``/``texture`` for every block in place; return (ok_count, failures)."""
    resolver = TextureResolver(client_jar)
    ok = 0
    failures: list[tuple[str, str]] = []
    try:
        for block_id, entry in catalog["blocks"].items():
            try:
                entry.update(resolver.appearance(block_id))
                ok += 1
            except Exception as e:  # noqa: BLE001 - one bad block must never abort the run
                entry.update({"color": None, "noise": None, "texture": None})
                failures.append((block_id, f"{type(e).__name__}: {e}"))
    finally:
        resolver.close()
    return ok, failures


def report_appearance(catalog: dict, ok: int, failures: list[tuple[str, str]]) -> None:
    log(f"appearance: {ok} blocks coloured, {len(failures)} unresolved")
    for block_id, why in failures[:8]:
        log(f"  unresolved {block_id}: {why}")
    blocks = catalog["blocks"]
    for name in NOISE_SAMPLE:
        entry = blocks.get(f"minecraft:{name}")
        if entry is not None:
            log(f"  noise {name:<20} {entry.get('noise')}  color={entry.get('color')}")


# ------------------------------------------------------------------------ output


def build_catalog(version: str, data_version: int, report: Path) -> dict:
    raw = json.loads(report.read_text())
    blocks = {}
    for block_id, entry in raw.items():
        props = entry.get("properties", {})
        blocks[block_id] = {"properties": {k: list(v) for k, v in props.items()}}
    return {"version": version, "data_version": data_version, "blocks": blocks}


def write_catalog(catalog: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", default=DEFAULT_VERSION, help="Minecraft version id (default 26.2)")
    ap.add_argument("--force", action="store_true", help="rerun the data generator even if cached")
    ap.add_argument("--output", type=Path, default=None, help="output path (default data/blocks_<version>.json)")
    ap.add_argument("--colors-only", action="store_true",
                    help="skip the data generator; recompute colour/noise/texture into the existing data file")
    args = ap.parse_args(argv)

    version = args.version
    out_path = args.output or (DATA / f"blocks_{version}.json")

    try:
        CACHE.mkdir(exist_ok=True)
        if args.colors_only:
            if not out_path.exists():
                raise GenError(f"{out_path.relative_to(ROOT)} does not exist; run without --colors-only first")
            catalog = json.loads(out_path.read_text())
            data_version = catalog.get("data_version")
        else:
            java = find_java()
            major = java_major(java)
            if major < 21:
                raise GenError(f"Java {major} found at {java}; Minecraft 26.x needs Java 21 or newer")
            jar = ensure_server_jar(version)
            report = run_data_generator(java, jar, CACHE / f"generated-{version}", args.force)
            data_version = read_data_version(jar)
            catalog = build_catalog(version, data_version, report)
        client_jar = ensure_client_jar(version)
        ok, failures = add_appearance(catalog, client_jar)
        write_catalog(catalog, out_path)
    except GenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report_appearance(catalog, ok, failures)
    log(f"wrote {out_path.relative_to(ROOT)}: data_version={data_version}, blocks={len(catalog['blocks'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
