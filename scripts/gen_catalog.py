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

Usage:
    uv run python scripts/gen_catalog.py            # defaults to 26.2
    uv run python scripts/gen_catalog.py 26.2
    uv run python scripts/gen_catalog.py 26.2 --force   # rerun the generator

Reruns are idempotent: the jar and generated reports are cached under
``.cache/`` and reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
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

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
DEFAULT_VERSION = "26.2"
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


def ensure_server_jar(version: str) -> Path:
    jar = CACHE / f"server-{version}.jar"
    if jar.exists() and jar.stat().st_size > 0:
        log(f"using cached {jar.relative_to(ROOT)}")
        return jar
    log(f"looking up {version} in the version manifest")
    entry = find_version_entry(version)
    vjson = fetch_json(entry["url"])
    try:
        server_url = vjson["downloads"]["server"]["url"]
    except KeyError as e:
        raise GenError(f"version JSON for {version} has no downloads.server.url") from e
    log(f"downloading server jar -> {jar.relative_to(ROOT)}")
    download(server_url, jar)
    return jar


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


# ------------------------------------------------------------------------ output


def build_catalog(version: str, data_version: int, report: Path) -> dict:
    raw = json.loads(report.read_text())
    blocks = {}
    for block_id, entry in raw.items():
        props = entry.get("properties", {})
        blocks[block_id] = {"properties": {k: list(v) for k, v in props.items()}}
    return {"version": version, "data_version": data_version, "blocks": blocks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", default=DEFAULT_VERSION, help="Minecraft version id (default 26.2)")
    ap.add_argument("--force", action="store_true", help="rerun the data generator even if cached")
    ap.add_argument("--output", type=Path, default=None, help="output path (default data/blocks_<version>.json)")
    args = ap.parse_args(argv)

    version = args.version
    out_path = args.output or (DATA / f"blocks_{version}.json")

    try:
        CACHE.mkdir(exist_ok=True)
        java = find_java()
        major = java_major(java)
        if major < 21:
            raise GenError(f"Java {major} found at {java}; Minecraft 26.x needs Java 21 or newer")
        jar = ensure_server_jar(version)
        report = run_data_generator(java, jar, CACHE / f"generated-{version}", args.force)
        data_version = read_data_version(jar)
        catalog = build_catalog(version, data_version, report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(catalog, indent=1) + "\n")
    except GenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    log(f"wrote {out_path.relative_to(ROOT)}: data_version={data_version}, blocks={len(catalog['blocks'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
