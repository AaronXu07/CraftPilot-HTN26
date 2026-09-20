"""Configuration from environment variables and a .env file in the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SCHEMATICS_DIR = Path.home() / "Library" / "Application Support" / "minecraft" / "schematics" / "craftpilot"


@dataclass
class Settings:
    schematics_dir: Path
    exemplars_dir: Path
    mc_data_version: int
    safety_limit: tuple[int, int, int]
    default_bounds: tuple[int, int, int]
    azure_endpoint: str | None
    azure_api_key: str | None
    azure_api_version: str
    compose_deployment: str | None
    edit_deployment: str | None
    llm_timeout: float
    service_port: int
    mod_url: str
    place_gap: int
    place_sink: int
    place_chunk: int
    place_chunk_max: int
    place_seconds: float
    place_delay_ms: int
    place_terrain: bool

    @property
    def llm_configured(self) -> bool:
        return bool(self.azure_endpoint and self.azure_api_key and self.compose_deployment)


def _path(value: str | None, default: Path) -> Path:
    """Paths from the environment may use ~ and quotes, which the shell would expand but Python does not."""
    if not value:
        return default
    return Path(value.strip().strip('"').strip("'")).expanduser().resolve()


def _triple(value: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not value:
        return default
    parts = [int(p) for p in value.lower().replace("x", " ").split()]
    if len(parts) != 3:
        return default
    return (parts[0], parts[1], parts[2])


def load_settings() -> Settings:
    return Settings(
        schematics_dir=_path(os.environ.get("CRAFTPILOT_SCHEMATICS_DIR"), DEFAULT_SCHEMATICS_DIR),
        exemplars_dir=_path(os.environ.get("CRAFTPILOT_EXEMPLARS_DIR"), PROJECT_ROOT / "exemplars"),
        # 4903 = Minecraft 26.2 (see data/blocks_26.2.json). Litematica upgrades older data versions on load;
        # use 3578 for a 1.20.2 world.
        mc_data_version=int(os.environ.get("CRAFTPILOT_MC_DATA_VERSION", "4903")),
        safety_limit=_triple(os.environ.get("CRAFTPILOT_SAFETY_LIMIT"), (256, 256, 256)),
        default_bounds=_triple(os.environ.get("CRAFTPILOT_DEFAULT_BOUNDS"), (25, 30, 25)),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "preview"),
        compose_deployment=os.environ.get("AZURE_OPENAI_COMPOSE_DEPLOYMENT"),
        edit_deployment=os.environ.get("AZURE_OPENAI_EDIT_DEPLOYMENT"),
        llm_timeout=float(os.environ.get("CRAFTPILOT_LLM_TIMEOUT", "60")),
        # The Fabric mod's bridge listens on 7777, so the service takes the next port.
        service_port=int(os.environ.get("CRAFTPILOT_PORT", "7778")),
        mod_url=os.environ.get("CRAFTPILOT_MOD_URL", "http://127.0.0.1:7777").rstrip("/"),
        # In-game placement: air gap between the player and the building, how many blocks the plinth
        # sinks below the player's feet, blocks per tick-chunk, and the pause between chunks.
        place_gap=int(os.environ.get("CRAFTPILOT_PLACE_GAP", "2")),
        place_sink=int(os.environ.get("CRAFTPILOT_PLACE_SINK", "0")),
        place_chunk=int(os.environ.get("CRAFTPILOT_PLACE_CHUNK", "1500")),
        # Big builds get fatter chunks so they land in about place_seconds, capped per tick by chunk_max.
        place_chunk_max=int(os.environ.get("CRAFTPILOT_PLACE_CHUNK_MAX", "6000")),
        place_seconds=float(os.environ.get("CRAFTPILOT_PLACE_SECONDS", "6")),
        place_delay_ms=int(os.environ.get("CRAFTPILOT_PLACE_DELAY_MS", "60")),
        # Seat builds on the terrain (mod /heightmap survey, foundation, graded apron). Off = player's feet.
        place_terrain=os.environ.get("CRAFTPILOT_PLACE_TERRAIN", "1").lower() not in ("0", "false", "no", "off"),
    )


SETTINGS = load_settings()
