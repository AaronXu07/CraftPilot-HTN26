"""Bridge to the Fabric mod (CONTRACTS.md §7).

`Bridge` is the protocol every world backend implements; `HttpBridge` talks to the real mod over
HTTP, `mock_mod.MockBridge` keeps an in-memory world. `get_bridge()` picks one from the env.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import httpx

from .engine.coretypes import BlockMap, IVec3

DEFAULT_MOD_URL = "http://127.0.0.1:7777"
SCAN_CHUNK = 64  # /scan requests are split into <= 64x64x64 boxes
BRIDGE_TIMEOUT_S = 30.0  # cap for every mod call (health/player/say use shorter ones)

SetblockChunk = Tuple[List[Tuple[int, int, int, str]], int]  # (blocks, delay_ms)


class BridgeError(RuntimeError):
    """Raised when the mod is unreachable or answers with an error."""


@runtime_checkable
class Bridge(Protocol):
    """World backend protocol. All coordinates are absolute world coordinates."""

    def health(self) -> Dict[str, Any]:
        """GET /health -> {ok, mod_version, mc_version, client_jar, world_loaded, player}."""
        ...

    def player(self) -> Dict[str, Any]:
        """GET /player -> {name, pos, yaw, pitch, facing, looking_at, dimension}."""
        ...

    def scan(self, lo: Sequence[int], hi: Sequence[int]) -> BlockMap:
        """Inclusive box scan. Returns every position including air ("minecraft:air")."""
        ...

    def setblocks(self, chunks: List[SetblockChunk], flags: int = 3) -> int:
        """Place blocks; `chunks` = [(blocks, delay_ms)], blocks = [(x, y, z, state)]. Returns count queued."""
        ...

    def say(self, text: str) -> None:
        """Print a line in the player's chat."""
        ...

    def blocks(self) -> List[Dict[str, Any]]:
        """GET /blocks -> [{id, properties: {name: [values]}, default: state}]."""
        ...

    def camera(self, **kw: Any) -> Dict[str, Any]:
        """POST /camera (orbit / return)."""
        ...


class HttpBridge:
    """HTTP client for the mod's 7 endpoints. JSON exactly per CONTRACTS.md §7."""

    def __init__(self, base_url: str = DEFAULT_MOD_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- helpers ----------------------------------------------------------------------------
    def _get(self, path: str, timeout: float) -> Any:
        try:
            r = self._client.get(path, timeout=timeout)
        except httpx.HTTPError as e:
            raise BridgeError(f"mod unreachable at {self.base_url}{path}: {e}") from e
        if r.status_code >= 400:
            raise BridgeError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, body: Dict[str, Any], timeout: float) -> Any:
        try:
            r = self._client.post(path, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            raise BridgeError(f"mod unreachable at {self.base_url}{path}: {e}") from e
        if r.status_code >= 400:
            raise BridgeError(f"POST {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    # -- endpoints --------------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._get("/health", timeout=3.0)

    def player(self) -> Dict[str, Any]:
        return self._get("/player", timeout=5.0)

    def scan(self, lo: Sequence[int], hi: Sequence[int]) -> BlockMap:
        lo_i = [int(min(a, b)) for a, b in zip(lo, hi)]
        hi_i = [int(max(a, b)) for a, b in zip(lo, hi)]
        out: BlockMap = {}
        for box_lo, box_hi in split_boxes(lo_i, hi_i, SCAN_CHUNK):
            data = self._post("/scan", {"min": list(box_lo), "max": list(box_hi)}, timeout=BRIDGE_TIMEOUT_S)
            out.update(decode_scan(data))
        return out

    def setblocks(self, chunks: List[SetblockChunk], flags: int = 3) -> int:
        body = {"chunks": [{"blocks": [[int(x), int(y), int(z), str(s)] for (x, y, z, s) in blocks], "delay_ms": int(delay)} for blocks, delay in chunks], "flags": int(flags)}
        n = sum(len(b) for b, _ in chunks)
        if n == 0:
            return 0
        data = self._post("/setblocks", body, timeout=BRIDGE_TIMEOUT_S)
        return int(data.get("queued", n))

    def say(self, text: str) -> None:
        self._post("/say", {"text": str(text)}, timeout=5.0)

    def blocks(self) -> List[Dict[str, Any]]:
        data = self._get("/blocks", timeout=BRIDGE_TIMEOUT_S)
        return list(data.get("blocks", []))

    def camera(self, **kw: Any) -> Dict[str, Any]:
        return self._post("/camera", dict(kw), timeout=10.0)

    def close(self) -> None:
        self._client.close()


def split_boxes(lo: Sequence[int], hi: Sequence[int], size: int) -> List[Tuple[IVec3, IVec3]]:
    """Split an inclusive box into inclusive sub-boxes of at most `size` per axis."""
    boxes: List[Tuple[IVec3, IVec3]] = []
    xs = list(range(lo[0], hi[0] + 1, size))
    ys = list(range(lo[1], hi[1] + 1, size))
    zs = list(range(lo[2], hi[2] + 1, size))
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                boxes.append(((x0, y0, z0), (min(x0 + size - 1, hi[0]), min(y0 + size - 1, hi[1]), min(z0 + size - 1, hi[2]))))
    return boxes


def decode_scan(data: Dict[str, Any]) -> BlockMap:
    """Decode a /scan response ({palette, blocks:[[x,y,z,idx]]}) into a BlockMap including air."""
    palette = data.get("palette", [])
    out: BlockMap = {}
    for entry in data.get("blocks", []):
        x, y, z, idx = entry
        state = palette[int(idx)] if isinstance(idx, int) or str(idx).isdigit() else str(idx)
        out[(int(x), int(y), int(z))] = state
    return out


def encode_scan(block_map: BlockMap) -> Dict[str, Any]:
    """Inverse of decode_scan (used by the mock HTTP server)."""
    palette: List[str] = ["minecraft:air"]
    index: Dict[str, int] = {"minecraft:air": 0}
    blocks = []
    for (x, y, z), state in block_map.items():
        if state not in index:
            index[state] = len(palette)
            palette.append(state)
        blocks.append([x, y, z, index[state]])
    return {"palette": palette, "blocks": blocks, "count": len(blocks)}


def get_bridge(prefer: Optional[str] = None, url: Optional[str] = None) -> Bridge:
    """Pick a bridge: env COPILOT_BRIDGE=mock|http|auto (default auto), COPILOT_MOD_URL for http.

    auto: HttpBridge if the mod answers /health, else MockBridge.
    """
    prefer = (prefer or os.environ.get("COPILOT_BRIDGE") or "auto").lower()
    url = url or os.environ.get("COPILOT_MOD_URL") or DEFAULT_MOD_URL
    if prefer == "mock":
        from mock_mod.bridge import MockBridge

        return MockBridge()
    http = HttpBridge(url)
    if prefer == "http":
        return http
    try:
        http.health()
        return http
    except BridgeError:
        from mock_mod.bridge import MockBridge

        return MockBridge()
