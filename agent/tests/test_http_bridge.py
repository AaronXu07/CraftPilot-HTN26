"""HttpBridge against the mock_mod FastAPI app served in a background thread."""
import socket
import threading
import time

import pytest
import uvicorn

from copilot.bridge import BridgeError, HttpBridge, decode_scan, encode_scan, get_bridge, split_boxes
from mock_mod import MockBridge
from mock_mod.server import create_app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    mock = MockBridge()
    port = _free_port()
    config = uvicorn.Config(create_app(mock), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield mock, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=3)


def test_all_endpoints_roundtrip(server):
    mock, url = server
    b = HttpBridge(url)
    h = b.health()
    assert h["ok"] is True and h["mc_version"] == "1.21.1"
    p = b.player()
    assert p["facing"] == "north" and len(p["pos"]) == 3
    n = b.setblocks([([(5, 64, 5, "minecraft:stone"), (5, 65, 5, "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]")], 60)])
    assert n == 2 and mock.get_block(5, 64, 5) == "minecraft:stone"
    m = b.scan((4, 63, 4), (6, 66, 6))
    assert len(m) == 3 * 4 * 3
    assert m[(5, 65, 5)].startswith("minecraft:oak_stairs[facing=east")
    assert m[(4, 64, 4)] == "minecraft:air" and m[(4, 63, 4)].startswith("minecraft:grass_block")
    b.say("hi from http")
    assert mock.chat[-1] == "hi from http"
    blocks = b.blocks()
    assert len(blocks) >= 20 and blocks[0]["id"].startswith("minecraft:")
    r = b.camera(mode="orbit", center=[0, 64, 0], radius=10, seconds=2)
    assert r["ok"] and mock.camera_calls[-1]["seconds"] == 2
    assert b.setblocks([]) == 0
    b.close()


def test_scan_splits_large_boxes(server):
    mock, url = server
    b = HttpBridge(url)
    m = b.scan((0, 64, 0), (70, 64, 3))  # 71 wide -> two boxes along x
    assert len(m) == 71 * 1 * 4
    assert split_boxes((0, 0, 0), (70, 0, 3), 64) == [((0, 0, 0), (63, 0, 3)), ((64, 0, 0), (70, 0, 3))]


def test_scan_codec_roundtrip():
    m = {(0, 0, 0): "minecraft:air", (1, 0, 0): "minecraft:stone", (2, 0, 0): "minecraft:stone"}
    enc = encode_scan(m)
    assert enc["palette"][0] == "minecraft:air" and enc["count"] == 3 and len(enc["palette"]) == 2
    assert decode_scan(enc) == m


def test_unreachable_raises_and_get_bridge_falls_back(monkeypatch):
    b = HttpBridge("http://127.0.0.1:1")  # nothing listens here
    with pytest.raises(BridgeError):
        b.health()
    monkeypatch.setenv("COPILOT_BRIDGE", "auto")
    monkeypatch.setenv("COPILOT_MOD_URL", "http://127.0.0.1:1")
    assert isinstance(get_bridge(), MockBridge)
    monkeypatch.setenv("COPILOT_BRIDGE", "mock")
    assert isinstance(get_bridge(), MockBridge)
    assert isinstance(get_bridge("http", "http://127.0.0.1:1"), HttpBridge)
