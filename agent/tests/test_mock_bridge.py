from mock_mod import MockBridge


def test_health_and_player_shape():
    b = MockBridge()
    h = b.health()
    assert h["ok"] and h["world_loaded"] and h["client_jar"] is None and "mc_version" in h
    p = b.player()
    assert p["name"] == "Steve" and p["facing"] == "north" and p["pos"] == [0.5, 64.0, 0.5]
    assert p["dimension"] == "minecraft:overworld" and p["looking_at"] is None


def test_flat_world_and_scan_includes_air():
    b = MockBridge(ground_y=63)
    m = b.scan((0, 62, 0), (1, 64, 1))
    assert len(m) == 2 * 3 * 2
    assert m[(0, 63, 0)].startswith("minecraft:grass_block")
    assert m[(0, 62, 0)] == "minecraft:dirt"
    assert m[(0, 64, 0)] == "minecraft:air"


def test_setblocks_applies_immediately_and_records_calls():
    b = MockBridge()
    n = b.setblocks([([(1, 64, 1, "minecraft:stone"), (1, 65, 1, "minecraft:oak_planks")], 60), ([(2, 64, 1, "minecraft:glass")], 60)], flags=3)
    assert n == 3
    assert b.get_block(1, 65, 1) == "minecraft:oak_planks"
    assert b.calls[-1]["count"] == 3 and len(b.calls[-1]["chunks"]) == 2 and b.calls[-1]["flags"] == 3
    assert b.count_non_air() == 3
    b.setblocks([([(1, 64, 1, "minecraft:air")], 0)])
    assert b.count_non_air() == 2


def test_say_blocks_camera():
    b = MockBridge()
    b.say("hello")
    assert b.chat == ["hello"]
    blocks = b.blocks()
    assert len(blocks) >= 20 and all("id" in x and "properties" in x for x in blocks)
    assert any(x["id"] == "minecraft:oak_stairs" for x in blocks)
    r = b.camera(mode="orbit", center=[0, 64, 0], radius=20, seconds=5)
    assert r["ok"] and b.camera_calls[-1]["radius"] == 20


def test_looking_at_hits_placed_block():
    b = MockBridge()
    b.set_block(0, 65, -3, "minecraft:stone")  # north of the player at eye height
    p = b.player()
    assert p["looking_at"] is not None and p["looking_at"]["block"] == "minecraft:stone"
    assert p["looking_at"]["pos"] == [0, 65, -3]


def test_facing_from_yaw():
    from mock_mod.bridge import facing_from_yaw

    assert facing_from_yaw(0) == "south" and facing_from_yaw(90) == "west"
    assert facing_from_yaw(180) == "north" and facing_from_yaw(-90) == "east" and facing_from_yaw(270) == "east"
