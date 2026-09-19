package dev.craftpilot.copilot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.state.properties.Property;
import net.minecraft.resources.Identifier;
import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure helpers over the game's own registries and world: state string round-trips, region scans
 * and the block registry dump. Methods that touch a world must run on the server thread.
 */
public final class WorldOps {
    private WorldOps() {
    }

    /** {@code minecraft:oak_stairs[facing=north,half=bottom,...]} — the game's canonical form. */
    public static String stringify(BlockState state) {
        return BlockStateParser.serialize(state);
    }

    /** Parse a state string; unknown properties/values raise {@link CommandSyntaxException}. */
    public static BlockState parseState(String text) throws CommandSyntaxException {
        String s = text.trim();
        if (!s.contains(":")) {
            s = "minecraft:" + s;
        }
        return BlockStateParser.parseForBlock(BuiltInRegistries.BLOCK, s, false).blockState();
    }

    /**
     * Scan an inclusive region into {@code {palette:[...], blocks:[[x,y,z,idx],...], count}}.
     * Palette index 0 is always {@code minecraft:air}. Must run on the server thread.
     */
    public static JsonObject scan(ServerLevel world, int x0, int y0, int z0, int x1, int y1, int z1) {
        int minX = Math.min(x0, x1), maxX = Math.max(x0, x1);
        int minY = Math.min(y0, y1), maxY = Math.max(y0, y1);
        int minZ = Math.min(z0, z1), maxZ = Math.max(z0, z1);
        List<String> palette = new ArrayList<>();
        Map<BlockState, Integer> index = new HashMap<>();
        palette.add("minecraft:air");
        index.put(net.minecraft.world.level.block.Blocks.AIR.defaultBlockState(), 0);
        JsonArray blocks = new JsonArray();
        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        int count = 0;
        for (int y = minY; y <= maxY; y++) {
            for (int z = minZ; z <= maxZ; z++) {
                for (int x = minX; x <= maxX; x++) {
                    pos.set(x, y, z);
                    BlockState state = world.getBlockState(pos);
                    Integer idx = index.get(state);
                    if (idx == null) {
                        idx = palette.size();
                        palette.add(stringify(state));
                        index.put(state, idx);
                    }
                    JsonArray entry = new JsonArray();
                    entry.add(x);
                    entry.add(y);
                    entry.add(z);
                    entry.add(idx);
                    blocks.add(entry);
                    count++;
                }
            }
        }
        JsonArray pal = new JsonArray();
        for (String p : palette) {
            pal.add(p);
        }
        JsonObject out = new JsonObject();
        out.add("palette", pal);
        out.add("blocks", blocks);
        out.addProperty("count", count);
        return out;
    }

    /** Every registered block with its properties, allowed values and default state. */
    public static JsonObject blockDump() {
        JsonArray arr = new JsonArray();
        for (Block block : BuiltInRegistries.BLOCK) {
            Identifier id = BuiltInRegistries.BLOCK.getKey(block);
            JsonObject entry = new JsonObject();
            entry.addProperty("id", id.toString());
            JsonObject props = new JsonObject();
            for (Property<?> property : block.getStateDefinition().getProperties()) {
                JsonArray values = new JsonArray();
                for (String v : valueNames(property)) {
                    values.add(v);
                }
                props.add(property.getName(), values);
            }
            entry.add("properties", props);
            entry.addProperty("default", stringify(block.defaultBlockState()));
            arr.add(entry);
        }
        JsonObject out = new JsonObject();
        out.add("blocks", arr);
        out.addProperty("count", arr.size());
        return out;
    }

    private static <T extends Comparable<T>> List<String> valueNames(Property<T> property) {
        Collection<T> values = property.getPossibleValues();
        List<String> names = new ArrayList<>(values.size());
        for (T v : values) {
            names.add(property.getName(v));
        }
        return names;
    }
}
