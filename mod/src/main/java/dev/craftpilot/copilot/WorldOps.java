package dev.craftpilot.copilot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.command.argument.BlockArgumentParser;
import net.minecraft.registry.Registries;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.state.property.Property;
import net.minecraft.util.Identifier;
import net.minecraft.util.math.BlockPos;

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
        return BlockArgumentParser.stringifyBlockState(state);
    }

    /** Parse a state string; unknown properties/values raise {@link CommandSyntaxException}. */
    public static BlockState parseState(String text) throws CommandSyntaxException {
        String s = text.trim();
        if (!s.contains(":")) {
            s = "minecraft:" + s;
        }
        return BlockArgumentParser.block(Registries.BLOCK.getReadOnlyWrapper(), s, false).blockState();
    }

    /**
     * Scan an inclusive region into {@code {palette:[...], blocks:[[x,y,z,idx],...], count}}.
     * Palette index 0 is always {@code minecraft:air}. Must run on the server thread.
     */
    public static JsonObject scan(ServerWorld world, int x0, int y0, int z0, int x1, int y1, int z1) {
        int minX = Math.min(x0, x1), maxX = Math.max(x0, x1);
        int minY = Math.min(y0, y1), maxY = Math.max(y0, y1);
        int minZ = Math.min(z0, z1), maxZ = Math.max(z0, z1);
        List<String> palette = new ArrayList<>();
        Map<BlockState, Integer> index = new HashMap<>();
        palette.add("minecraft:air");
        index.put(net.minecraft.block.Blocks.AIR.getDefaultState(), 0);
        JsonArray blocks = new JsonArray();
        BlockPos.Mutable pos = new BlockPos.Mutable();
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
        for (Block block : Registries.BLOCK) {
            Identifier id = Registries.BLOCK.getId(block);
            JsonObject entry = new JsonObject();
            entry.addProperty("id", id.toString());
            JsonObject props = new JsonObject();
            for (Property<?> property : block.getStateManager().getProperties()) {
                JsonArray values = new JsonArray();
                for (String v : valueNames(property)) {
                    values.add(v);
                }
                props.add(property.getName(), values);
            }
            entry.add("properties", props);
            entry.addProperty("default", stringify(block.getDefaultState()));
            arr.add(entry);
        }
        JsonObject out = new JsonObject();
        out.add("blocks", arr);
        out.addProperty("count", arr.size());
        return out;
    }

    private static <T extends Comparable<T>> List<String> valueNames(Property<T> property) {
        Collection<T> values = property.getValues();
        List<String> names = new ArrayList<>(values.size());
        for (T v : values) {
            names.add(property.name(v));
        }
        return names;
    }
}
