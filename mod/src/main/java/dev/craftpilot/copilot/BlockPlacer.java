package dev.craftpilot.copilot;

import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.Level;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

/**
 * Tick-driven placement queue. Each {@link Chunk} is placed in one server tick; consecutive chunks
 * are spaced by {@code delay_ms} (rounded up to ticks, 50 ms each), which is what animates a build
 * bottom-up. Everything here runs on the server thread via {@code END_SERVER_TICK}.
 *
 * <p>After a chunk is set, a <b>connectivity post-process pass</b> re-evaluates every placed block
 * with {@link Block#updateFromNeighbourShapes}: {@code setBlock(..., UPDATE_ALL)} updates the
 * neighbours' shapes but never recomputes the placed block's own connectivity (we skip
 * {@code getStateForPlacement}), so without this pass the last stair in a row stays
 * {@code shape=straight} and fences/walls/panes placed after their neighbours stay unconnected.
 */
public final class BlockPlacer {
    public record Placement(BlockPos pos, BlockState state) {
    }

    public record Chunk(ResourceKey<Level> dimension, List<Placement> blocks, int delayTicks, int flags) {
    }

    private static final Deque<Chunk> QUEUE = new ArrayDeque<>();
    private static int waitTicks = 0;
    private static long placedTotal = 0;
    private static long postprocessedTotal = 0;

    private BlockPlacer() {
    }

    public static void register() {
        ServerTickEvents.END_SERVER_TICK.register(BlockPlacer::tick);
    }

    public static synchronized void enqueue(List<Chunk> chunks) {
        QUEUE.addAll(chunks);
    }

    public static synchronized int pendingChunks() {
        return QUEUE.size();
    }

    public static synchronized int pendingBlocks() {
        int n = 0;
        for (Chunk c : QUEUE) {
            n += c.blocks().size();
        }
        return n;
    }

    public static synchronized long placedTotal() {
        return placedTotal;
    }

    /** Blocks whose own connectivity (stairs shape, fence/wall/pane sides, ...) was corrected. */
    public static synchronized long postprocessedTotal() {
        return postprocessedTotal;
    }

    public static synchronized void clear() {
        QUEUE.clear();
        waitTicks = 0;
    }

    private static void tick(MinecraftServer server) {
        Chunk chunk;
        synchronized (BlockPlacer.class) {
            if (waitTicks > 0) {
                waitTicks--;
                return;
            }
            chunk = QUEUE.pollFirst();
            if (chunk == null) {
                return;
            }
            waitTicks = Math.max(0, chunk.delayTicks());
        }
        ServerLevel world = server.getLevel(chunk.dimension());
        if (world == null) {
            world = server.overworld();
        }
        int placed = 0;
        for (Placement p : chunk.blocks()) {
            try {
                if (world.setBlock(p.pos(), p.state(), chunk.flags())) {
                    placed++;
                }
            } catch (Exception e) {
                CopilotClientMod.LOGGER.warn("[copilot] setBlock failed at {}: {}", p.pos(), e.toString());
            }
        }
        int fixedCount = postProcess(world, chunk);
        synchronized (BlockPlacer.class) {
            placedTotal += placed;
            postprocessedTotal += fixedCount;
        }
    }

    /**
     * Recompute each placed block's own connectivity from its (now complete) neighbourhood. One
     * extra {@code getBlockState} per block; only changed states are written, with
     * {@code UPDATE_CLIENTS | UPDATE_KNOWN_SHAPE} so there is no further neighbour cascade.
     * Air (removed) positions are skipped, and a result of air is ignored so props the agent
     * placed deliberately on unsupported spots are never deleted here.
     */
    private static int postProcess(ServerLevel world, Chunk chunk) {
        int fixedCount = 0;
        int flags = Block.UPDATE_CLIENTS | Block.UPDATE_KNOWN_SHAPE;
        for (Placement p : chunk.blocks()) {
            try {
                BlockState cur = world.getBlockState(p.pos());
                if (cur.isAir()) {
                    continue;
                }
                BlockState fixed = Block.updateFromNeighbourShapes(cur, world, p.pos());
                if (fixed != cur && !fixed.isAir()) {
                    if (world.setBlock(p.pos(), fixed, flags)) {
                        fixedCount++;
                    }
                }
            } catch (Exception e) {
                CopilotClientMod.LOGGER.warn("[copilot] updateFromNeighbourShapes failed at {}: {}", p.pos(), e.toString());
            }
        }
        return fixedCount;
    }
}
