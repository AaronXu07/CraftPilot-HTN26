package dev.craftpilot.copilot;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.render.RenderLayer;
import net.minecraft.client.render.VertexConsumer;
import net.minecraft.client.render.VertexConsumerProvider;
import net.minecraft.client.render.WorldRenderer;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.util.math.Vec3d;

import java.util.List;

/**
 * Wireframe of the box a build will fill, drawn while the build is in flight so the player sees
 * where it lands and that something is happening. A scan line sweeps the box while we wait for the
 * service and then tracks the highest placed row once blocks stream in; the box flashes green and
 * disappears when the placement queue drains.
 *
 * <p>Phases: {@code AIMING} follows the player until they lock the location (see
 * {@link PendingBuild}); {@code PLANNING} is that locked, client-side guess; {@code REVIEW} holds the
 * generated hologram ({@link GhostRender}) still where it was locked until the player builds or
 * moves it; the service replaces the box through {@code POST /outline} with {@code GENERATING}
 * (exact bounds, right after the LLM composes) and {@code PLACING} (the trimmed box it actually
 * fills). State is static and synchronized like {@link CameraOrbit}: HTTP threads set it, the render
 * thread reads it, the server thread reports placed blocks.
 */
public final class BuildOutline {
    public enum Phase { AIMING, REVIEW, PLANNING, GENERATING, PLACING, DONE }

    /** The pose a box was anchored from, so the request can carry exactly what the player saw. */
    public record Pose(double x, double y, double z, float yaw) {
    }

    /** Default bounds the service uses when the program does not say (CRAFTPILOT_DEFAULT_BOUNDS). */
    private static final int GUESS_W = 25;
    private static final int GUESS_H = 30;
    private static final int GUESS_D = 25;
    private static final int GUESS_GAP = 2;
    private static final int DONE_TICKS = 15;
    private static final int SWEEP_TICKS = 60;
    private static final double INFLATE = 0.02;

    private static boolean active = false;
    private static Phase phase = Phase.PLANNING;
    private static double minX, minY, minZ, maxX, maxY, maxZ;
    private static int placedTopY = Integer.MIN_VALUE;
    private static long ticks = 0;
    private static long doneAt = -1;
    private static int aimW = GUESS_W, aimH = GUESS_H, aimD = GUESS_D;
    private static Pose aimPose;
    private static int aimLook = 0;              // 0 south, 1 west, 2 north, 3 east: where the player looked
    // Where the hologram's rotated min corner sits. Pinned when the box is anchored from a pose and
    // deliberately not moved by /outline updates: the service's "placing" box is trimmed to the
    // blocks, while the hologram (and the real build's origin) use the full bounds box.
    private static double ghostX, ghostY, ghostZ;

    private BuildOutline() {
    }

    public static void register() {
        WorldRenderEvents.BEFORE_DEBUG_RENDER.register(BuildOutline::render);
        ClientTickEvents.END_CLIENT_TICK.register(client -> onTick());
    }

    // ---- state changes (any thread) -----------------------------------

    /** Guess where a default-sized building would land from where the player stands right now. */
    public static void showPlaceholder(ClientPlayerEntity player) {
        anchor(poseOf(player), GUESS_W, GUESS_H, GUESS_D, Phase.PLANNING);
    }

    /** Hold a {@code w x h x d} box (with its hologram) still, anchored from {@code pose}. */
    public static void showAt(Pose pose, int w, int h, int d) {
        synchronized (BuildOutline.class) {
            aimW = w > 0 ? w : GUESS_W;
            aimH = h > 0 ? h : GUESS_H;
            aimD = d > 0 ? d : GUESS_D;
            doneAt = -1;
            placedTopY = Integer.MIN_VALUE;
        }
        anchor(pose, aimW, aimH, aimD, Phase.REVIEW);
    }

    /** From REVIEW back to following the player, keeping the box size. */
    public static synchronized void resumeAiming() {
        if (active && phase == Phase.REVIEW) {
            phase = Phase.AIMING;
        }
    }

    public static synchronized boolean isReviewing() {
        return active && phase == Phase.REVIEW;
    }

    public static Pose poseOf(ClientPlayerEntity player) {
        return new Pose(player.getX(), player.getY(), player.getZ(), player.getYaw());
    }

    /**
     * Start following the player with a {@code w x h x d} box (0 for any dimension means the default)
     * until {@link #lock()} freezes it. {@code w} and {@code d} are the building's own footprint; the
     * box is rotated to face the player like the building will be.
     */
    public static void startAiming(int w, int h, int d) {
        synchronized (BuildOutline.class) {
            aimW = w > 0 ? w : GUESS_W;
            aimH = h > 0 ? h : GUESS_H;
            aimD = d > 0 ? d : GUESS_D;
            phase = Phase.AIMING;
            active = true;
            doneAt = -1;
            placedTopY = Integer.MIN_VALUE;
        }
        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            anchor(poseOf(player), aimW, aimH, aimD, Phase.AIMING);
        }
    }

    /** Freeze the box where it is. Returns the pose it was anchored from, or null if not aiming or reviewing. */
    public static synchronized Pose lock() {
        if (!active || (phase != Phase.AIMING && phase != Phase.REVIEW)) {
            return null;
        }
        phase = Phase.PLANNING;
        return aimPose;
    }

    public static synchronized boolean isAiming() {
        return active && phase == Phase.AIMING;
    }

    /**
     * Anchor a {@code w x h x d} box in front of the player. Mirrors {@code plan_origin} in
     * {@code src/craftpilot/place/anchor.py} (look direction from yaw, laterally centred, {@code gap}
     * blocks past the player's block, floor at the feet) and must stay in step with it; the service
     * replaces this box with the real one as soon as it knows it. The building faces the player, so
     * looking east or west swaps its width and depth, exactly as the service rotates the grid.
     */
    private static void anchor(Pose pose, int w, int h, int d, Phase p) {
        double px = pose.x();
        double py = pose.y();
        double pz = pose.z();
        int bx = (int) Math.floor(px);
        int bz = (int) Math.floor(pz);
        float yaw = pose.yaw();
        int look = (int) Math.floor((((yaw % 360) + 360) % 360 + 45) / 90) % 4; // 0 south, 1 west, 2 north, 3 east
        if (look == 1 || look == 3) {
            int t = w;
            w = d;
            d = t;
        }
        int cx = (int) Math.round(px - w / 2.0);
        int cz = (int) Math.round(pz - d / 2.0);
        int ox;
        int oz;
        switch (look) {
            case 0 -> { ox = cx; oz = bz + 1 + GUESS_GAP; }
            case 2 -> { ox = cx; oz = bz - 1 - GUESS_GAP - (d - 1); }
            case 3 -> { ox = bx + 1 + GUESS_GAP; oz = cz; }
            default -> { ox = bx - 1 - GUESS_GAP - (w - 1); oz = cz; }
        }
        int oy = (int) Math.floor(py);
        set(ox, oy, oz, ox + w - 1, oy + h - 1, oz + d - 1, p);
        synchronized (BuildOutline.class) {
            aimPose = pose;
            aimLook = look;
            ghostX = ox;
            ghostY = oy;
            ghostZ = oz;
        }
    }

    /** Inclusive block coordinates of the box. */
    public static synchronized void set(int x0, int y0, int z0, int x1, int y1, int z1, Phase p) {
        minX = Math.min(x0, x1);
        minY = Math.min(y0, y1);
        minZ = Math.min(z0, z1);
        maxX = Math.max(x0, x1) + 1;
        maxY = Math.max(y0, y1) + 1;
        maxZ = Math.max(z0, z1) + 1;
        phase = p;
        active = true;
        doneAt = -1;
        if (p != Phase.PLACING) {
            placedTopY = Integer.MIN_VALUE;
        }
    }

    public static Phase phaseOf(String name) {
        return switch (name == null ? "" : name.toLowerCase()) {
            case "planning" -> Phase.PLANNING;
            case "placing" -> Phase.PLACING;
            case "done" -> Phase.DONE;
            default -> Phase.GENERATING;
        };
    }

    /** Called by {@link BlockPlacer} on the server thread after each chunk lands. */
    public static synchronized void onPlaced(List<BlockPlacer.Placement> blocks) {
        if (!active || phase == Phase.AIMING || phase == Phase.REVIEW) {
            return; // a previous build still landing must not disturb a box the player is aiming
        }
        if (phase != Phase.PLACING) {
            phase = Phase.PLACING; // blocks are landing: whatever the service said, we are placing now
        }
        for (BlockPlacer.Placement p : blocks) {
            int y = p.pos().getY();
            if (y > placedTopY) {
                placedTopY = y;
            }
        }
    }

    /** The placement queue drained: drop the hologram, flash green briefly, then clear. */
    public static synchronized void finish() {
        if (!active || phase == Phase.AIMING || phase == Phase.REVIEW) {
            return;
        }
        phase = Phase.DONE;
        doneAt = ticks;
        GhostRender.clear();
    }

    public static synchronized void clear() {
        active = false;
        doneAt = -1;
        placedTopY = Integer.MIN_VALUE;
        GhostRender.clear();
    }

    public static synchronized boolean isActive() {
        return active;
    }

    // ---- client thread --------------------------------------------------

    private static void onTick() {
        boolean aiming;
        int w, h, d;
        synchronized (BuildOutline.class) {
            ticks++;
            if (active && phase == Phase.DONE && doneAt >= 0 && ticks - doneAt > DONE_TICKS) {
                active = false;
                GhostRender.clear();
            }
            aiming = active && phase == Phase.AIMING;
            w = aimW;
            h = aimH;
            d = aimD;
        }
        if (aiming) {
            ClientPlayerEntity player = MinecraftClient.getInstance().player;
            if (player != null) {
                anchor(poseOf(player), w, h, d, Phase.AIMING);
            }
        }
    }

    private static void render(WorldRenderContext context) {
        double x0, y0, z0, x1, y1, z1;
        Phase p;
        long t;
        int top;
        int look;
        double gx, gy, gz;
        synchronized (BuildOutline.class) {
            if (!active) {
                return;
            }
            x0 = minX; y0 = minY; z0 = minZ; x1 = maxX; y1 = maxY; z1 = maxZ;
            p = phase;
            t = ticks;
            top = placedTopY;
            look = aimLook;
            gx = ghostX; gy = ghostY; gz = ghostZ;
        }
        VertexConsumerProvider consumers = context.consumers();
        if (consumers == null) {
            return;
        }
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.world == null) {
            return;
        }
        float partial = context.tickCounter().getTickDelta(false);
        double time = t + partial;
        float r, g, b, base;
        switch (p) {
            case AIMING -> { r = 0.4f; g = 0.9f; b = 1.0f; base = 0.6f; }
            case REVIEW -> { r = 0.4f; g = 0.9f; b = 1.0f; base = 0.8f; }
            case PLANNING -> { r = 0.85f; g = 0.85f; b = 0.9f; base = 0.35f; }
            case DONE -> { r = 0.45f; g = 0.95f; b = 0.45f; base = 0.9f; }
            default -> { r = 1.0f; g = 0.78f; b = 0.25f; base = 0.65f; }
        }
        float pulse = (float) (base + (p == Phase.REVIEW ? 0.0 : 0.25) * Math.sin(time / (p == Phase.AIMING ? 14.0 : 8.0)));
        float alpha = Math.max(0.15f, Math.min(1.0f, pulse));

        Vec3d cam = context.camera().getPos();
        MatrixStack matrices = context.matrixStack();
        matrices.push();
        matrices.translate(-cam.x, -cam.y, -cam.z);

        // The hologram first: the immediate consumer keeps one live buffer for ad-hoc layers, so asking
        // for the quads layer would finish a lines buffer we were still writing ("Not building!").
        // The building faces the player, so it is turned by the look direction: looking south means
        // the building faces north (2 turns from the engine's south), and so on.
        if (GhostRender.isPresent() && p != Phase.DONE) {
            int turns = switch (look) { case 0 -> 2; case 1 -> 3; case 2 -> 0; default -> 1; };
            float hideBelow = top == Integer.MIN_VALUE ? Float.NEGATIVE_INFINITY : (float) (top - gy + 1);
            GhostRender.render(matrices, consumers, gx, gy, gz, turns, hideBelow, p == Phase.AIMING || p == Phase.REVIEW);
        }

        VertexConsumer lines = consumers.getBuffer(RenderLayer.getLines());
        WorldRenderer.drawBox(matrices, lines, x0 - INFLATE, y0 - INFLATE, z0 - INFLATE,
                x1 + INFLATE, y1 + INFLATE, z1 + INFLATE, r, g, b, alpha);

        // Scan line: sweeps while we wait, sits on the highest placed row while blocks land.
        if (p != Phase.DONE && p != Phase.AIMING && p != Phase.REVIEW) {
            double scanY;
            if (p == Phase.PLACING && top != Integer.MIN_VALUE) {
                scanY = Math.min(y1, top + 1 + 0.05 * Math.sin(time / 3.0));
            } else {
                double frac = (time % SWEEP_TICKS) / SWEEP_TICKS;
                scanY = y0 + frac * (y1 - y0);
            }
            float scanAlpha = Math.min(1.0f, alpha + 0.3f);
            WorldRenderer.drawBox(matrices, lines, x0 - INFLATE, scanY, z0 - INFLATE,
                    x1 + INFLATE, scanY, z1 + INFLATE, r, g, b, scanAlpha);
        }
        matrices.pop();
    }
}
