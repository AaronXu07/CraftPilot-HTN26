package dev.craftpilot.copilot;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.GameMode;

/**
 * Best-effort demo camera: switches the player to spectator, orbits the given centre for N
 * seconds while looking at it, then returns the player and restores the previous game mode.
 * Requires cheats (the game-mode switch goes through {@code /gamemode}).
 */
public final class CameraOrbit {
    private static boolean active = false;
    private static Vec3d center;
    private static double radius;
    private static int totalTicks;
    private static int tick;
    private static Vec3d returnPos;
    private static float returnYaw;
    private static float returnPitch;
    private static GameMode returnMode;

    private CameraOrbit() {
    }

    public static void register() {
        ClientTickEvents.END_CLIENT_TICK.register(CameraOrbit::onTick);
    }

    public static synchronized boolean isActive() {
        return active;
    }

    /** Called from an HTTP thread; the actual work happens on the client thread. */
    public static void start(double cx, double cy, double cz, double r, double seconds) {
        MinecraftClient client = MinecraftClient.getInstance();
        client.execute(() -> {
            ClientPlayerEntity player = client.player;
            if (player == null) {
                return;
            }
            synchronized (CameraOrbit.class) {
                if (!active) {
                    returnPos = player.getPos();
                    returnYaw = player.getYaw();
                    returnPitch = player.getPitch();
                    returnMode = client.interactionManager != null ? client.interactionManager.getCurrentGameMode() : GameMode.CREATIVE;
                }
                center = new Vec3d(cx, cy, cz);
                radius = Math.max(4.0, r);
                totalTicks = Math.max(20, (int) Math.round(seconds * 20.0));
                tick = 0;
                active = true;
            }
            player.networkHandler.sendCommand("gamemode spectator");
        });
    }

    public static void stop() {
        MinecraftClient client = MinecraftClient.getInstance();
        client.execute(() -> finish(client));
    }

    private static void onTick(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) {
            synchronized (CameraOrbit.class) {
                active = false;
            }
            return;
        }
        Vec3d c;
        double r;
        int t;
        int total;
        synchronized (CameraOrbit.class) {
            if (!active) {
                return;
            }
            c = center;
            r = radius;
            t = tick++;
            total = totalTicks;
        }
        if (t >= total) {
            finish(client);
            return;
        }
        double angle = (2.0 * Math.PI) * t / total;
        double height = r * 0.55;
        double x = c.x + r * Math.cos(angle);
        double z = c.z + r * Math.sin(angle);
        double y = c.y + height;
        double dx = c.x - x;
        double dy = c.y - y;
        double dz = c.z - z;
        float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float pitch = (float) Math.toDegrees(Math.atan2(-dy, Math.sqrt(dx * dx + dz * dz)));
        player.setVelocity(Vec3d.ZERO);
        player.refreshPositionAndAngles(x, y, z, yaw, pitch);
    }

    private static void finish(MinecraftClient client) {
        Vec3d pos;
        float yaw;
        float pitch;
        GameMode mode;
        synchronized (CameraOrbit.class) {
            if (!active) {
                return;
            }
            active = false;
            pos = returnPos;
            yaw = returnYaw;
            pitch = returnPitch;
            mode = returnMode;
        }
        ClientPlayerEntity player = client.player;
        if (player == null || pos == null) {
            return;
        }
        player.setVelocity(Vec3d.ZERO);
        player.refreshPositionAndAngles(pos.x, pos.y, pos.z, yaw, pitch);
        String modeName = mode != null ? mode.getName() : "creative";
        player.networkHandler.sendCommand("gamemode " + modeName);
    }
}
