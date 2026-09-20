package dev.craftpilot.copilot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.KeyMapping;
import net.minecraft.resources.Identifier;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.network.chat.Component;
import org.lwjgl.glfw.GLFW;

/**
 * A service request that is waiting for the player to pick where the building goes. While one is
 * armed the {@link BuildOutline} follows the player (or, once the service has sent the hologram,
 * holds it where the player first locked it); the lock key stamps that pose into the request body and
 * posts it, the move key lets the hologram follow the player again. Also remembers the pending plan
 * (from {@code /build plan}) so {@code /build go} knows its size.
 */
public final class PendingBuild {
    private static final int HINT_EVERY = 20;

    private static KeyMapping lockKey;
    private static KeyMapping moveKey;
    private static String path;
    private static JsonObject body;
    private static String label;
    private static BuildOutline.Pose lockedPose;   // where the wireframe was locked; the ghost lands there
    private static boolean planPending = false;
    private static int planW = 0, planH = 0, planD = 0;
    private static long ticks = 0;

    private PendingBuild() {
    }

    public static void register() {
        KeyMapping.Category category = KeyMapping.Category.register(Identifier.fromNamespaceAndPath("copilot", "copilot"));
        lockKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.copilot.lock", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_G, category));
        moveKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.copilot.move", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_H, category));
        ClientTickEvents.END_CLIENT_TICK.register(PendingBuild::onTick);
    }

    /** Arm {@code body} for {@code path}; the outline starts following the player at {@code w x h x d}. */
    public static void arm(String requestPath, JsonObject requestBody, String what, int w, int h, int d) {
        synchronized (PendingBuild.class) {
            path = requestPath;
            body = requestBody;
            label = what;
            lockedPose = null;
        }
        GhostRender.clear();
        BuildOutline.startAiming(w, h, d);
        CopilotClientMod.chat("§6[craftpilot]§7 aim the box and press §f[" + keyName(lockKey) + "]§7 to build " + what
                + " there, or /build cancel");
    }

    /**
     * The service sent the hologram: show it where the wireframe was locked (or at the player if there
     * was no wireframe step) and wait for the lock key to build it there or the move key to re-aim.
     */
    public static void armGhost(String requestPath, JsonObject requestBody, int w, int h, int d, int[] voxels) {
        BuildOutline.Pose at;
        synchronized (PendingBuild.class) {
            path = requestPath;
            body = requestBody;
            label = "it";
            at = lockedPose;
        }
        Minecraft client = Minecraft.getInstance();
        if (at == null) {
            LocalPlayer player = client.player;
            if (player == null) {
                return;
            }
            at = BuildOutline.poseOf(player);
        }
        GhostRender.set(w, h, d, voxels);
        BuildOutline.showAt(at, w, h, d);
        CopilotClientMod.chat("§6[craftpilot]§7 preview ready: §f[" + keyName(lockKey) + "]§7 build it here, §f["
                + keyName(moveKey) + "]§7 move it, or /build cancel");
    }

    /** Freeze the box, add its pose to the body, and send it. No-op unless something is armed. */
    public static void lock() {
        String p;
        JsonObject b;
        String what;
        synchronized (PendingBuild.class) {
            p = path;
            b = body;
            what = label;
            path = null;
            body = null;
            label = null;
        }
        if (p == null || b == null) {
            return;
        }
        BuildOutline.Pose pose = BuildOutline.lock();
        if (pose != null) {
            JsonArray pos = new JsonArray();
            pos.add(pose.x());
            pos.add(pose.y());
            pos.add(pose.z());
            b.add("pos", pos);
            b.addProperty("yaw", pose.yaw());
            synchronized (PendingBuild.class) {
                lockedPose = pose;
            }
        }
        boolean ghostRequest = b.has("ghost") && b.get("ghost").getAsBoolean();
        CopilotClientMod.chat("§6[craftpilot]§7 " + (ghostRequest ? "composing " + what + "..." : "building " + what + "..."));
        ServiceClient.postAsync(p, b);
    }

    /** Let the hologram follow the player again. */
    public static void move() {
        if (BuildOutline.isReviewing()) {
            BuildOutline.resumeAiming();
        }
    }

    /** Drop the armed request, the hologram and the pending plan. */
    public static void cancel() {
        synchronized (PendingBuild.class) {
            path = null;
            body = null;
            label = null;
            lockedPose = null;
            planPending = false;
        }
        BuildOutline.clear();
    }

    public static synchronized boolean isArmed() {
        return path != null;
    }

    public static synchronized void setPlan(int w, int h, int d) {
        planPending = true;
        planW = w;
        planH = h;
        planD = d;
    }

    public static synchronized void clearPlan() {
        planPending = false;
    }

    public static synchronized boolean hasPlan() {
        return planPending;
    }

    public static synchronized int[] planBounds() {
        return new int[] {planW, planH, planD};
    }

    public static String keyName() {
        return keyName(lockKey);
    }

    private static String keyName(KeyMapping key) {
        return key != null ? key.getTranslatedKeyMessage().getString() : "?";
    }

    private static void onTick(Minecraft client) {
        ticks++;
        if (lockKey != null) {
            while (lockKey.consumeClick()) {
                lock();
            }
        }
        if (moveKey != null) {
            while (moveKey.consumeClick()) {
                move();
            }
        }
        if (!isArmed()) {
            return;
        }
        LocalPlayer player = client.player;
        if (player == null || ticks % HINT_EVERY != 0) {
            return;
        }
        if (BuildOutline.isReviewing()) {
            player.sendOverlayMessage(Component.literal("§f[" + keyName(lockKey) + "]§b build here  -  §f[" + keyName(moveKey)
                    + "]§b move  -  /build cancel"));
        } else if (BuildOutline.isAiming()) {
            player.sendOverlayMessage(Component.literal("§bWalk or turn to aim  -  §f[" + keyName(lockKey)
                    + "]§b build here  -  /build cancel"));
        }
    }
}
