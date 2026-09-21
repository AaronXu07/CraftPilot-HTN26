package dev.craftpilot.copilot;

import com.google.gson.JsonArray;
import net.fabricmc.fabric.api.event.player.AttackBlockCallback;
import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.gizmos.GizmoStyle;
import net.minecraft.gizmos.Gizmos;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.AABB;

/**
 * The base area a build should stand on, marked with two blocks: left click with the wand (a wooden
 * axe unless {@code -Dcraftpilot.wand=<item id>} says otherwise) sets corner 1, right click corner 2;
 * {@code /build pos1} and {@code /build pos2} do the same for the block under the crosshair. While
 * both corners are set, {@code /build <text>} skips the aiming step: the service sizes the building to
 * the area's footprint, chooses the height, faces it toward the side the player stands on and shows
 * the hologram pinned on the area. The selection stays until {@code /build sel clear}.
 *
 * <p>Only the client reacts to the wand (the integrated server also runs these callbacks in
 * singleplayer, hence the {@code isClientSide} guard), and the click is cancelled so the axe neither
 * chops the block nor uses it.
 */
public final class Selection {
    public static final String WAND_ID = System.getProperty("craftpilot.wand", "minecraft:wooden_axe");
    private static final double INFLATE = 0.01;

    private static BlockPos first;
    private static BlockPos second;

    private Selection() {
    }

    public static void register() {
        AttackBlockCallback.EVENT.register((player, level, hand, pos, direction) -> {
            if (!level.isClientSide() || !holdingWand(player.getMainHandItem())) {
                return InteractionResult.PASS;
            }
            if (hand == InteractionHand.MAIN_HAND) {
                set(1, pos);
            }
            return InteractionResult.FAIL;
        });
        UseBlockCallback.EVENT.register((player, level, hand, hit) -> {
            if (!level.isClientSide() || !holdingWand(player.getMainHandItem())) {
                return InteractionResult.PASS;
            }
            if (hand == InteractionHand.MAIN_HAND) {
                set(2, hit.getBlockPos());
            }
            return InteractionResult.FAIL;
        });
    }

    private static boolean holdingWand(ItemStack stack) {
        if (stack == null || stack.isEmpty()) {
            return false;
        }
        return WAND_ID.equals(BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
    }

    /** Set corner 1 or 2. Repeating the same corner is silent so a held mouse button does not spam chat. */
    public static void set(int which, BlockPos pos) {
        BlockPos p = pos.immutable();
        synchronized (Selection.class) {
            if (which == 1) {
                if (p.equals(first)) {
                    return;
                }
                first = p;
            } else {
                if (p.equals(second)) {
                    return;
                }
                second = p;
            }
        }
        CopilotClientMod.chat("corner " + which + " set to (" + p.getX() + ", " + p.getY() + ", " + p.getZ() + ")"
                + (isComplete() ? " - base area " + describeSize() + "; /build <text> builds on it" : ""));
    }

    public static synchronized void clear() {
        first = null;
        second = null;
    }

    public static synchronized boolean isComplete() {
        return first != null && second != null;
    }

    public static synchronized BlockPos min() {
        return new BlockPos(Math.min(first.getX(), second.getX()), Math.min(first.getY(), second.getY()),
                Math.min(first.getZ(), second.getZ()));
    }

    public static synchronized BlockPos max() {
        return new BlockPos(Math.max(first.getX(), second.getX()), Math.max(first.getY(), second.getY()),
                Math.max(first.getZ(), second.getZ()));
    }

    /** "15 x 13 blocks" (x by z), for chat. */
    public static String describeSize() {
        if (!isComplete()) {
            return "not set";
        }
        BlockPos lo = min();
        BlockPos hi = max();
        return (hi.getX() - lo.getX() + 1) + " x " + (hi.getZ() - lo.getZ() + 1) + " blocks";
    }

    public static String describe() {
        BlockPos a;
        BlockPos b;
        synchronized (Selection.class) {
            a = first;
            b = second;
        }
        if (a == null && b == null) {
            return "no base area marked (wand: " + WAND_ID + ", or /build pos1 and /build pos2)";
        }
        String s = "base area: corner 1 " + (a == null ? "unset" : "(" + a.getX() + ", " + a.getY() + ", " + a.getZ() + ")")
                + ", corner 2 " + (b == null ? "unset" : "(" + b.getX() + ", " + b.getY() + ", " + b.getZ() + ")");
        return isComplete() ? s + " - " + describeSize() : s;
    }

    /** {@code [[x, y, z], [x, y, z]]} for the service, or null when incomplete. */
    public static JsonArray toJson() {
        BlockPos a;
        BlockPos b;
        synchronized (Selection.class) {
            if (first == null || second == null) {
                return null;
            }
            a = first;
            b = second;
        }
        JsonArray out = new JsonArray();
        for (BlockPos p : new BlockPos[] {a, b}) {
            JsonArray point = new JsonArray();
            point.add(p.getX());
            point.add(p.getY());
            point.add(p.getZ());
            out.add(point);
        }
        return out;
    }

    /** Draw the marked corners and, when both are set, the area's slab. Called during level extraction. */
    static void emit(double time) {
        BlockPos a;
        BlockPos b;
        synchronized (Selection.class) {
            a = first;
            b = second;
        }
        if (a == null && b == null) {
            return;
        }
        float pulse = (float) (0.75 + 0.2 * Math.sin(time / 10.0));
        GizmoStyle corner = GizmoStyle.stroke(BuildOutline.argb(0.55f, 1.0f, 0.55f, pulse), 2.5f);
        for (BlockPos p : new BlockPos[] {a, b}) {
            if (p != null) {
                Gizmos.cuboid(new AABB(p.getX() - INFLATE, p.getY() - INFLATE, p.getZ() - INFLATE,
                        p.getX() + 1 + INFLATE, p.getY() + 1 + INFLATE, p.getZ() + 1 + INFLATE), corner);
            }
        }
        if (a != null && b != null) {
            int x0 = Math.min(a.getX(), b.getX());
            int y0 = Math.min(a.getY(), b.getY());
            int z0 = Math.min(a.getZ(), b.getZ());
            int x1 = Math.max(a.getX(), b.getX()) + 1;
            int y1 = Math.max(a.getY(), b.getY()) + 1;
            int z1 = Math.max(a.getZ(), b.getZ()) + 1;
            GizmoStyle slab = GizmoStyle.stroke(BuildOutline.argb(0.55f, 1.0f, 0.55f, 0.55f * pulse), 1.5f);
            Gizmos.cuboid(new AABB(x0 - INFLATE, y0 - INFLATE, z0 - INFLATE, x1 + INFLATE, y1 + INFLATE, z1 + INFLATE), slab);
        }
    }
}
