package dev.craftpilot.copilot;

import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client entrypoint. Starts the HTTP bridge on 127.0.0.1:7777 that the CraftPilot service pushes
 * blocks through, and registers the {@code /build} client command that forwards requests to that
 * service. {@link BlockPlacer} drains the placement queue one chunk per server tick.
 *
 * <p>No build logic lives in Java: the mod is a thin bridge.
 */
public class CopilotClientMod implements ClientModInitializer {
    public static final String MOD_ID = "copilot";
    public static final String MOD_VERSION = "0.4.0";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    /** Where the mod listens (the service talks to this). */
    public static final String BRIDGE_HOST = "127.0.0.1";
    public static final int BRIDGE_PORT = 7777;
    /** Where {@code craftpilot serve} listens (the /build command talks to this). */
    public static final String SERVICE_URL = System.getProperty("craftpilot.service", "http://127.0.0.1:7778");

    private static HttpBridgeServer bridge;

    @Override
    public void onInitializeClient() {
        BlockPlacer.register();
        CameraOrbit.register();
        BuildOutline.register();
        PendingBuild.register();
        Selection.register();

        try {
            bridge = new HttpBridgeServer(BRIDGE_HOST, BRIDGE_PORT);
            bridge.start();
            LOGGER.info("[craftpilot] bridge listening on http://{}:{}", BRIDGE_HOST, BRIDGE_PORT);
        } catch (Exception e) {
            LOGGER.error("[craftpilot] failed to start bridge server", e);
        }

        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            if (bridge != null) {
                bridge.stop();
            }
        });

        // Literal subcommands must be registered before the greedy <text> argument.
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> dispatcher.register(
                ClientCommands.literal("build")
                        .executes(ctx -> usage(ctx))
                        .then(ClientCommands.literal("status").executes(ctx -> status(ctx)))
                        .then(ClientCommands.literal("cancel").executes(ctx -> cancel(ctx)))
                        .then(ClientCommands.literal("go").executes(ctx -> go(ctx)))
                        // Base area: two corners set with the wand or these commands (the block under the crosshair).
                        .then(ClientCommands.literal("pos1").executes(ctx -> pos(ctx, 1)))
                        .then(ClientCommands.literal("pos2").executes(ctx -> pos(ctx, 2)))
                        .then(ClientCommands.literal("sel")
                                .executes(ctx -> sel(ctx))
                                .then(ClientCommands.literal("clear").executes(ctx -> selClear(ctx))))
                        .then(ClientCommands.literal("wand").executes(ctx -> wand(ctx)))
                        .then(ClientCommands.literal("plan")
                                .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> plan(ctx, StringArgumentType.getString(ctx, "text")))))
                        .then(ClientCommands.literal("again")
                                .executes(ctx -> regenerate(ctx, null))
                                .then(ClientCommands.argument("seed", IntegerArgumentType.integer())
                                        .executes(ctx -> regenerate(ctx, IntegerArgumentType.getInteger(ctx, "seed")))))
                        .then(ClientCommands.literal("edit")
                                .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> edit(ctx, StringArgumentType.getString(ctx, "text")))))
                        .then(ClientCommands.literal("preview")
                                .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), false, "auto"))))
                        // Manual overrides of the service's building-vs-object router.
                        .then(ClientCommands.literal("object")
                                .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), true, "object"))))
                        .then(ClientCommands.literal("building")
                                .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), true, "building"))))
                        .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), true, "auto")))));
    }

    private static int usage(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r /build <what to build>  |  /build object <text>"
                + "  |  /build building <text>  |  /build plan <text>"
                + "  |  /build go  |  /build again [seed]  |  /build edit <change>  |  /build preview <text>"
                + "  |  /build pos1 | pos2 | sel | sel clear | wand"
                + "  |  /build cancel  |  /build status   (aim, press [" + PendingBuild.keyName()
                + "]; when the preview shows, [" + PendingBuild.keyName() + "] builds it, [H] moves it; "
                + "with a base area marked the building is sized to it and shown there)"));
        return 1;
    }

    /** {@code /build pos1} / {@code /build pos2}: mark a corner at the block under the crosshair (else under the feet). */
    private static int pos(CommandContext<FabricClientCommandSource> ctx, int which) {
        Minecraft client = Minecraft.getInstance();
        var player = ctx.getSource().getPlayer();
        BlockPos target;
        HitResult hit = client.hitResult;
        if (hit instanceof BlockHitResult bhr && hit.getType() == HitResult.Type.BLOCK) {
            target = bhr.getBlockPos();
        } else {
            target = player.blockPosition().below();
        }
        Selection.set(which, target);
        return 1;
    }

    private static int sel(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r " + Selection.describe()));
        return 1;
    }

    private static int selClear(CommandContext<FabricClientCommandSource> ctx) {
        Selection.clear();
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r base area cleared; /build aims in front of you again"));
        return 1;
    }

    private static int wand(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r wand: hold " + Selection.WAND_ID
                + " - left click a block for corner 1, right click for corner 2 (-Dcraftpilot.wand=<item id> changes it)"));
        return 1;
    }

    /**
     * {@code /build <text>}: aim the wireframe, lock, the service composes and answers with a hologram
     * held at that spot; lock again (or move first) and the service builds it there. {@code kind} is
     * "auto" (the service routes the text to its building generator or its image-to-3D object path),
     * or "object" / "building" to force one ({@code /build object ...}, {@code /build building ...}).
     */
    private static int build(CommandContext<FabricClientCommandSource> ctx, String text, boolean place, String kind) {
        JsonObject body = base(ctx);
        body.addProperty("text", text);
        body.addProperty("place", place);
        body.addProperty("kind", kind);
        if (!place) {
            ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 composing \"" + text + "\"..."));
            ServiceClient.postAsync("/build", body);
            return 1;
        }
        body.addProperty("ghost", true);
        PendingBuild.clearPlan();
        PendingBuild.arm("/build", body, "\"" + text + "\"", 0, 0, 0);
        return 1;
    }

    /** {@code /build plan <text>}: compose and print the plan; nothing is built until {@code /build go}. */
    private static int plan(CommandContext<FabricClientCommandSource> ctx, String text) {
        JsonObject body = base(ctx);
        body.addProperty("text", text);
        PendingBuild.cancel();
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 planning \"" + text + "\"..."));
        ServiceClient.postAsync("/plan", body);
        return 1;
    }

    /** {@code /build go}: preview the pending plan as a hologram, then lock to build it. */
    private static int go(CommandContext<FabricClientCommandSource> ctx) {
        if (!PendingBuild.hasPlan()) {
            ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§c no plan pending; use /build plan <text> first"));
            return 1;
        }
        JsonObject body = base(ctx);
        body.addProperty("ghost", true);
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 generating the preview..."));
        ServiceClient.postAsync("/regenerate", body);
        return 1;
    }

    /** {@code /build edit <change>}: refine the pending plan, or rebuild the last building with the change. */
    private static int edit(CommandContext<FabricClientCommandSource> ctx, String text) {
        JsonObject body = base(ctx);
        body.addProperty("text", text);
        if (PendingBuild.hasPlan()) {
            body.addProperty("place", false);
            body.addProperty("plan", true);
            ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 updating the plan: " + text + "..."));
            ServiceClient.postAsync("/edit", body);
            return 1;
        }
        body.addProperty("ghost", true);
        PendingBuild.cancel();
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 editing: " + text + "..."));
        ServiceClient.postAsync("/edit", body);
        return 1;
    }

    /** {@code /build again [seed]}: preview the last program with a new seed, then lock to build it. */
    private static int regenerate(CommandContext<FabricClientCommandSource> ctx, Integer seed) {
        JsonObject body = base(ctx);
        if (seed != null) {
            body.addProperty("seed", seed);
        }
        body.addProperty("ghost", true);
        PendingBuild.cancel();
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§7 generating the preview..."));
        ServiceClient.postAsync("/regenerate", body);
        return 1;
    }

    private static int cancel(CommandContext<FabricClientCommandSource> ctx) {
        boolean armed = PendingBuild.isArmed();
        boolean plan = PendingBuild.hasPlan();
        PendingBuild.cancel();
        int n = BlockPlacer.clear();
        String what = armed ? "the pending build" : plan ? "the plan" : n + " queued chunks";
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r cancelled " + what));
        return 1;
    }

    private static int status(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Component.literal("§6[craftpilot]§r pending " + BlockPlacer.pendingChunks()
                + " chunks / " + BlockPlacer.pendingBlocks() + " blocks, placed " + BlockPlacer.placedTotal()
                + " (skipped " + BlockPlacer.skippedTotal() + " already right)"
                + " total; service " + SERVICE_URL + ", bridge http://" + BRIDGE_HOST + ":" + BRIDGE_PORT));
        return 1;
    }

    /**
     * Body fields every service call carries. Position and yaw are captured now, when the command is
     * typed; {@link PendingBuild#lock()} overwrites them with the pose the player locked the box at,
     * so the build lands where the outline was, even if they walk off while the service composes.
     */
    private static JsonObject base(CommandContext<FabricClientCommandSource> ctx) {
        JsonObject body = new JsonObject();
        var player = ctx.getSource().getPlayer();
        body.addProperty("player", player.nameAndId().name());
        body.addProperty("yaw", player.getYRot());
        com.google.gson.JsonArray pos = new com.google.gson.JsonArray();
        pos.add(player.getX());
        pos.add(player.getY());
        pos.add(player.getZ());
        body.add("pos", pos);
        if (Selection.isComplete()) {
            body.add("area", Selection.toJson());   // plans, edits and previews all size to the marked base area
        }
        return body;
    }

    /** Print a line in the player's chat, from any thread. */
    public static void chat(String text) {
        Minecraft client = Minecraft.getInstance();
        client.execute(() -> {
            if (client.player != null) {
                client.player.sendSystemMessage(Component.literal("§6[craftpilot]§r " + text));
            }
        });
    }
}
