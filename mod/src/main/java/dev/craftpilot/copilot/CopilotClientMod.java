package dev.craftpilot.copilot;

import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;
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
    public static final String MOD_VERSION = "0.2.0";
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
                ClientCommandManager.literal("build")
                        .executes(ctx -> usage(ctx))
                        .then(ClientCommandManager.literal("status").executes(ctx -> status(ctx)))
                        .then(ClientCommandManager.literal("cancel").executes(ctx -> cancel(ctx)))
                        .then(ClientCommandManager.literal("again")
                                .executes(ctx -> regenerate(ctx, null))
                                .then(ClientCommandManager.argument("seed", IntegerArgumentType.integer())
                                        .executes(ctx -> regenerate(ctx, IntegerArgumentType.getInteger(ctx, "seed")))))
                        .then(ClientCommandManager.literal("edit")
                                .then(ClientCommandManager.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> edit(ctx, StringArgumentType.getString(ctx, "text")))))
                        .then(ClientCommandManager.literal("preview")
                                .then(ClientCommandManager.argument("text", StringArgumentType.greedyString())
                                        .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), false))))
                        .then(ClientCommandManager.argument("text", StringArgumentType.greedyString())
                                .executes(ctx -> build(ctx, StringArgumentType.getString(ctx, "text"), true)))));
    }

    private static int usage(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§r /build <what to build>  |  /build again [seed]"
                + "  |  /build edit <change>  |  /build preview <text>  |  /build cancel  |  /build status"));
        return 1;
    }

    private static int build(CommandContext<FabricClientCommandSource> ctx, String text, boolean place) {
        JsonObject body = base(ctx);
        body.addProperty("text", text);
        body.addProperty("place", place);
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§7 composing \"" + text + "\"..."));
        ServiceClient.postAsync("/build", body);
        return 1;
    }

    private static int edit(CommandContext<FabricClientCommandSource> ctx, String text) {
        JsonObject body = base(ctx);
        body.addProperty("text", text);
        body.addProperty("place", true);
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§7 editing: " + text + "..."));
        ServiceClient.postAsync("/edit", body);
        return 1;
    }

    private static int regenerate(CommandContext<FabricClientCommandSource> ctx, Integer seed) {
        JsonObject body = base(ctx);
        if (seed != null) {
            body.addProperty("seed", seed);
        }
        body.addProperty("place", true);
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§7 building again..."));
        ServiceClient.postAsync("/regenerate", body);
        return 1;
    }

    private static int cancel(CommandContext<FabricClientCommandSource> ctx) {
        int n = BlockPlacer.clear();
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§r cancelled " + n + " queued chunks"));
        return 1;
    }

    private static int status(CommandContext<FabricClientCommandSource> ctx) {
        ctx.getSource().sendFeedback(Text.literal("§6[craftpilot]§r pending " + BlockPlacer.pendingChunks()
                + " chunks / " + BlockPlacer.pendingBlocks() + " blocks, placed " + BlockPlacer.placedTotal()
                + " total; service " + SERVICE_URL + ", bridge http://" + BRIDGE_HOST + ":" + BRIDGE_PORT));
        return 1;
    }

    /** Body fields every service call carries. The service asks the bridge for position and yaw itself. */
    private static JsonObject base(CommandContext<FabricClientCommandSource> ctx) {
        JsonObject body = new JsonObject();
        body.addProperty("player", ctx.getSource().getPlayer().getGameProfile().getName());
        body.addProperty("yaw", ctx.getSource().getPlayer().getYaw());
        return body;
    }

    /** Print a line in the player's chat, from any thread. */
    public static void chat(String text) {
        MinecraftClient client = MinecraftClient.getInstance();
        client.execute(() -> {
            if (client.inGameHud != null) {
                client.inGameHud.getChatHud().addMessage(Text.literal("§6[craftpilot]§r " + text));
            }
        });
    }
}
