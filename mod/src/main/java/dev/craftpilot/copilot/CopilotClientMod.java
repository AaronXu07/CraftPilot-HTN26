package dev.craftpilot.copilot;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client entrypoint. Starts the HTTP bridge on 127.0.0.1:7777, registers the {@code /cp <text>}
 * client command that forwards chat to the Python agent, and hooks the tick-driven helpers
 * ({@link BlockPlacer} for animated placement, {@link CameraOrbit} for the demo camera).
 *
 * <p>No build logic lives in Java: the mod is a thin bridge (plan.md §10).
 */
public class CopilotClientMod implements ClientModInitializer {
    public static final String MOD_ID = "copilot";
    public static final String MOD_VERSION = "0.1.0";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    /** Where the mod listens (the agent talks to this). */
    public static final String BRIDGE_HOST = "127.0.0.1";
    public static final int BRIDGE_PORT = 7777;
    /** Where the Python agent listens (the /cp command talks to this). */
    public static final String AGENT_CHAT_URL = System.getProperty("copilot.agent", "http://127.0.0.1:8000/chat");
    /** Agent base URL (AGENT_CHAT_URL without the trailing /chat) for the /jobs endpoints. */
    public static final String AGENT_BASE_URL = AGENT_CHAT_URL.endsWith("/chat")
            ? AGENT_CHAT_URL.substring(0, AGENT_CHAT_URL.length() - "/chat".length())
            : AGENT_CHAT_URL;

    private static HttpBridgeServer bridge;

    @Override
    public void onInitializeClient() {
        BlockPlacer.register();
        CameraOrbit.register();

        try {
            bridge = new HttpBridgeServer(BRIDGE_HOST, BRIDGE_PORT);
            bridge.start();
            LOGGER.info("[copilot] bridge listening on http://{}:{}", BRIDGE_HOST, BRIDGE_PORT);
        } catch (Exception e) {
            LOGGER.error("[copilot] failed to start bridge server", e);
        }

        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            if (bridge != null) {
                bridge.stop();
            }
        });

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> dispatcher.register(
                ClientCommands.literal("cp")
                        .executes(ctx -> {
                            ctx.getSource().sendFeedback(Component.literal(
                                    "§6[copilot]§r usage: /cp <what to build or change>  e.g. /cp build a castle with four towers"));
                            return 1;
                        })
                        .then(ClientCommands.argument("text", StringArgumentType.greedyString())
                                .executes(ctx -> {
                                    String text = StringArgumentType.getString(ctx, "text");
                                    String player = ctx.getSource().getPlayer().nameAndId().name();
                                    String cmd = text.trim().toLowerCase();
                                    // Job controls talk to /jobs/{id} directly; if no job is known they
                                    // fall through to the agent as ordinary chat (`status` = scene outline).
                                    if ((cmd.equals("cancel") || cmd.equals("stop")) && AgentChatClient.cancelAsync()) {
                                        return 1;
                                    }
                                    if (cmd.equals("status") && AgentChatClient.statusAsync()) {
                                        return 1;
                                    }
                                    ctx.getSource().sendFeedback(Component.literal("§6[copilot]§7 thinking..."));
                                    AgentChatClient.sendAsync(player, text);
                                    return 1;
                                }))));
    }

    /**
     * Print a line in the player's chat, from any thread. Lines the agent already tagged
     * ({@code [cp·plan 0:04] …}, {@code [cp] working…}) are shown with the tag in gold and no
     * extra {@code [copilot]} prefix; anything else gets the prefix.
     */
    public static void chat(String text) {
        String line = formatLine(text);
        Minecraft client = Minecraft.getInstance();
        client.execute(() -> {
            if (client.player != null) {
                client.player.sendSystemMessage(Component.literal(line));
            }
        });
    }

    static String formatLine(String text) {
        String t = text == null ? "" : text;
        if (t.startsWith("[cp")) {
            int close = t.indexOf(']');
            if (close > 0) {
                String tag = t.substring(0, close + 1);
                String rest = t.substring(close + 1);
                if (tag.startsWith("[cp·build")) {
                    return "§a" + tag + "§r" + rest;
                }
                if (tag.startsWith("[cp·critic")) {
                    return "§e" + tag + "§r" + rest;
                }
                return "§6" + tag + "§r" + rest;
            }
        }
        return "§6[copilot]§r " + t;
    }
}
