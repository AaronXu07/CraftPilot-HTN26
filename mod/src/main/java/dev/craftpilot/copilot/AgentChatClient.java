package dev.craftpilot.copilot;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Forwards {@code /cp <text>} to the Python agent ({@code POST /chat {"player","text"}}) on a
 * background thread and prints the {@code reply} in chat. Builds can take minutes, so the request
 * timeout is generous; progress arrives separately through the mod's {@code /say} endpoint.
 */
public final class AgentChatClient {
    // HTTP/1.1 only: the default HTTP_2 client sends an `Upgrade: h2c` header on plain-http requests,
    // which the agent's uvicorn server answers with 500 Internal Server Error.
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(5))
            .build();
    private static final ExecutorService POOL = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "copilot-chat");
        t.setDaemon(true);
        return t;
    });

    private AgentChatClient() {
    }

    public static void sendAsync(String player, String text) {
        POOL.submit(() -> {
            try {
                JsonObject body = new JsonObject();
                body.addProperty("player", player);
                body.addProperty("text", text);
                HttpRequest req = HttpRequest.newBuilder(URI.create(CopilotClientMod.AGENT_CHAT_URL))
                        .timeout(Duration.ofMinutes(10))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                        .build();
                HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                if (resp.statusCode() / 100 != 2) {
                    CopilotClientMod.chat("§cagent error " + resp.statusCode() + ": " + trim(resp.body()));
                    return;
                }
                String reply;
                try {
                    JsonObject json = JsonParser.parseString(resp.body()).getAsJsonObject();
                    reply = json.has("reply") && !json.get("reply").isJsonNull() ? json.get("reply").getAsString() : resp.body();
                } catch (Exception e) {
                    reply = resp.body();
                }
                for (String line : reply.split("\n")) {
                    if (!line.isBlank()) {
                        CopilotClientMod.chat(line);
                    }
                }
            } catch (java.net.ConnectException e) {
                CopilotClientMod.chat("§cagent not running at " + CopilotClientMod.AGENT_CHAT_URL
                        + " (start it with: cd agent && uvicorn copilot.server:app --port 8000)");
            } catch (Exception e) {
                CopilotClientMod.chat("§cchat failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
        });
    }

    private static String trim(String s) {
        if (s == null) {
            return "";
        }
        return s.length() > 200 ? s.substring(0, 200) + "..." : s;
    }
}
