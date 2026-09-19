package dev.craftpilot.copilot;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
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
 * Talks to the CraftPilot FastAPI service ({@code craftpilot serve}) on a background thread. The
 * service composes the building, then pushes the blocks back through this mod's bridge, so a call
 * here returns once the blocks are queued, not once they are placed. Composing with the LLM can take
 * a while, hence the generous timeout.
 */
public final class ServiceClient {
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();
    private static final ExecutorService POOL = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "craftpilot-service");
        t.setDaemon(true);
        return t;
    });

    private ServiceClient() {
    }

    /** POST {@code body} to {@code path} on the service and print the result in chat. */
    public static void postAsync(String path, JsonObject body) {
        POOL.submit(() -> {
            String url = CopilotClientMod.SERVICE_URL + path;
            try {
                HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                        .timeout(Duration.ofMinutes(10))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                        .build();
                HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                JsonObject json = parse(resp.body());
                if (resp.statusCode() / 100 != 2) {
                    String detail = json != null && json.has("detail") ? json.get("detail").toString() : trim(resp.body());
                    CopilotClientMod.chat("§cservice error " + resp.statusCode() + ": " + detail);
                    return;
                }
                if (json == null) {
                    CopilotClientMod.chat(trim(resp.body()));
                    return;
                }
                report(json);
            } catch (java.net.ConnectException e) {
                CopilotClientMod.chat("§cservice not running at " + CopilotClientMod.SERVICE_URL
                        + " (start it with: uv run craftpilot serve)");
            } catch (Exception e) {
                CopilotClientMod.chat("§crequest failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
        });
    }

    /** Chat lines for a /build, /edit or /regenerate result: summary, notes, then placement warnings. */
    private static void report(JsonObject json) {
        if (json.has("summary")) {
            CopilotClientMod.chat(json.get("summary").getAsString());
        }
        if (json.has("notes") && json.get("notes").isJsonArray()) {
            for (JsonElement n : json.getAsJsonArray("notes")) {
                CopilotClientMod.chat("§7" + n.getAsString());
            }
        }
        if (json.has("placement") && json.get("placement").isJsonObject()) {
            JsonObject placement = json.getAsJsonObject("placement");
            int invalid = placement.has("invalid") ? placement.get("invalid").getAsInt() : 0;
            if (invalid > 0) {
                StringBuilder sb = new StringBuilder("§c" + invalid + " block states rejected by this game version");
                if (placement.has("invalid_samples")) {
                    JsonArray samples = placement.getAsJsonArray("invalid_samples");
                    if (!samples.isEmpty()) {
                        sb.append(", e.g. ").append(samples.get(0).getAsString());
                    }
                }
                CopilotClientMod.chat(sb.toString());
            }
        }
        if (!json.has("summary") && json.has("ok")) {
            CopilotClientMod.chat(json.toString());
        }
    }

    private static JsonObject parse(String body) {
        try {
            JsonElement el = JsonParser.parseString(body);
            return el.isJsonObject() ? el.getAsJsonObject() : null;
        } catch (Exception e) {
            return null;
        }
    }

    private static String trim(String s) {
        if (s == null) {
            return "";
        }
        return s.length() > 200 ? s.substring(0, 200) + "..." : s;
    }
}
