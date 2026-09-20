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
                    BuildOutline.clear();
                    return;
                }
                if (json == null) {
                    CopilotClientMod.chat(trim(resp.body()));
                    BuildOutline.clear();
                    return;
                }
                if (json.has("ghost") && json.get("ghost").isJsonObject()) {
                    armGhost(json, body);
                } else if (!json.has("placement")) {
                    BuildOutline.clear(); // preview or a build that never reached the queue
                }
                report(json);
            } catch (java.net.ConnectException e) {
                CopilotClientMod.chat("§cservice not running at " + CopilotClientMod.SERVICE_URL
                        + " (start it with: uv run craftpilot serve)");
                BuildOutline.clear();
            } catch (Exception e) {
                CopilotClientMod.chat("§crequest failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                BuildOutline.clear();
            }
        });
    }

    /**
     * The service answered with a hologram: hand it to {@link PendingBuild} with the body that will
     * commit it ({@code /regenerate} with the same seed, so the build is the preview block for block).
     */
    private static void armGhost(JsonObject json, JsonObject request) {
        JsonObject ghost = json.getAsJsonObject("ghost");
        JsonArray flat = ghost.getAsJsonArray("blocks");
        int[] voxels = new int[flat.size()];
        for (int i = 0; i < voxels.length; i++) {
            voxels[i] = flat.get(i).getAsInt();
        }
        JsonObject commit = new JsonObject();
        if (request.has("player")) {
            commit.add("player", request.get("player"));
        }
        commit.addProperty("place", true);
        if (json.has("seed")) {
            commit.add("seed", json.get("seed"));
        }
        PendingBuild.armGhost("/regenerate", commit, ghost.get("width").getAsInt(), ghost.get("height").getAsInt(),
                ghost.get("depth").getAsInt(), voxels);
    }

    /** Chat lines for a service result: summary, plan lines, notes, then placement warnings. */
    private static void report(JsonObject json) {
        if (json.has("summary")) {
            CopilotClientMod.chat(json.get("summary").getAsString());
        }
        if (json.has("plan") && json.get("plan").isJsonArray() && !json.has("ghost")) {
            for (JsonElement line : json.getAsJsonArray("plan")) {
                CopilotClientMod.chat("§7  " + line.getAsString());
            }
            int w = 0, h = 0, d = 0;
            if (json.has("bounds") && json.get("bounds").isJsonObject()) {
                JsonObject b = json.getAsJsonObject("bounds");
                w = b.get("width").getAsInt();
                h = b.get("height").getAsInt();
                d = b.get("depth").getAsInt();
            }
            PendingBuild.setPlan(w, h, d);
            CopilotClientMod.chat("§6[craftpilot]§7 /build edit <change> to refine, /build go to preview and build it");
        }
        if (json.has("notes") && json.get("notes").isJsonArray()) {
            for (JsonElement n : json.getAsJsonArray("notes")) {
                CopilotClientMod.chat("§7" + n.getAsString());
            }
        }
        if (json.has("placement") && json.get("placement").isJsonObject()) {
            PendingBuild.clearPlan(); // the plan is built; /build edit goes back to rebuilding
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
