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
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Forwards {@code /cp <text>} to the Python agent ({@code POST /chat {"player","text"}}) on a
 * background thread. The agent answers within a second with a job id ({@code {"job_id", "status",
 * "reply"}}); short turns carry the reply inline, long builds push progress and the final answer
 * through the mod's {@code /say} endpoint. {@code /cp status} and {@code /cp cancel} talk to
 * {@code GET /jobs/{id}} and {@code POST /jobs/{id}/cancel}. No request ever waits longer than
 * {@link #REQUEST_TIMEOUT}.
 */
public final class AgentChatClient {
    static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(15);

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

    /** Id of the most recent job started from this client (-1 = none). */
    private static final AtomicInteger CURRENT_JOB = new AtomicInteger(-1);

    private AgentChatClient() {
    }

    public static int currentJob() {
        return CURRENT_JOB.get();
    }

    /** {@code /cp <text>}: start a job; print the inline reply or a "working" line. */
    public static void sendAsync(String player, String text) {
        POOL.submit(() -> {
            try {
                JsonObject body = new JsonObject();
                body.addProperty("player", player);
                body.addProperty("text", text);
                JsonObject json = post(CopilotClientMod.AGENT_CHAT_URL, body);
                if (json == null) {
                    return;
                }
                int jobId = intOr(json, "job_id", -1);
                String status = stringOr(json, "status", "");
                String reply = stringOr(json, "reply", null);
                boolean terminal = isTerminal(status);
                if (jobId >= 0 && !terminal && !json.has("busy")) {
                    CURRENT_JOB.set(jobId);
                }
                if (reply != null && !reply.isBlank()) {
                    chatLines(reply);
                } else if (jobId >= 0) {
                    CopilotClientMod.chat("§7[cp] working (job " + jobId + ")… §8/cp status · /cp cancel");
                } else {
                    CopilotClientMod.chat("§c[cp] unexpected agent reply: " + trim(json.toString()));
                }
            } catch (Exception e) {
                report(e);
            }
        });
    }

    /** {@code /cp status}: one line about the current job. Returns false if no job is known. */
    public static boolean statusAsync() {
        int jobId = CURRENT_JOB.get();
        if (jobId < 0) {
            return false;
        }
        POOL.submit(() -> {
            try {
                JsonObject json = get(jobsUrl(jobId));
                if (json == null) {
                    return;
                }
                CopilotClientMod.chat("§7[cp] " + stringOr(json, "line", "job " + jobId + ": " + stringOr(json, "status", "?")));
            } catch (Exception e) {
                report(e);
            }
        });
        return true;
    }

    /** {@code /cp cancel}: ask the agent to stop the current job. Returns false if no job is known. */
    public static boolean cancelAsync() {
        int jobId = CURRENT_JOB.get();
        if (jobId < 0) {
            return false;
        }
        POOL.submit(() -> {
            try {
                JsonObject json = post(jobsUrl(jobId) + "/cancel", new JsonObject());
                if (json == null) {
                    return;
                }
                CopilotClientMod.chat("§7[cp] " + stringOr(json, "line", "job " + jobId + ": " + stringOr(json, "status", "?")));
            } catch (Exception e) {
                report(e);
            }
        });
        return true;
    }

    // -- HTTP helpers ---------------------------------------------------------------------------

    private static String jobsUrl(int jobId) {
        return CopilotClientMod.AGENT_BASE_URL + "/jobs/" + jobId;
    }

    private static JsonObject post(String url, JsonObject body) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(REQUEST_TIMEOUT)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                .build();
        return send(req);
    }

    private static JsonObject get(String url) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url)).timeout(REQUEST_TIMEOUT).GET().build();
        return send(req);
    }

    /** Sends the request; prints an error line and returns null on a non-2xx status or non-JSON body. */
    private static JsonObject send(HttpRequest req) throws Exception {
        HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (resp.statusCode() / 100 != 2) {
            CopilotClientMod.chat("§cagent error " + resp.statusCode() + ": " + trim(resp.body()));
            return null;
        }
        try {
            return JsonParser.parseString(resp.body()).getAsJsonObject();
        } catch (Exception e) {
            CopilotClientMod.chat(trim(resp.body()));
            return null;
        }
    }

    private static void report(Exception e) {
        if (e instanceof java.net.ConnectException) {
            CopilotClientMod.chat("§cagent not running at " + CopilotClientMod.AGENT_BASE_URL
                    + " (start it with: cd agent && python -m copilot.server --port 8000)");
        } else if (e instanceof java.net.http.HttpTimeoutException) {
            CopilotClientMod.chat("§cagent did not answer within " + REQUEST_TIMEOUT.toSeconds()
                    + " s; it may still be working — try /cp status");
        } else {
            CopilotClientMod.chat("§cchat failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private static void chatLines(String reply) {
        for (String line : reply.split("\n")) {
            if (!line.isBlank()) {
                CopilotClientMod.chat(line);
            }
        }
    }

    private static boolean isTerminal(String status) {
        return status.equals("done") || status.equals("failed") || status.equals("cancelled") || status.equals("timeout");
    }

    private static int intOr(JsonObject json, String key, int fallback) {
        try {
            return json.has(key) && !json.get(key).isJsonNull() ? json.get(key).getAsInt() : fallback;
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String stringOr(JsonObject json, String key, String fallback) {
        try {
            return json.has(key) && !json.get(key).isJsonNull() ? json.get(key).getAsString() : fallback;
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String trim(String s) {
        if (s == null) {
            return "";
        }
        return s.length() > 200 ? s.substring(0, 200) + "..." : s;
    }
}
