package dev.craftpilot.copilot;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import net.minecraft.SharedConstants;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.registry.RegistryKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The HTTP bridge the CraftPilot service pushes blocks through (see mod/README.md). JSON in, JSON out. Every world access is marshalled onto
 * the integrated server thread with {@code server.submit(...)}; chat goes to the client thread.
 */
public final class HttpBridgeServer {
    private static final Gson GSON = new Gson();
    private static final long MAX_SCAN_BLOCKS = 2_000_000L;
    private static final int DEFAULT_FLAGS = Block.NOTIFY_ALL; // 3: notify neighbours + clients

    private final HttpServer server;
    private final ExecutorService executor;

    public HttpBridgeServer(String host, int port) throws IOException {
        this.server = HttpServer.create(new InetSocketAddress(host, port), 16);
        this.executor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "copilot-bridge");
            t.setDaemon(true);
            return t;
        });
        server.setExecutor(executor);
        server.createContext("/health", wrap(this::health));
        server.createContext("/player", wrap(this::player));
        server.createContext("/scan", wrap(this::scan));
        server.createContext("/setblocks/status", wrap(this::setblocksStatus));
        server.createContext("/setblocks/cancel", wrap(this::setblocksCancel));
        server.createContext("/setblocks", wrap(this::setblocks));
        server.createContext("/say", wrap(this::say));
        server.createContext("/blocks", wrap(this::blocks));
        server.createContext("/camera", wrap(this::camera));
        server.createContext("/outline", wrap(this::outline));
    }

    public void start() {
        server.start();
    }

    public void stop() {
        server.stop(0);
        executor.shutdownNow();
    }

    // ------------------------------------------------------------------ plumbing

    @FunctionalInterface
    private interface Route {
        JsonElement handle(HttpExchange ex, JsonObject body) throws Exception;
    }

    /** Thrown by routes to produce a specific HTTP status. */
    private static final class HttpError extends RuntimeException {
        final int status;

        HttpError(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    private HttpHandler wrap(Route route) {
        return ex -> {
            try {
                JsonObject body = readJson(ex);
                JsonElement out = route.handle(ex, body);
                respond(ex, 200, out);
            } catch (HttpError e) {
                respond(ex, e.status, error(e.getMessage()));
            } catch (Exception e) {
                CopilotClientMod.LOGGER.warn("[copilot] {} {} failed", ex.getRequestMethod(), ex.getRequestURI(), e);
                respond(ex, 500, error(e.getClass().getSimpleName() + ": " + e.getMessage()));
            }
        };
    }

    private static JsonObject readJson(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            byte[] bytes = in.readAllBytes();
            if (bytes.length == 0) {
                return new JsonObject();
            }
            String text = new String(bytes, StandardCharsets.UTF_8).trim();
            if (text.isEmpty()) {
                return new JsonObject();
            }
            JsonElement el = JsonParser.parseString(text);
            return el.isJsonObject() ? el.getAsJsonObject() : new JsonObject();
        } catch (RuntimeException e) {
            throw new HttpError(400, "invalid JSON body: " + e.getMessage());
        }
    }

    private static void respond(HttpExchange ex, int status, JsonElement json) throws IOException {
        byte[] bytes = GSON.toJson(json).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static JsonObject error(String message) {
        JsonObject o = new JsonObject();
        o.addProperty("ok", false);
        o.addProperty("error", message);
        return o;
    }

    private static JsonObject ok() {
        JsonObject o = new JsonObject();
        o.addProperty("ok", true);
        return o;
    }

    private static MinecraftClient client() {
        return MinecraftClient.getInstance();
    }

    /** The integrated server, or a 409 if no world is loaded. */
    private static MinecraftServer requireServer() {
        MinecraftServer s = client().getServer();
        if (s == null || client().world == null) {
            throw new HttpError(409, "no world loaded (open a single-player world first)");
        }
        return s;
    }

    private static ClientPlayerEntity requirePlayer() {
        ClientPlayerEntity p = client().player;
        if (p == null) {
            throw new HttpError(409, "no player (open a single-player world first)");
        }
        return p;
    }

    private static RegistryKey<World> playerDimension() {
        ClientPlayerEntity p = client().player;
        return p != null ? p.getWorld().getRegistryKey() : World.OVERWORLD;
    }

    private static int[] vec3i(JsonObject body, String key) {
        if (!body.has(key) || !body.get(key).isJsonArray() || body.getAsJsonArray(key).size() != 3) {
            throw new HttpError(400, "'" + key + "' must be [x, y, z]");
        }
        JsonArray a = body.getAsJsonArray(key);
        return new int[] {(int) Math.floor(a.get(0).getAsDouble()), (int) Math.floor(a.get(1).getAsDouble()),
                (int) Math.floor(a.get(2).getAsDouble())};
    }

    private static double[] vec3d(JsonObject body, String key) {
        if (!body.has(key) || !body.get(key).isJsonArray() || body.getAsJsonArray(key).size() != 3) {
            throw new HttpError(400, "'" + key + "' must be [x, y, z]");
        }
        JsonArray a = body.getAsJsonArray(key);
        return new double[] {a.get(0).getAsDouble(), a.get(1).getAsDouble(), a.get(2).getAsDouble()};
    }

    private static JsonArray arr(double... v) {
        JsonArray a = new JsonArray();
        for (double d : v) {
            a.add(d);
        }
        return a;
    }

    private static JsonArray arr(int... v) {
        JsonArray a = new JsonArray();
        for (int d : v) {
            a.add(d);
        }
        return a;
    }

    // ------------------------------------------------------------------ routes

    private JsonElement health(HttpExchange ex, JsonObject body) {
        MinecraftClient client = client();
        JsonObject o = ok();
        o.addProperty("mod_version", CopilotClientMod.MOD_VERSION);
        o.addProperty("mc_version", SharedConstants.getGameVersion().getName());
        String jar = findClientJar(client);
        if (jar != null) {
            o.addProperty("client_jar", jar);
        } else {
            o.add("client_jar", null);
        }
        o.addProperty("world_loaded", client.world != null && client.getServer() != null);
        if (client.player != null) {
            o.addProperty("player", client.player.getGameProfile().getName());
        } else {
            o.add("player", null);
        }
        o.addProperty("pending_chunks", BlockPlacer.pendingChunks());
        return o;
    }

    private static String findClientJar(MinecraftClient client) {
        try {
            URL loc = MinecraftClient.class.getProtectionDomain().getCodeSource().getLocation();
            if (loc != null) {
                File f = new File(loc.toURI());
                if (f.isFile() && f.getName().endsWith(".jar")) {
                    return f.getAbsolutePath();
                }
            }
        } catch (Exception ignored) {
            // fall through to the launcher layout
        }
        try {
            String v = SharedConstants.getGameVersion().getName();
            File f = new File(client.runDirectory, "versions/" + v + "/" + v + ".jar");
            if (f.isFile()) {
                return f.getAbsolutePath();
            }
        } catch (Exception ignored) {
            // unknown
        }
        return null;
    }

    private JsonElement player(HttpExchange ex, JsonObject body) {
        MinecraftClient client = client();
        ClientPlayerEntity p = requirePlayer();
        JsonObject o = new JsonObject();
        o.addProperty("name", p.getGameProfile().getName());
        o.add("pos", arr(p.getX(), p.getY(), p.getZ()));
        o.addProperty("yaw", p.getYaw());
        o.addProperty("pitch", p.getPitch());
        o.addProperty("facing", p.getHorizontalFacing().getName());
        JsonElement looking = null;
        HitResult hit = client.crosshairTarget;
        if (hit instanceof BlockHitResult bhr && hit.getType() == HitResult.Type.BLOCK && client.world != null) {
            BlockPos bp = bhr.getBlockPos();
            JsonObject l = new JsonObject();
            l.add("pos", arr(bp.getX(), bp.getY(), bp.getZ()));
            l.addProperty("block", WorldOps.stringify(client.world.getBlockState(bp)));
            l.addProperty("side", bhr.getSide().getName());
            looking = l;
        }
        o.add("looking_at", looking);
        o.addProperty("dimension", p.getWorld().getRegistryKey().getValue().toString());
        return o;
    }

    private JsonElement scan(HttpExchange ex, JsonObject body) {
        MinecraftServer server = requireServer();
        int[] lo = vec3i(body, "min");
        int[] hi = vec3i(body, "max");
        long volume = (long) (Math.abs(hi[0] - lo[0]) + 1) * (Math.abs(hi[1] - lo[1]) + 1) * (Math.abs(hi[2] - lo[2]) + 1);
        if (volume > MAX_SCAN_BLOCKS) {
            throw new HttpError(400, "scan volume " + volume + " exceeds the " + MAX_SCAN_BLOCKS + " block cap");
        }
        RegistryKey<World> dim = playerDimension();
        return server.submit(() -> {
            ServerWorld world = server.getWorld(dim);
            if (world == null) {
                world = server.getOverworld();
            }
            return WorldOps.scan(world, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
        }).join();
    }

    private JsonElement setblocks(HttpExchange ex, JsonObject body) {
        requireServer();
        int flags = body.has("flags") ? body.get("flags").getAsInt() : DEFAULT_FLAGS;
        // Skip the connectivity pass when the caller already sends complete states.
        boolean postprocess = !body.has("postprocess") || body.get("postprocess").getAsBoolean();
        RegistryKey<World> dim = playerDimension();
        List<JsonObject> chunkSpecs = new ArrayList<>();
        if (body.has("chunks") && body.get("chunks").isJsonArray()) {
            for (JsonElement c : body.getAsJsonArray("chunks")) {
                chunkSpecs.add(c.getAsJsonObject());
            }
        } else if (body.has("blocks") && body.get("blocks").isJsonArray()) {
            JsonObject single = new JsonObject();
            single.add("blocks", body.getAsJsonArray("blocks"));
            single.addProperty("delay_ms", 0);
            chunkSpecs.add(single);
        } else {
            throw new HttpError(400, "body needs 'chunks': [{blocks, delay_ms}] or 'blocks': [[x,y,z,state],...]");
        }

        int queued = 0;
        int invalid = 0;
        List<String> invalidSamples = new ArrayList<>();
        List<BlockPlacer.Chunk> chunks = new ArrayList<>();
        for (JsonObject spec : chunkSpecs) {
            List<BlockPlacer.Placement> placements = new ArrayList<>();
            JsonArray blocks = spec.has("blocks") ? spec.getAsJsonArray("blocks") : new JsonArray();
            for (JsonElement el : blocks) {
                JsonArray b = el.getAsJsonArray();
                if (b.size() < 4) {
                    invalid++;
                    continue;
                }
                String stateText = b.get(3).getAsString();
                try {
                    BlockState state = WorldOps.parseState(stateText);
                    BlockPos pos = new BlockPos((int) Math.floor(b.get(0).getAsDouble()),
                            (int) Math.floor(b.get(1).getAsDouble()), (int) Math.floor(b.get(2).getAsDouble()));
                    placements.add(new BlockPlacer.Placement(pos, state));
                } catch (CommandSyntaxException | RuntimeException e) {
                    invalid++;
                    if (invalidSamples.size() < 5) {
                        invalidSamples.add(stateText);
                    }
                }
            }
            int delayMs = spec.has("delay_ms") ? spec.get("delay_ms").getAsInt() : 0;
            int delayTicks = (int) Math.ceil(delayMs / 50.0);
            queued += placements.size();
            chunks.add(new BlockPlacer.Chunk(dim, placements, delayTicks, flags, postprocess));
        }
        BlockPlacer.enqueue(chunks);
        JsonObject o = new JsonObject();
        o.addProperty("queued", queued);
        o.addProperty("chunks", chunks.size());
        o.addProperty("invalid", invalid);
        if (!invalidSamples.isEmpty()) {
            JsonArray s = new JsonArray();
            for (String v : invalidSamples) {
                s.add(v);
            }
            o.add("invalid_samples", s);
        }
        return o;
    }

    private JsonElement setblocksStatus(HttpExchange ex, JsonObject body) {
        JsonObject o = new JsonObject();
        o.addProperty("pending_chunks", BlockPlacer.pendingChunks());
        o.addProperty("pending_blocks", BlockPlacer.pendingBlocks());
        o.addProperty("placed_total", BlockPlacer.placedTotal());
        o.addProperty("skipped_total", BlockPlacer.skippedTotal());
        o.addProperty("postprocessed", BlockPlacer.postprocessedTotal());
        return o;
    }

    private JsonElement setblocksCancel(HttpExchange ex, JsonObject body) {
        JsonObject o = new JsonObject();
        o.addProperty("ok", true);
        o.addProperty("cleared_chunks", BlockPlacer.clear());
        return o;
    }

    private JsonElement say(HttpExchange ex, JsonObject body) {
        String text = body.has("text") ? body.get("text").getAsString() : "";
        if (text.isEmpty()) {
            throw new HttpError(400, "'text' is required");
        }
        for (String line : text.split("\n")) {
            if (!line.isBlank()) {
                CopilotClientMod.chat(line);
            }
        }
        return ok();
    }

    private JsonElement blocks(HttpExchange ex, JsonObject body) {
        return WorldOps.blockDump();
    }

    /** Show ({@code min}, {@code max}, {@code phase}) or hide ({@code clear: true}) the in-progress build box. */
    private JsonElement outline(HttpExchange ex, JsonObject body) {
        if (body.has("clear") && body.get("clear").getAsBoolean()) {
            BuildOutline.clear();
            return ok();
        }
        int[] lo = vec3i(body, "min");
        int[] hi = vec3i(body, "max");
        String phase = body.has("phase") ? body.get("phase").getAsString() : "generating";
        BuildOutline.set(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], BuildOutline.phaseOf(phase));
        return ok();
    }

    private JsonElement camera(HttpExchange ex, JsonObject body) {
        requirePlayer();
        String mode = body.has("mode") ? body.get("mode").getAsString() : "orbit";
        if ("return".equals(mode) || "stop".equals(mode)) {
            CameraOrbit.stop();
            return ok();
        }
        if (!"orbit".equals(mode)) {
            throw new HttpError(400, "mode must be 'orbit' or 'return'");
        }
        double[] c = vec3d(body, "center");
        double radius = body.has("radius") ? body.get("radius").getAsDouble() : 30.0;
        double seconds = body.has("seconds") ? body.get("seconds").getAsDouble() : 10.0;
        CameraOrbit.start(c[0], c[1], c[2], radius, seconds);
        JsonObject o = ok();
        o.addProperty("seconds", seconds);
        return o;
    }
}
