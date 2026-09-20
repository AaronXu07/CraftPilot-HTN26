package dev.craftpilot.copilot;

import net.minecraft.gizmos.Gizmo;
import net.minecraft.gizmos.GizmoPrimitives;
import net.minecraft.gizmos.Gizmos;
import net.minecraft.world.phys.Vec3;

/**
 * Hologram of the building the service generated: one translucent coloured cube per block, outer
 * faces only, drawn inside the {@link BuildOutline} box so the player sees what will be built before
 * committing. The cloud arrives facing south (the engine's native orientation) and is rotated here
 * to face the player with the same quarter-turn rule the service uses when it places the real thing
 * ({@code grid/ops.py rotate_cw}: one clockwise turn maps (x, z) to (D-1-z, x) for blocks, i.e.
 * (x, z) to (D-z, x) for the continuous corner coordinates used here). Colours need no block-state
 * rotation, which is what makes this cheap.
 */
public final class GhostRender {
    private static final float INSET = 0.06f;   // cubes shrink toward their centre so real blocks never z-fight
    private static final float ALPHA_AIM = 0.5f;
    private static final float ALPHA_LOCKED = 0.32f;
    private static final int STRIDE = 14;       // 4 corners x (x, y, z) + rgb + face

    // +x, -x, +y, -y, +z, -z: corner offsets per face, counter-clockwise seen from outside.
    private static final int[][][] FACES = {
            {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}},
            {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}},
            {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
            {{0, 0, 1}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}},
            {{1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}},
            {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
    };
    private static final int[][] DIRS = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    private static final float[] SHADE = {0.85f, 0.85f, 1.0f, 0.6f, 0.75f, 0.75f};

    private static int w, h, d;
    private static int[] voxels;                 // x, y, z, rgb, ...
    private static boolean[] solid;              // (x * h + y) * d + z
    private static final float[][] faceCache = new float[4][];
    private static boolean present = false;

    private GhostRender() {
    }

    public static synchronized void set(int width, int height, int depth, int[] flat) {
        w = Math.max(1, width);
        h = Math.max(1, height);
        d = Math.max(1, depth);
        voxels = flat;
        solid = new boolean[w * h * d];
        for (int i = 0; i + 3 < flat.length; i += 4) {
            int x = flat[i], y = flat[i + 1], z = flat[i + 2];
            if (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d) {
                solid[(x * h + y) * d + z] = true;
            }
        }
        for (int t = 0; t < 4; t++) {
            faceCache[t] = null;
        }
        present = true;
    }

    public static synchronized void clear() {
        present = false;
        voxels = null;
        solid = null;
        for (int t = 0; t < 4; t++) {
            faceCache[t] = null;
        }
    }

    public static synchronized boolean isPresent() {
        return present;
    }

    private static boolean isSolid(int x, int y, int z) {
        if (x < 0 || x >= w || y < 0 || y >= h || z < 0 || z >= d) {
            return false;
        }
        return solid[(x * h + y) * d + z];
    }

    /** Exposed, inset faces for {@code turns} clockwise quarter turns, in rotated local coordinates. */
    private static float[] faces(int turns) {
        float[] cached = faceCache[turns];
        if (cached != null) {
            return cached;
        }
        int count = 0;
        for (int i = 0; i + 3 < voxels.length; i += 4) {
            int x = voxels[i], y = voxels[i + 1], z = voxels[i + 2];
            for (int f = 0; f < 6; f++) {
                if (!isSolid(x + DIRS[f][0], y + DIRS[f][1], z + DIRS[f][2])) {
                    count++;
                }
            }
        }
        float[] out = new float[count * STRIDE];
        int n = 0;
        for (int i = 0; i + 3 < voxels.length; i += 4) {
            int x = voxels[i], y = voxels[i + 1], z = voxels[i + 2], rgb = voxels[i + 3];
            for (int f = 0; f < 6; f++) {
                if (isSolid(x + DIRS[f][0], y + DIRS[f][1], z + DIRS[f][2])) {
                    continue;
                }
                for (int c = 0; c < 4; c++) {
                    // Corner pulled toward the cube centre, then rotated like the engine rotates the grid.
                    float cx = x + (FACES[f][c][0] == 1 ? 1 - INSET : INSET);
                    float cy = y + (FACES[f][c][1] == 1 ? 1 - INSET : INSET);
                    float cz = z + (FACES[f][c][2] == 1 ? 1 - INSET : INSET);
                    float rw = w, rd = d;
                    for (int t = 0; t < turns; t++) {
                        float nx = rd - cz;
                        float nz = cx;
                        cx = nx;
                        cz = nz;
                        float tmp = rw;
                        rw = rd;
                        rd = tmp;
                    }
                    out[n++] = cx;
                    out[n++] = cy;
                    out[n++] = cz;
                }
                out[n++] = Float.intBitsToFloat(rgb);
                out[n++] = f;
            }
        }
        faceCache[turns] = out;
        return out;
    }

    /**
     * Emit the hologram as one gizmo with its rotated min corner at world ({@code ox, oy, oz}). Faces
     * whose lowest corner is at or below {@code hideBelowY} (local, exclusive of the row itself) are
     * skipped so real blocks visibly replace the ghost from the bottom up. Must be called while the
     * game collects gizmos (level extraction).
     */
    static void emit(double ox, double oy, double oz, int turns, float hideBelowY, boolean aiming) {
        float[] fl;
        synchronized (GhostRender.class) {
            if (!present || voxels == null) {
                return;
            }
            fl = faces(turns & 3);
        }
        float alpha = aiming ? ALPHA_AIM : ALPHA_LOCKED;
        Gizmos.addGizmo(new Cloud(fl, ox, oy, oz, hideBelowY, alpha));
    }

    /** The whole voxel cloud as a single gizmo: one translucent quad per visible face. */
    private record Cloud(float[] fl, double ox, double oy, double oz, float hideBelowY, float alpha) implements Gizmo {
        @Override
        public void emit(GizmoPrimitives out, float alphaScale) {
            int a = Math.round(Math.max(0f, Math.min(1f, alpha * alphaScale)) * 255);
            Vec3[] corner = new Vec3[4];
            for (int i = 0; i + STRIDE - 1 < fl.length; i += STRIDE) {
                float lowest = Math.min(Math.min(fl[i + 1], fl[i + 4]), Math.min(fl[i + 7], fl[i + 10]));
                if (lowest < hideBelowY) {
                    continue;
                }
                int rgb = Float.floatToRawIntBits(fl[i + 12]);
                float shade = SHADE[(int) fl[i + 13]];
                int r = Math.round(((rgb >> 16) & 0xFF) * shade);
                int g = Math.round(((rgb >> 8) & 0xFF) * shade);
                int b = Math.round((rgb & 0xFF) * shade);
                for (int c = 0; c < 4; c++) {
                    corner[c] = new Vec3(ox + fl[i + c * 3], oy + fl[i + c * 3 + 1], oz + fl[i + c * 3 + 2]);
                }
                out.addQuad(corner[0], corner[1], corner[2], corner[3], (a << 24) | (r << 16) | (g << 8) | b);
            }
        }
    }
}
