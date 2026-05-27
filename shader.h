#pragma once
#include "vec3.h"
#include "geodesic.h"   // RayOutcome, TraceResult, R_DISK_MIN, R_DISK_MAX
#include "config.h"     // all tunable look parameters (cfg::...)
#include <cmath>

#ifndef __CUDACC__
#include <vector>
#include <cstdio>
#include "stb_image.h"  // single-header image loader (vendored, public domain)
#endif

// -----------------------------------------------------------------------
// Trevor's module: pixel shading
//
// Three cases:
//   CAPTURED → black (inside event horizon)
//   DISK     → layered procedural emission (Shakura-Sunyaev T(r) profile
//              + fBm turbulence + Keplerian banding + hot spots), HDR
//   ESCAPED  → equirectangular skybox sample (procedural starfield fallback)
//
// All look parameters live in config.h so this file holds only algorithm.
// -----------------------------------------------------------------------

// === Deterministic hashing (CUDA-safe, no stdlib random) =================
__host__ __device__ inline unsigned int uhash(unsigned int n) {
    n = (n ^ 61u) ^ (n >> 16u);
    n *= 9u;
    n ^= n >> 4u;
    n *= 0x27d4eb2du;
    n ^= n >> 15u;
    return n;
}

__host__ __device__ inline float hash01(unsigned int n) {
    return (float)(uhash(n) & 0xFFFFFFu) / (float)0x1000000u;
}

// === Value noise + fBm ===================================================
// Lattice value noise: hash the four surrounding integer corners and
// smoothstep-interpolate.  Cheap and CUDA-friendly.
// GPU: this evaluates per thread; if it becomes a bottleneck, precompute a
// small noise texture in __constant__/texture memory and bilinearly fetch it.
__host__ __device__ inline float lattice01(int xi, int yi) {
    unsigned int h = (unsigned int)(xi * 73856093) ^ (unsigned int)(yi * 19349663);
    return hash01(h);
}

__host__ __device__ inline float smoothfade(float t) { return t * t * (3.0f - 2.0f * t); }

__host__ __device__ inline float noise(float x, float y) {
    float xf = floorf(x), yf = floorf(y);
    int   xi = (int)xf,    yi = (int)yf;
    float fx = x - xf,     fy = y - yf;

    float v00 = lattice01(xi,     yi);
    float v10 = lattice01(xi + 1, yi);
    float v01 = lattice01(xi,     yi + 1);
    float v11 = lattice01(xi + 1, yi + 1);

    float ux = smoothfade(fx), uy = smoothfade(fy);
    float a  = v00 + (v10 - v00) * ux;
    float b  = v01 + (v11 - v01) * ux;
    return a + (b - a) * uy;                 // [0, 1]
}

// fBm: sum `octaves` of noise with gain 0.5 and lacunarity 2.0, normalized.
__host__ __device__ inline float fbm(float x, float y, int octaves) {
    float sum = 0.0f, amp = 0.5f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum  += amp * noise(x * freq, y * freq);
        norm += amp;
        amp  *= 0.5f;   // gain
        freq *= 2.0f;   // lacunarity
    }
    return (norm > 0.0f) ? sum / norm : 0.0f;  // [0, 1]
}

// === Procedural starfield (skybox fallback) ==============================
// Map an escaped ray direction to a star-field RGB sample.
__host__ __device__ inline Vec3 sample_starfield(Vec3 dir) {
    float len = dir.norm();
    float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));  // [0, π]
    float phi   = atan2f(dz, dx) + 3.14159265f;           // [0, 2π]

    const int GRID = 256;
    int iu = (int)(phi   / (2.0f * 3.14159265f) * (float)GRID) % GRID;
    int iv = (int)(theta /        3.14159265f    * (float)GRID) % GRID;
    unsigned int cell = (unsigned int)(iu * GRID + iv);

    Vec3 color(0.004f, 0.004f, 0.006f);   // near-black, faintly cool background

    if (hash01(cell) > 0.9994f) {          // sparse, dim stars (reference shows few)
        float brightness = 0.2f + 0.4f * hash01(cell + 1000u);
        float temp       = hash01(cell + 2000u);   // 0=red, 1=blue-white
        Vec3 star;
        if (temp > 0.6f) {
            float t = (temp - 0.6f) / 0.4f;
            star = Vec3(0.7f + 0.3f*t, 0.85f + 0.15f*t, 1.0f) * brightness;
        } else {
            float t = temp / 0.6f;
            star = Vec3(1.0f, 0.5f + 0.45f*t, 0.1f + 0.5f*t) * brightness;
        }
        color = star;
    }
    return color;
}

// === Equirectangular skybox texture (host only) ==========================
// Loaded once at startup from cfg::SKYBOX_PATH. stbi_loadf converts 8-bit
// LDR images to linear float, which is what we want before tone mapping.
// GPU: replace this host buffer with a cudaTextureObject_t and tex2D() fetch.
#ifndef __CUDACC__
struct SkyboxTexture {
    int w = 0, h = 0;
    std::vector<float> rgb;   // w*h*3, linear
    bool loaded = false;
};

inline SkyboxTexture& skybox() { static SkyboxTexture s; return s; }

// Returns true if a skybox image was loaded; false → procedural fallback.
inline bool load_skybox(const char* path) {
    int w = 0, h = 0, n = 0;
    float* data = stbi_loadf(path, &w, &h, &n, 3);
    if (!data) {
        fprintf(stderr,
                "[skybox] could not open '%s' (%s) — falling back to "
                "procedural starfield.\n", path, stbi_failure_reason());
        return false;
    }
    SkyboxTexture& s = skybox();
    s.w = w; s.h = h;
    s.rgb.assign(data, data + (size_t)w * h * 3);
    s.loaded = true;
    stbi_image_free(data);
    printf("[skybox] loaded %dx%d equirectangular map from %s\n", w, h, path);
    return true;
}

inline Vec3 skybox_texel(int x, int y) {
    const SkyboxTexture& s = skybox();
    size_t i = ((size_t)y * s.w + x) * 3;
    return Vec3(s.rgb[i], s.rgb[i + 1], s.rgb[i + 2]);
}

// Bilinear sample, wrapping horizontally (u) and clamping vertically (v).
inline Vec3 sample_skybox_texture(float u, float v) {
    const SkyboxTexture& s = skybox();
    float fx = u * s.w - 0.5f, fy = v * s.h - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - x0, ty = fy - y0;

    int xw0 = ((x0 % s.w) + s.w) % s.w;
    int xw1 = ((x0 + 1) % s.w + s.w) % s.w;
    int yc0 = y0 < 0 ? 0 : (y0 >= s.h ? s.h - 1 : y0);
    int yc1 = (y0 + 1) < 0 ? 0 : (y0 + 1 >= s.h ? s.h - 1 : y0 + 1);

    Vec3 top = skybox_texel(xw0, yc0) * (1.0f - tx) + skybox_texel(xw1, yc0) * tx;
    Vec3 bot = skybox_texel(xw0, yc1) * (1.0f - tx) + skybox_texel(xw1, yc1) * tx;
    return (top * (1.0f - ty) + bot * ty) * cfg::SKYBOX_INTENSITY;
}
#endif  // !__CUDACC__

// Dispatch: textured skybox if loaded, else procedural starfield.
__host__ __device__ inline Vec3 sample_skybox(Vec3 dir) {
#ifndef __CUDACC__
    if (skybox().loaded) {
        float len = dir.norm();
        float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;
        float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));   // [0, π]
        float phi   = atan2f(dz, dx) + 3.14159265f;            // [0, 2π]
        float u = phi   / (2.0f * 3.14159265f);
        float v = theta /        3.14159265f;
        return sample_skybox_texture(u, v);
    }
#endif
    return sample_starfield(dir);
}

// === Accretion disk shading ==============================================
// Map normalized temperature t∈[0,1] through hand-tuned control colors.
// Tuned to the Interstellar palette: warm throughout, white-hot center,
// no blue tint (the reference disk is pale/pinkish-white, not blue-white).
__host__ __device__ inline Vec3 temperature_color(float t) {
    const Vec3 c0(0.50f, 0.12f, 0.04f);   // cool: deep warm red
    const Vec3 c1(1.00f, 0.50f, 0.18f);   // orange
    const Vec3 c2(1.00f, 0.85f, 0.65f);   // warm tan-white
    const Vec3 c3(1.00f, 0.96f, 0.92f);   // hot: warm white (pinkish)
    if (t < 0.33f) { float u = t / 0.33f;          return c0 + (c1 - c0) * u; }
    if (t < 0.66f) { float u = (t - 0.33f) / 0.33f; return c1 + (c2 - c1) * u; }
    {              float u = (t - 0.66f) / 0.34f;   return c2 + (c3 - c2) * u; }
}

// Smallest signed angular difference a-b, wrapped to [-π, π] (for hot spots).
__host__ __device__ inline float ang_diff(float a, float b) {
    float d = a - b;
    while (d >  3.14159265f) d -= 6.28318531f;
    while (d < -3.14159265f) d += 6.28318531f;
    return d;
}

// Layered procedural disk emission. disk_r, disk_phi from the geodesic hit.
__host__ __device__ inline Vec3 shade_disk(float disk_r, float disk_phi) {
    // --- Shakura-Sunyaev temperature: T ∝ r^{-3/4}, T_norm = 1 at ISCO ---
    float T_norm = powf(R_DISK_MIN / disk_r, 0.75f);
    T_norm = fmaxf(0.0f, fminf(1.0f, T_norm));
    Vec3  base = temperature_color(T_norm);

    // --- fBm turbulence: two layers (broad + fine) summed by weight ------
    float n1 = fbm(disk_r * cfg::DISK_N1_FREQ_R, disk_phi * cfg::DISK_N1_FREQ_PHI,
                   cfg::DISK_FBM_OCTAVES);
    float n2 = fbm(disk_r * cfg::DISK_N2_FREQ_R, disk_phi * cfg::DISK_N2_FREQ_PHI,
                   cfg::DISK_FBM_OCTAVES);
    float n  = cfg::DISK_N1_WEIGHT * n1 + cfg::DISK_N2_WEIGHT * n2;  // ~[0,1]
    float turb = cfg::DISK_TURB_FLOOR + cfg::DISK_TURB_GAIN * n;     // dark lanes

    // --- Concentric Keplerian banding ------------------------------------
    float band = 1.0f + cfg::DISK_BAND_DEPTH * sinf(disk_r * cfg::DISK_BAND_FREQ);

    // --- Hot spots: gaussian bumps at fixed (r, phi) ---------------------
    float spot = 1.0f;
    for (int i = 0; i < cfg::DISK_NUM_SPOTS; ++i) {
        float dr   = (disk_r - cfg::DISK_SPOTS[i].r)        / cfg::DISK_SPOT_SIGMA_R;
        float dphi = ang_diff(disk_phi, cfg::DISK_SPOTS[i].phi) / cfg::DISK_SPOT_SIGMA_PHI;
        spot += cfg::DISK_SPOTS[i].amp * expf(-0.5f * (dr * dr + dphi * dphi));
    }

    // --- Radial brightness falloff (hotter/brighter toward ISCO) ---------
    float lum = 0.05f + 0.95f * T_norm * T_norm;

    // Combine and lift into HDR; bloom + tone map handle the range.
    return base * (lum * turb * band * spot * cfg::DISK_HDR_SCALE);
}

// === Main shade entry point ==============================================
__host__ __device__ inline Vec3 shade(const TraceResult& result) {
    switch (result.outcome) {
        case RayOutcome::CAPTURED: return Vec3(0.0f, 0.0f, 0.0f);
        case RayOutcome::DISK:     return shade_disk(result.disk_r, result.disk_phi);
        case RayOutcome::ESCAPED:
        default:                   return sample_skybox(result.direction);
    }
}
