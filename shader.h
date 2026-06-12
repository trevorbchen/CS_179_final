#pragma once
#include "vec3.h"
#include "geodesic.h"   // RayOutcome, TraceResult, R_DISK_MIN, R_DISK_MAX
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// =======================================================================
// ACCRETION-DISK FILAMENT TUNING CONSTANTS  (Module 2 / Trevor)
//
// These shape the "Interstellar" disk look: thin bright filaments of gas
// stretched along the orbit, wound into spirals, streaming around the hole
// with inner material faster than outer (differential/Keplerian rotation).
// Grouped here so they are fast to tweak. DISK_SPIN_SPEED and SPIRAL_WIND
// are the DEFAULTS for the GUI-tunable RenderParams fields of the same
// name (sliders override them at runtime); the rest are compile-time.
//
// The look comes from three ingredients:
//   (a) ANISOTROPY  — BAND_FREQ_R >> BAND_FREQ_PHI: high radial / low
//                     azimuthal frequency stretches features into streaks
//                     that follow the orbit instead of round blobs.
//   (b) SHARP LINES — ridge() + pow(.,FILAMENT_SHARPNESS) concentrates
//                     brightness into thin filaments with dark gaps; the
//                     emission stays HDR (DISK_HDR_SCALE) so bloom turns
//                     the ridges into glowing streaks.
//   (c) SHEAR       — sampling at phi_eff = phi - omega(r)*(WIND + speed*RATE*time)
//                     winds straight streaks into spirals at a frozen
//                     instant (WIND) and makes them stream as `time`
//                     advances (speed). omega(r) = r^-1.5 is the Keplerian
//                     profile, so inner material is faster than outer.
//
// NOTE on decoupling: the speed scale multiplies ONLY the time term and the
// winding scale ONLY the static term, so the "disk spin speed" and "spiral
// tightness" sliders are independent (a single omega*(WIND+time) term would
// make spin also change tightness). DISK_ANIM_RATE converts the small
// Keplerian omega (~0.07 at the ISCO) into a visible streaming rate, so the
// default speed = 1.0 already animates clearly (~10 s inner orbit).
// =======================================================================
static constexpr float DISK_SPIN_SPEED    = 1.0f;   // animation speed scale (default for slider)
static constexpr float DISK_ANIM_RATE     = 8.0f;   // base streaming rate so speed=1 is clearly visible
static constexpr float SPIRAL_WIND        = 8.0f;   // static winding amount -> spiral tightness at t=0 (default for slider)
static constexpr float BAND_FREQ_R        = 9.0f;   // radial frequency; HIGH -> many thin concentric bands
static constexpr float BAND_FREQ_PHI      = 2.5f;   // azimuthal frequency; LOW vs BAND_FREQ_R -> elongated streaks
static constexpr float ANGULAR_NOISE_SCALE = 1.0f;  // overall ring-radius scale for the seamless azimuthal sampling
static constexpr float FILAMENT_SHARPNESS = 3.0f;   // exponent concentrating brightness into thin lines
static constexpr float DETAIL_FREQ_R      = 24.0f;  // fine-wisp radial frequency
static constexpr float DETAIL_FREQ_PHI    = 6.0f;   // fine-wisp azimuthal frequency
static constexpr float DETAIL_WEIGHT      = 0.3f;   // how much fine detail to mix in
static constexpr float DISK_HDR_SCALE     = 18.0f;  // disk emission brightness; HDR (>bloom threshold) so bloom still catches filaments
static constexpr float DISK_EDGE_FADE     = 0.15f;  // fraction of outer radial range over which density fades to 0
// Blackbody COLOR endpoints (Kelvin). The disk hue is the Planckian color of a
// temperature that runs smoothly from DISK_T_OUTER_K at the outer edge to
// DISK_T_INNER_K at the ISCO, parameterized by the Shakura-Sunyaev T ~ r^-3/4
// profile -> a continuous deep-red -> orange -> gold -> white sweep (no hard
// ring). Brightness is kept separate (radial falloff x density x HDR scale).
static constexpr float DISK_T_INNER_K     = 7500.0f; // blackbody temp at inner edge (hot: white / blue-white)
static constexpr float DISK_T_OUTER_K     = 2400.0f; // blackbody temp at outer edge (cool: deep orange-red)

// -----------------------------------------------------------------------
// pixel shading
//
// Three cases:
//   CAPTURED → black (inside event horizon)
//   DISK     → blackbody-like gradient based on disk radius
//   ESCAPED  → procedural starfield skybox
// -----------------------------------------------------------------------

// --- Procedural starfield -----------------------------------------------
// Deterministic integer hash; no stdlib random needed (CUDA-safe).
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

// Stellar spectral class colors: temp in [0,1] maps red→orange→yellow→white→blue
__host__ __device__ inline Vec3 star_color(float temp, float brightness) {
    Vec3 c;
    if (temp > 0.75f) {                         // O/B: blue-white
        float t = (temp - 0.75f) / 0.25f;
        c = Vec3(0.7f + 0.3f*t, 0.85f + 0.15f*t, 1.0f);
    } else if (temp > 0.50f) {                  // A/F: white-yellow
        float t = (temp - 0.50f) / 0.25f;
        c = Vec3(1.0f, 1.0f, 0.6f + 0.4f*t);
    } else if (temp > 0.25f) {                  // G/K: yellow-orange (sun-like)
        float t = (temp - 0.25f) / 0.25f;
        c = Vec3(1.0f, 0.55f + 0.45f*t, 0.05f + 0.55f*t);
    } else {                                    // M: red giant
        float t = temp / 0.25f;
        c = Vec3(0.7f + 0.3f*t, 0.05f + 0.15f*t, 0.0f);
    }
    return c * brightness;
}

// Map an escaped ray direction to a star-field RGB sample.
// The direction need not be unit length (will be normalized internally).
__host__ __device__ inline Vec3 sample_skybox(Vec3 dir) {
    float len = dir.norm();
    float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));  // [0, π]
    float phi   = atan2f(dz, dx) + 3.14159265f;           // [0, 2π]

    // Milky Way: Gaussian band of scattered light along galactic equator
    float lat = theta - 1.5707963f;
    float mw  = expf(-lat * lat * 6.0f);
    Vec3 color(0.004f + mw * 0.010f, 0.004f + mw * 0.008f, 0.012f + mw * 0.022f);

    // Layer 1: fine dense field (512-grid, ~0.5% density) — faint distant stars
    {
        const int G = 512;
        int iu = (int)(phi / (2.0f * 3.14159265f) * G) % G;
        int iv = (int)(theta / 3.14159265f * G) % G;
        unsigned int cell = (unsigned int)(iu * G + iv);
        if (hash01(cell) > 0.995f) {
            float b = 0.12f + 0.28f * hash01(cell + 1000u);
            color = color + star_color(hash01(cell + 2000u), b);
        }
    }

    // Layer 2: medium field (256-grid, ~0.8% density) — mid-distance stars
    {
        const int G = 256;
        int iu = (int)(phi / (2.0f * 3.14159265f) * G) % G;
        int iv = (int)(theta / 3.14159265f * G) % G;
        unsigned int cell = (unsigned int)(iu * G + iv) + 300000u;
        if (hash01(cell) > 0.992f) {
            float b = 0.30f + 0.45f * hash01(cell + 1000u);
            color = color + star_color(hash01(cell + 2000u), b);
        }
    }

    // Layer 3: sparse bright stars (128-grid, ~1% density) — nearby bright stars
    {
        const int G = 128;
        int iu = (int)(phi / (2.0f * 3.14159265f) * G) % G;
        int iv = (int)(theta / 3.14159265f * G) % G;
        unsigned int cell = (unsigned int)(iu * G + iv) + 700000u;
        if (hash01(cell) > 0.990f) {
            float b = 0.65f + 0.35f * hash01(cell + 1000u);
            color = color + star_color(hash01(cell + 2000u), b);
        }
    }

    return color;
}

// --- Blackbody (Planckian) color approximation ---------------------------
// Tanner Helland's piecewise fit: maps a temperature in Kelvin (~1000-40000 K)
// to an RGB HUE, normalized so the brightest channel is ~1 (brightness is
// applied separately by the caller). Smooth and monotone across the range, so
// it eases red -> orange -> gold -> white -> blue-white with no hard steps.
// __host__ __device__ + deterministic, so the CPU reference and the GPU path
// produce identical colors (parity-safe). Domain note: the disk only feeds
// temperatures in [DISK_T_OUTER_K, DISK_T_INNER_K], where every logf/powf
// argument below is safely positive.
__host__ __device__ inline Vec3 blackbody_rgb(float kelvin) {
    float t = fmaxf(1000.0f, fminf(40000.0f, kelvin)) / 100.0f;
    float r, g, b;

    if (t <= 66.0f) r = 255.0f;
    else            r = 329.698727446f * powf(t - 60.0f, -0.1332047592f);

    if (t <= 66.0f) g = 99.4708025861f  * logf(t) - 161.1195681661f;
    else            g = 288.1221695283f * powf(t - 60.0f, -0.0755148492f);

    if      (t >= 66.0f) b = 255.0f;
    else if (t <= 19.0f) b = 0.0f;
    else                 b = 138.5177312231f * logf(t - 10.0f) - 305.0447927307f;

    r = fmaxf(0.0f, fminf(255.0f, r));
    g = fmaxf(0.0f, fminf(255.0f, g));
    b = fmaxf(0.0f, fminf(255.0f, b));
    return Vec3(r, g, b) * (1.0f / 255.0f);
}

// --- Accretion disk shading ----------------------------------------------
// Temperature profile: T ∝ r^{-3/4} (thin Keplerian / Shakura-Sunyaev disk).
// HUE and BRIGHTNESS are kept separate:
//   * HUE       = blackbody color of the local temperature, parameterized by a
//                 temperature-normalized t_norm so the gradient is spread
//                 evenly across the disk (the color change isn't crammed into a
//                 thin inner annulus -> no abrupt yellow->red ring).
//   * BRIGHTNESS = the original inner-brighter radial falloff (unchanged), so
//                 filaments still glow correctly through bloom.
// disk_r in geometric units (M=1).
__host__ __device__ inline Vec3 shade_disk(float disk_r) {
    // Inner-brighter radial intensity falloff — UNCHANGED.
    float T_norm = powf(R_DISK_MIN / disk_r, 0.75f);
    T_norm = fmaxf(0.0f, fminf(1.0f, T_norm));
    float lum = 0.05f + 0.95f * T_norm * T_norm;

    // Temperature-parameterized color: normalize the local Shakura-Sunyaev
    // temperature against the disk's inner/outer temperatures so t_norm runs
    // 0 (outer edge) -> 1 (ISCO) with color variation spread where T varies.
    float T_loc = powf(disk_r,     -0.75f);
    float T_in  = powf(R_DISK_MIN,  -0.75f);
    float T_out = powf(R_DISK_MAX,  -0.75f);
    float t_norm = (T_loc - T_out) / (T_in - T_out);
    t_norm = fmaxf(0.0f, fminf(1.0f, t_norm));

    // Map t_norm to a physical temperature, then to a smooth blackbody hue.
    float kelvin = DISK_T_OUTER_K + t_norm * (DISK_T_INNER_K - DISK_T_OUTER_K);
    return blackbody_rgb(kelvin) * lum;
}

// --- Filament noise helpers (shared host/device; single source of truth) --
// 3D value-noise fBm. The disk samples it on a RING — (cos phi, sin phi) — so
// the pattern is inherently periodic in azimuth (period 2*pi) and therefore
// has NO atan2 branch-cut seam; the disk radius is fed on the third axis so
// neighbouring radii get decorrelated filament patterns. Deterministic
// (reuses the integer hash above) -> CUDA-safe and frame-stable. Used ONLY by
// the disk-filament path, which the parity config leaves disabled, so it never
// affects the CPU/GPU parity render.
__host__ __device__ inline float disk_hash3(int a, int b, int c) {
    return hash01((unsigned int)(a * 73856093) ^ (unsigned int)(b * 19349663)
                  ^ (unsigned int)(c * 83492791));
}

// Smooth trilinear value noise, output in [0,1].
__host__ __device__ inline float disk_value_noise3(float x, float y, float z) {
    int   xi = (int)floorf(x), yi = (int)floorf(y), zi = (int)floorf(z);
    float xf = x - xi, yf = y - yi, zf = z - zi;
    float u = xf * xf * (3.0f - 2.0f * xf);    // smoothstep fades
    float v = yf * yf * (3.0f - 2.0f * yf);
    float w = zf * zf * (3.0f - 2.0f * zf);
    float c000 = disk_hash3(xi,     yi,     zi    );
    float c100 = disk_hash3(xi + 1, yi,     zi    );
    float c010 = disk_hash3(xi,     yi + 1, zi    );
    float c110 = disk_hash3(xi + 1, yi + 1, zi    );
    float c001 = disk_hash3(xi,     yi,     zi + 1);
    float c101 = disk_hash3(xi + 1, yi,     zi + 1);
    float c011 = disk_hash3(xi,     yi + 1, zi + 1);
    float c111 = disk_hash3(xi + 1, yi + 1, zi + 1);
    float x00 = c000 + (c100 - c000) * u;
    float x10 = c010 + (c110 - c010) * u;
    float x01 = c001 + (c101 - c001) * u;
    float x11 = c011 + (c111 - c011) * u;
    float y0  = x00 + (x10 - x00) * v;
    float y1  = x01 + (x11 - x01) * v;
    return y0 + (y1 - y0) * w;
}

// 4-octave fractal Brownian motion, normalized to ~[0,1].
__host__ __device__ inline float disk_fbm3(float x, float y, float z) {
    float sum = 0.0f, amp = 0.5f, freq = 1.0f, norm = 0.0f;
    for (int o = 0; o < 4; ++o) {
        sum  += amp * disk_value_noise3(x * freq, y * freq, z * freq);
        norm += amp;
        freq *= 2.0f;
        amp  *= 0.5f;
    }
    return sum / norm;   // [0,1]
}

// smoothstep(edge0, edge1, x): 0 below edge0, 1 above edge1, Hermite between.
__host__ __device__ inline float disk_smoothstep(float e0, float e1, float x) {
    float t = (x - e0) / (e1 - e0);
    t = fmaxf(0.0f, fminf(1.0f, t));
    return t * t * (3.0f - 2.0f * t);
}

// --- Accretion-disk FILAMENTS (the "Interstellar" look) -------------------
// Returns an HDR color: the existing radial blackbody profile modulated by a
// sheared, ridge-sharpened filament density. See the tuning block at the top.
//   r, phi        : disk-crossing cylindrical radius and azimuth
//   time_seconds  : animation clock (0 = frozen; CPU/parity callers pass 0)
//   spin_speed    : DISK_SPIN_SPEED (GUI-tunable)
//   spiral_wind   : SPIRAL_WIND     (GUI-tunable)
//   r_min, r_max  : active disk radial bounds (for the soft outer edge)
__host__ __device__ inline Vec3 shade_disk_filaments(
    float r, float phi, float time_seconds,
    float spin_speed, float spiral_wind, float r_min, float r_max)
{
    // Keplerian shear: inner material faster (omega ~ r^-3/2). Sampling at a
    // radius-dependent angular offset winds straight streaks into spirals at a
    // frozen instant (spiral_wind) and makes them stream as `time` advances
    // (spin_speed). The two scales are independent; DISK_ANIM_RATE makes the
    // small Keplerian omega visible so spin_speed = 1 already streams clearly.
    float omega   = powf(r, -1.5f);
    float anim    = spin_speed * DISK_ANIM_RATE * time_seconds;
    float phi_eff = phi - omega * (spiral_wind + anim);

    // Seamless azimuth: sample the noise on a RING built from cos/sin of the
    // sheared angle, NOT from the raw atan2 value. cos/sin are exactly 2*pi-
    // periodic in phi, so the sample matches across the atan2 branch cut -> no
    // seam. The azimuthal frequency comes from the RING RADIUS (FREQ_PHI *
    // ANGULAR_NOISE_SCALE); putting the frequency inside the angle instead
    // (cos(phi*FREQ_PHI)) would re-break periodicity for non-integer FREQ_PHI
    // and bring the seam back. Disk radius is the third noise axis so adjacent
    // radii get decorrelated patterns. The shear/animation/spirals are intact
    // because they all live in phi_eff, which still drives cos/sin.
    float cphi = cosf(phi_eff), sphi = sinf(phi_eff);

    // Anisotropic bands: high radial freq (3rd axis), low azimuthal freq (ring)
    // -> elongated streaks following the orbit.
    float band_ring = BAND_FREQ_PHI * ANGULAR_NOISE_SCALE;
    float bands = disk_fbm3(cphi * band_ring, sphi * band_ring, r * BAND_FREQ_R); // [0,1]

    // Ridge + sharpen into thin bright filaments with dark gaps between them.
    float ridge = 1.0f - fabsf(2.0f * bands - 1.0f);
    ridge = powf(fmaxf(ridge, 0.0f), FILAMENT_SHARPNESS);

    // Fine wisps: same sheared angle (co-move with the filaments), own freqs.
    float detail_ring = DETAIL_FREQ_PHI * ANGULAR_NOISE_SCALE;
    float detail = disk_fbm3(cphi * detail_ring, sphi * detail_ring, r * DETAIL_FREQ_R);

    float density = ridge * (1.0f - DETAIL_WEIGHT + DETAIL_WEIGHT * detail);

    // Soft outer edge so the disk doesn't look like a flat plate.
    float r_frac = (r - r_min) / fmaxf(r_max - r_min, 1e-6f);
    float edge   = 1.0f - disk_smoothstep(1.0f - DISK_EDGE_FADE, 1.0f, r_frac);
    density *= edge;

    // Existing radial temperature/color profile, scaled by filament density, HDR.
    return shade_disk(r) * (density * DISK_HDR_SCALE);
}

// --- Main shade entry point ----------------------------------------------
__host__ __device__ inline Vec3 shade(const TraceResult& result) {
    switch (result.outcome) {
        case RayOutcome::CAPTURED: return Vec3(0.0f, 0.0f, 0.0f);
        case RayOutcome::DISK:     return shade_disk(result.disk_r);
        case RayOutcome::ESCAPED:
        default:                   return sample_skybox(result.direction);
    }
}
