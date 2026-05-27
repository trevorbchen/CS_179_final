#pragma once
#include "vec3.h"

// -----------------------------------------------------------------------
// Trevor's module: render/post-process tuning constants
//
// All "look" parameters live here so the disk texture, skybox, bloom and
// tone-mapping passes can be retuned without touching algorithm code.
// Everything is a plain constexpr (no <vector>, no host-only types) so the
// file is safe to #include from the __host__ __device__ shader path.
// -----------------------------------------------------------------------

// ===== Camera (z is the disk normal; disk lies in the z = 0 plane) ========
// Tilted slightly *below* the disk plane (z < 0) for the iconic Interstellar
// framing where the disk lenses over the top and bottom of the shadow.
// Distance is kept > R_DISK_MAX (24) so the whole disk stays in frame.
namespace cfg {
    constexpr Vec3  CAM_POS    = Vec3(0.0f, -30.0f, 0.0f);
    constexpr Vec3  CAM_TARGET = Vec3(0.0f,   0.0f,  0.0f);
    constexpr Vec3  CAM_UP     = Vec3(0.0f,   0.0f,  1.0f);   // disk normal
    constexpr float CAM_FOV    = 55.0f;                        // vertical FoV (deg)

    // ===== Tone mapping ===================================================
    enum class ToneMap { Reinhard = 0, ACES = 1 };
    constexpr ToneMap TONE_MAP = ToneMap::ACES;   // default: cinematic ACES
    constexpr float   GAMMA    = 2.2f;

    // ===== Accretion-disk procedural texture =============================
    // Final emission is multiplied by DISK_HDR_SCALE so the disk sits well
    // above 1.0 (HDR); bloom + tone mapping compress the dynamic range.
    constexpr float DISK_HDR_SCALE = 9.0f;

    // fBm settings (gain 0.5, lacunarity 2.0 are hard-coded in fbm()).
    constexpr int   DISK_FBM_OCTAVES = 6;

    // Two noise layers, summed with these weights, evaluated at
    // (r * FREQ_R, phi * FREQ_PHI).  Layer 1 = broad bands, layer 2 = fine.
    constexpr float DISK_N1_FREQ_R   = 8.0f;
    constexpr float DISK_N1_FREQ_PHI = 16.0f;
    constexpr float DISK_N1_WEIGHT   = 0.6f;
    constexpr float DISK_N2_FREQ_R   = 32.0f;
    constexpr float DISK_N2_FREQ_PHI = 70.0f;
    constexpr float DISK_N2_WEIGHT   = 0.4f;

    // Turbulence brightness mapping: emission *= FLOOR + GAIN * noise.
    // A low FLOOR + high GAIN carves dark filament lanes that stay visible
    // even where the disk would otherwise saturate to flat white.
    constexpr float DISK_TURB_FLOOR = 0.10f;
    constexpr float DISK_TURB_GAIN  = 2.10f;

    // Concentric Keplerian banding (subtle radial brightness ripple).
    constexpr float DISK_BAND_FREQ  = 3.0f;   // cycles per unit radius
    constexpr float DISK_BAND_DEPTH = 0.18f;  // 0 = none, 1 = strong

    // Brighter "hot spots" at fixed (r, phi). Gaussian falloff.
    struct HotSpot { float r, phi, amp; };
    constexpr float   DISK_SPOT_SIGMA_R   = 1.20f;  // geometric radius units
    constexpr float   DISK_SPOT_SIGMA_PHI = 0.30f;  // radians
    constexpr int     DISK_NUM_SPOTS      = 4;
    constexpr HotSpot DISK_SPOTS[DISK_NUM_SPOTS] = {
        // r,    phi (rad),  extra intensity multiplier
        {  8.0f,  0.8f,  2.5f },
        { 11.0f, -1.9f,  2.0f },
        { 15.0f,  2.6f,  2.2f },
        { 19.0f, -0.4f,  1.8f },
    };

    // ===== Skybox =========================================================
    // Equirectangular image sampled for ESCAPED rays. If the file is missing
    // the renderer falls back to the procedural starfield.
    constexpr const char* SKYBOX_PATH      = "data/skybox.jpg";
    constexpr float        SKYBOX_INTENSITY = 0.7f;  // LDR->linear scale

    // ===== Bloom ==========================================================
    // Threshold for what "glows", three blur scales combined by weight, and
    // the overall strength added back to the HDR image before tone mapping.
    constexpr float BLOOM_THRESHOLD = 1.1f;
    constexpr float BLOOM_SIGMA_1   = 4.0f;
    constexpr float BLOOM_SIGMA_2   = 12.0f;
    constexpr float BLOOM_SIGMA_3   = 32.0f;
    constexpr float BLOOM_WEIGHT_1  = 0.5f;
    constexpr float BLOOM_WEIGHT_2  = 0.3f;
    constexpr float BLOOM_WEIGHT_3  = 0.2f;
    constexpr float BLOOM_STRENGTH  = 0.8f;   // final bloom contribution
}
