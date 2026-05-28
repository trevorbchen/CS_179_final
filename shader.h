#pragma once
#include "vec3.h"
#include "geodesic.h"   // RayOutcome, TraceResult, R_DISK_MIN, R_DISK_MAX
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

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

// --- Accretion disk shading ----------------------------------------------
// Temperature profile: T ∝ r^{-3/4} (thin Keplerian disk)
// disk_r in geometric units (M=1).
__host__ __device__ inline Vec3 shade_disk(float disk_r) {
    // Normalize: T_norm = 1 at ISCO, falls off outward
    float T_norm = powf(R_DISK_MIN / disk_r, 0.75f);
    T_norm = fmaxf(0.0f, fminf(1.0f, T_norm));

    float r, g, b;
    if (T_norm > 0.8f) {          // very hot: blue-white
        float t = (T_norm - 0.8f) / 0.2f;
        r = 1.0f;
        g = 0.75f + 0.25f*t;
        b = 0.5f  + 0.5f*t;
    } else if (T_norm > 0.45f) {  // warm: yellow-orange
        float t = (T_norm - 0.45f) / 0.35f;
        r = 1.0f;
        g = 0.2f + 0.55f*t;
        b = 0.0f + 0.5f*t;
    } else {                      // cool: dim red
        float t = T_norm / 0.45f;
        r = t;
        g = 0.0f;
        b = 0.0f;
    }

    float lum = 0.05f + 0.95f * T_norm * T_norm;
    return Vec3(r * lum, g * lum, b * lum);
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
