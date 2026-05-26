#pragma once
#include "vec3.h"
#include "geodesic.h"   // RayOutcome, TraceResult, R_DISK_MIN, R_DISK_MAX
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// -----------------------------------------------------------------------
// Trevor's module: pixel shading
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

// Map an escaped ray direction to a star-field RGB sample.
// The direction need not be unit length (will be normalized internally).
__host__ __device__ inline Vec3 sample_skybox(Vec3 dir) {
    // Map to spherical grid cell
    float len = dir.norm();
    float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));  // [0, π]
    float phi   = atan2f(dz, dx) + 3.14159265f;           // [0, 2π]

    const int GRID = 256;
    int iu = (int)(phi   / (2.0f * 3.14159265f) * (float)GRID) % GRID;
    int iv = (int)(theta /        3.14159265f    * (float)GRID) % GRID;
    unsigned int cell = (unsigned int)(iu * GRID + iv);

    // Background: very dim blue-black
    Vec3 color(0.005f, 0.005f, 0.015f);

    // ~0.3% of cells are stars
    if (hash01(cell) > 0.997f) {
        float brightness = 0.4f + 0.6f * hash01(cell + 1000u);
        float temp       = hash01(cell + 2000u);   // 0=red, 1=blue-white
        Vec3 star;
        if (temp > 0.6f) {
            // Blue-white hot star
            float t = (temp - 0.6f) / 0.4f;
            star = Vec3(0.7f + 0.3f*t, 0.85f + 0.15f*t, 1.0f) * brightness;
        } else {
            // Yellow-orange cool star
            float t = temp / 0.6f;
            star = Vec3(1.0f, 0.5f + 0.45f*t, 0.1f + 0.5f*t) * brightness;
        }
        color = star;
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
