#pragma once
#include "vec3.h"
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// -----------------------------------------------------------------------
// Kevin's module: Schwarzschild null-geodesic integrator
//
// Units: G = c = M = 1  →  r_s = 2
//
// Key insight: every photon path lies in an orbital plane (spherical
// symmetry).  We integrate a 3-component ODE in that plane:
//
//   y = (r, v_r, phi)
//   dr/dλ   = v_r
//   dv_r/dλ = L²(r − 3) / r⁴        (from Schwarzschild geodesic eq.)
//   dphi/dλ = L / r²                 (L = r·v_t, conserved)
//
// Derivation: differentiate the null condition
//   (v_r)² = E² − (1 − 2/r) L²/r²
// wrt λ; E drops out, leaving the pure spatial trajectory.
// -----------------------------------------------------------------------

static constexpr float RS         = 2.0f;          // Schwarzschild radius
static constexpr float R_HORIZON  = RS * 0.51f;    // absorbing boundary
static constexpr float R_INF      = 500.0f;        // "escaped"
static constexpr float R_DISK_MIN = 3.0f * RS;     // ISCO = 6M
static constexpr float R_DISK_MAX = 12.0f * RS;    // outer disk edge

enum class RayOutcome : int {
    ESCAPED  = 0,
    CAPTURED = 1,
    DISK     = 2,
};

struct TraceResult {
    RayOutcome outcome;
    Vec3  direction;  // final 3-D direction (skybox lookup for ESCAPED)
    float disk_r;     // cylindrical radius at disk crossing
    int   steps;
};

struct GeodesicParams {
    float step_size = 0.4f;
    int   max_steps = 5000;
};

// ---- ODE right-hand side ------------------------------------------------

__host__ __device__ inline void geodesic_rhs(
    float r, float vr, float /*phi*/, float L,
    float& dr_out, float& dvr_out, float& dphi_out)
{
    float r2   = r * r;
    float r4   = r2 * r2;
    dr_out   = vr;
    dvr_out  = L * L * (r - 3.0f) / r4;
    dphi_out = L / r2;
}

// ---- Single RK4 step in the orbital plane --------------------------------

__host__ __device__ inline void rk4_step_2d(
    float& r, float& vr, float& phi, float L, float h)
{
    float dr1, dvr1, dphi1;
    geodesic_rhs(r, vr, phi, L, dr1, dvr1, dphi1);

    float r2   = r   + 0.5f*h*dr1,
          vr2  = vr  + 0.5f*h*dvr1,
          phi2 = phi + 0.5f*h*dphi1;
    float dr2, dvr2, dphi2;
    geodesic_rhs(r2, vr2, phi2, L, dr2, dvr2, dphi2);

    float r3   = r   + 0.5f*h*dr2,
          vr3  = vr  + 0.5f*h*dvr2,
          phi3 = phi + 0.5f*h*dphi2;
    float dr3, dvr3, dphi3;
    geodesic_rhs(r3, vr3, phi3, L, dr3, dvr3, dphi3);

    float r4v  = r   + h*dr3,
          vr4  = vr  + h*dvr3,
          phi4 = phi + h*dphi3;
    float dr4, dvr4, dphi4;
    geodesic_rhs(r4v, vr4, phi4, L, dr4, dvr4, dphi4);

    r   += (h / 6.0f) * (dr1   + 2.0f*dr2   + 2.0f*dr3   + dr4);
    vr  += (h / 6.0f) * (dvr1  + 2.0f*dvr2  + 2.0f*dvr3  + dvr4);
    phi += (h / 6.0f) * (dphi1 + 2.0f*dphi2 + 2.0f*dphi3 + dphi4);
}

// ---- Orbital-plane → 3-D helpers ----------------------------------------

__host__ __device__ inline Vec3 orbital_pos3d(
    float r, float phi, Vec3 e1, Vec3 e2)
{
    return (e1 * cosf(phi) + e2 * sinf(phi)) * r;
}

__host__ __device__ inline Vec3 orbital_dir3d(
    float vr, float r, float phi, float L, Vec3 e1, Vec3 e2)
{
    Vec3 r_hat  =  e1 * cosf(phi) + e2 * sinf(phi);
    Vec3 ph_hat = -e1 * sinf(phi) + e2 * cosf(phi);
    return r_hat * vr + ph_hat * (L / r);
}

// ---- Kevin's core function ----------------------------------------------
//
// Trace a null geodesic backward from pos with initial direction dir.
// pos : camera position in world space (BH at origin)
// dir : ray direction (need not be unit; sets initial speed scale)
//
__host__ __device__ inline TraceResult trace_geodesic(
    Vec3 pos, Vec3 dir, GeodesicParams params = {})
{
    TraceResult result{};

    // --- set up orbital plane ---
    float r0 = pos.norm();
    Vec3  e1 = pos * (1.0f / r0);          // radial unit vector

    float vr0         = dir.dot(e1);
    Vec3  d_perp      = dir - e1 * vr0;   // tangential component
    float d_perp_mag  = d_perp.norm();
    float L           = r0 * d_perp_mag;  // conserved angular momentum

    // Tangential basis vector (zero for radial rays; handled gracefully below)
    Vec3 e2 = (d_perp_mag > 1e-8f) ? d_perp * (1.0f / d_perp_mag) : Vec3(0,0,0);

    // --- integration state ---
    float r   = r0;
    float vr  = vr0;
    float phi = 0.0f;

    Vec3 prev_pos3d = pos;   // for z-sign crossing detection

    for (int i = 0; i < params.max_steps; ++i) {
        // Adaptive step: shrink near the BH
        float scale = r / (5.0f * RS);
        float h = (scale < 1.0f) ? params.step_size * scale * scale : params.step_size;
        if (h < 0.005f) h = 0.005f;

        rk4_step_2d(r, vr, phi, L, h);

        Vec3 curr_pos3d = orbital_pos3d(r, phi, e1, e2);

        // --- disk crossing: equatorial plane z = 0 ---
        if (prev_pos3d.z * curr_pos3d.z < 0.0f) {
            // Linear interpolation to find the precise crossing position
            float t = prev_pos3d.z / (prev_pos3d.z - curr_pos3d.z);
            Vec3  cx = prev_pos3d + (curr_pos3d - prev_pos3d) * t;
            float disk_r = sqrtf(cx.x*cx.x + cx.y*cx.y);
            if (disk_r >= R_DISK_MIN && disk_r <= R_DISK_MAX) {
                result.outcome   = RayOutcome::DISK;
                result.disk_r    = disk_r;
                result.direction = orbital_dir3d(vr, r, phi, L, e1, e2).normalized();
                result.steps     = i + 1;
                return result;
            }
        }

        // --- horizon ---
        if (r <= R_HORIZON) {
            result.outcome   = RayOutcome::CAPTURED;
            result.direction = e1 * vr;   // inside horizon, direction is radial
            result.steps     = i + 1;
            return result;
        }

        // --- escaped ---
        if (r >= R_INF) {
            result.outcome   = RayOutcome::ESCAPED;
            result.direction = orbital_dir3d(vr, r, phi, L, e1, e2).normalized();
            result.steps     = i + 1;
            return result;
        }

        prev_pos3d = curr_pos3d;
    }

    // Exhausted steps → treat as escaped
    result.outcome   = RayOutcome::ESCAPED;
    result.direction = orbital_dir3d(vr, r, phi, L, e1, e2).normalized();
    return result;
}
