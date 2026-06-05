// =============================================================================
// MODULE 1 (Kevin): CUDA gravity kernel. Integrates null geodesics in
// Schwarzschild spacetime.
//
// Pass 1 of the two-pass pipeline. One CUDA thread per pixel (16x16 blocks):
//   1. obtain the primary ray via generate_camera_ray() — a Module 2 (Trevor)
//      device function, called here per the TA-approved cross-module pattern;
//   2. integrate the null geodesic backward with adaptive-step RK4, reusing the
//      exact physics building blocks from geodesic.h (geodesic_rhs / rk4_step_2d
//      / orbital_*), so results match the CPU trace_geodesic bit-for-bit modulo
//      floating-point ordering;
//   3. write a TerminalState (hit type + geometry) for Module 2 to shade.
//
// The kernel does NOT shade — that is Module 2's job. This file owns only the
// gravitational lensing simulation.
// =============================================================================
#include "cuda_renderer.h"
#include "cuda_helpers.h"
#include "geodesic.h"   // RS, R_HORIZON, R_INF, geodesic_rhs, rk4_step_2d, orbital_*
#include "camera.h"     // generate_camera_ray (Module 2, called from here)

// ----------------------------------------------------------------------------
// Device port of trace_geodesic() from geodesic.h.
//
// Line-for-line faithful to the CPU integrator: same orbital-plane reduction,
// same adaptive step h = step*(r/5r_s)^2 floored at 0.005, same termination
// order (disk crossing -> horizon -> escape), same linear interpolation to the
// equatorial crossing. The ONLY changes vs the CPU version are:
//   * disk_r_min / disk_r_max are runtime arguments (GUI-tunable) instead of
//     the compile-time R_DISK_MIN/MAX constants — defaults preserve parity;
//   * it records disk_phi and the 3-D hit position into TerminalState.
//
// Register pressure note: RK4 keeps the (r, vr, phi) state plus four sets of
// derivatives live across each step; this is the heaviest part of the kernel.
// float is used throughout (sufficient here — the conservation tests hold to
// ~1e-6). double would help only for extreme near-horizon grazing rays; the
// adaptive step + R_HORIZON cutoff keep float well-conditioned in practice.
// ----------------------------------------------------------------------------
__device__ TerminalState trace_geodesic_gpu(
    Vec3 pos, Vec3 dir, GeodesicParams params,
    float disk_r_min, float disk_r_max)
{
    TerminalState st;
    st.hit_type    = HIT_ESCAPED;
    st.hit_pos     = float3{0.0f, 0.0f, 0.0f};
    st.hit_dir     = float3{0.0f, 0.0f, 0.0f};
    st.disk_r      = 0.0f;
    st.disk_phi    = 0.0f;
    st.steps_taken = 0;

    // --- set up the orbital plane (spherical symmetry) ---
    float r0 = pos.norm();
    Vec3  e1 = pos * (1.0f / r0);          // radial unit vector

    float vr0        = dir.dot(e1);
    Vec3  d_perp     = dir - e1 * vr0;     // tangential component
    float d_perp_mag = d_perp.norm();
    float L          = r0 * d_perp_mag;    // conserved angular momentum

    Vec3 e2 = (d_perp_mag > 1e-8f) ? d_perp * (1.0f / d_perp_mag) : Vec3(0, 0, 0);

    // --- integration state ---
    float r   = r0;
    float vr  = vr0;
    float phi = 0.0f;

    Vec3 prev_pos3d = pos;   // z-sign tracking for disk crossing

    for (int i = 0; i < params.max_steps; ++i) {
        // Adaptive step: shrink near the BH (curvature is highest at r~3).
        float scale = r / (5.0f * RS);
        float h = (scale < 1.0f) ? params.step_size * scale * scale : params.step_size;
        if (h < 0.005f) h = 0.005f;

        rk4_step_2d(r, vr, phi, L, h);

        Vec3 curr_pos3d = orbital_pos3d(r, phi, e1, e2);

        // --- disk crossing: equatorial plane z = 0 ---
        if (prev_pos3d.z * curr_pos3d.z < 0.0f) {
            float t  = prev_pos3d.z / (prev_pos3d.z - curr_pos3d.z);
            Vec3  cx = prev_pos3d + (curr_pos3d - prev_pos3d) * t;
            float disk_r = sqrtf(cx.x * cx.x + cx.y * cx.y);
            if (disk_r >= disk_r_min && disk_r <= disk_r_max) {
                st.hit_type    = HIT_DISK;
                st.disk_r      = disk_r;
                st.disk_phi    = atan2f(cx.y, cx.x);
                st.hit_pos     = vec3_to_float3(cx);
                st.hit_dir     = vec3_to_float3(
                                   orbital_dir3d(vr, r, phi, L, e1, e2).normalized());
                st.steps_taken = i + 1;
                return st;
            }
        }

        // --- horizon (captured) ---
        if (r <= R_HORIZON) {
            st.hit_type    = HIT_CAPTURED;
            st.hit_pos     = vec3_to_float3(curr_pos3d);
            st.hit_dir     = vec3_to_float3(e1 * vr);   // radial inside the horizon
            st.steps_taken = i + 1;
            return st;
        }

        // --- escaped ---
        if (r >= R_INF) {
            st.hit_type    = HIT_ESCAPED;
            st.hit_pos     = vec3_to_float3(curr_pos3d);
            st.hit_dir     = vec3_to_float3(
                              orbital_dir3d(vr, r, phi, L, e1, e2).normalized());
            st.steps_taken = i + 1;
            return st;
        }

        prev_pos3d = curr_pos3d;
    }

    // Exhausted steps -> treat as escaped (matches CPU fallback).
    st.hit_type    = HIT_ESCAPED;
    st.hit_dir     = vec3_to_float3(orbital_dir3d(vr, r, phi, L, e1, e2).normalized());
    st.steps_taken = params.max_steps;
    return st;
}

// ----------------------------------------------------------------------------
// Pass-1 entry kernel: one thread per pixel, 16x16 blocks, row-major output.
// ----------------------------------------------------------------------------
__global__ void gravity_kernel(Camera cam, GeodesicParams gp,
                               float disk_r_min, float disk_r_max,
                               TerminalState* __restrict__ out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // MODULE 2 (Trevor) device function, invoked from Module 1's kernel.
    Ray ray = generate_camera_ray(cam, x, y);

    out[y * W + x] = trace_geodesic_gpu(ray.origin, ray.direction,
                                        gp, disk_r_min, disk_r_max);
}

// ----------------------------------------------------------------------------
// Host launcher.
// ----------------------------------------------------------------------------
void launch_gravity_kernel(const Camera& cam, const GeodesicParams& gp,
                           float disk_r_min, float disk_r_max,
                           TerminalState* d_states, int W, int H,
                           cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(div_up(W, block.x), div_up(H, block.y));
    gravity_kernel<<<grid, block, 0, stream>>>(cam, gp, disk_r_min, disk_r_max,
                                                d_states, W, H);
    CUDA_CHECK_KERNEL();
}
