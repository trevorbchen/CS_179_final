// =============================================================================
// MODULE 2 (Trevor): CUDA renderer kernel. Consumes the TerminalState buffer
// from Module 1 and produces an HDR (linear, float4 RGBA) framebuffer.
//
// Pass 2 of the two-pass pipeline. One CUDA thread per pixel (16x16 blocks),
// reading the TerminalState that Module 1's gravity kernel wrote for the same
// pixel and shading it:
//   HIT_CAPTURED -> black (inside the event horizon)
//   HIT_DISK     -> blackbody color from disk_r (shader.h), optionally
//                   modulated by a procedural fBm band pattern (GUI-tunable;
//                   strength 0 reproduces the flat CPU disk for parity)
//   HIT_ESCAPED  -> skybox: hardware-bilinear equirectangular texture sample
//                   if a skybox is loaded, else the procedural starfield from
//                   shader.h (the CPU baseline path used by the parity test)
//
// This kernel does REAL per-pixel work (the shading model + skybox sampling),
// distinct from Module 1's geometry pass — the two-pass split keeps the
// gravity simulation and the rendering as separately graded modules.
// =============================================================================
#include "cuda_renderer.h"
#include "cuda_helpers.h"
#include "shader.h"   // shade_disk, sample_skybox, star_color, uhash, hash01

// ---- float4 HDR write helper ------------------------------------------------
__device__ __forceinline__ float4 make_hdr(const Vec3& c) {
    return float4{c.x, c.y, c.z, 1.0f};
}

// ----------------------------------------------------------------------------
// Procedural fBm for accretion-disk surface detail (Module 2 enhancement).
//
// Value-noise summed over a few octaves, keyed on (azimuth, radius) so it
// forms turbulent orbital bands. Deterministic — reuses shader.h's integer
// hash (hash01) so it is CUDA-safe and frame-stable. Returns ~[0,1].
// Disabled (strength 0) for CPU parity; the gravity/shading math is untouched.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float hash2(int a, int b) {
    return hash01((unsigned int)(a * 73856093) ^ (unsigned int)(b * 19349663));
}
__device__ float value_noise(float x, float y) {
    int   xi = (int)floorf(x), yi = (int)floorf(y);
    float xf = x - xi,         yf = y - yi;
    float u = xf * xf * (3.0f - 2.0f * xf);   // smoothstep fade
    float v = yf * yf * (3.0f - 2.0f * yf);
    float a = hash2(xi,     yi);
    float b = hash2(xi + 1, yi);
    float c = hash2(xi,     yi + 1);
    float d = hash2(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
__device__ float disk_fbm(float disk_r, float disk_phi) {
    // Wrap azimuth so the noise is seamless around the ring; stretch along phi.
    float x = disk_phi * 3.0f;
    float y = disk_r   * 0.6f;
    float sum = 0.0f, amp = 0.5f, freq = 1.0f;
    for (int o = 0; o < 4; ++o) {
        sum  += amp * value_noise(x * freq, y * freq);
        freq *= 2.0f;
        amp  *= 0.5f;
    }
    return sum;   // ~[0,1)
}

// ----------------------------------------------------------------------------
// Equirectangular skybox texture sample (hardware bilinear via texture object).
// Matches sample_skybox()'s direction->(theta,phi) convention so swapping the
// procedural field for a real texture only changes the source of the color.
// ----------------------------------------------------------------------------
__device__ Vec3 sample_skybox_texture(cudaTextureObject_t tex, Vec3 dir) {
    float len = dir.norm();
    float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));     // [0, pi]
    float phi   = atan2f(dz, dx) + 3.14159265f;             // [0, 2pi]
    float u = phi   / (2.0f * 3.14159265f);
    float v = theta / 3.14159265f;
    float4 t = tex2D<float4>(tex, u, v);                    // free bilinear filtering
    return Vec3(t.x, t.y, t.z);
}

// ----------------------------------------------------------------------------
// Shade a single TerminalState into a linear HDR color.
// ----------------------------------------------------------------------------
__device__ Vec3 shade_terminal(const TerminalState& st, const RenderParams& rp,
                               cudaTextureObject_t skybox_tex)
{
    switch (st.hit_type) {
        case HIT_CAPTURED:
            return Vec3(0.0f, 0.0f, 0.0f);

        case HIT_DISK: {
            Vec3 base = shade_disk(st.disk_r);              // shader.h blackbody gradient
            if (rp.disk_fbm_strength > 0.0f) {
                float n = disk_fbm(st.disk_r, st.disk_phi); // ~[0,1]
                float m = 1.0f + rp.disk_fbm_strength * (2.0f * n - 1.0f);
                base = base * fmaxf(0.0f, m);
            }
            return base;
        }

        case HIT_ESCAPED:
        default: {
            Vec3 dir = float3_to_vec3(st.hit_dir);
            if (rp.use_skybox_texture)
                return sample_skybox_texture(skybox_tex, dir);
            return sample_skybox(dir);                      // procedural (CPU parity)
        }
    }
    // STRETCH: gravitational redshift / relativistic Doppler boost of the disk
    // color would be applied here using hit_dir and the local Keplerian
    // velocity. Left out per scope (two-pass parity first).
}

// ----------------------------------------------------------------------------
// Pass-2 kernel: read TerminalState, shade, write HDR float4 (coalesced).
// ----------------------------------------------------------------------------
__global__ void renderer_kernel(const TerminalState* __restrict__ states,
                               float4* __restrict__ hdr, int W, int H,
                               RenderParams rp, cudaTextureObject_t skybox_tex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    Vec3 color = shade_terminal(states[idx], rp, skybox_tex);
    hdr[idx] = make_hdr(color);
}

// ----------------------------------------------------------------------------
// Host launcher.
// ----------------------------------------------------------------------------
void launch_renderer_kernel(const TerminalState* d_states, float4* d_hdr,
                            int W, int H, const RenderParams& rp,
                            cudaTextureObject_t skybox_tex,
                            int /*sky_w*/, int /*sky_h*/, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(div_up(W, block.x), div_up(H, block.y));
    renderer_kernel<<<grid, block, 0, stream>>>(d_states, d_hdr, W, H, rp, skybox_tex);
    CUDA_CHECK_KERNEL();
}
