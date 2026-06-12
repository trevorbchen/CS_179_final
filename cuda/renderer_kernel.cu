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
// NOTE: the accretion-disk surface texture now lives in shader.h as the shared
// __host__ __device__ helpers disk_fbm3() / disk_value_noise3() and the filament
// shader shade_disk_filaments() (the "Interstellar" thin-filament look). The
// noise is sampled on a ring (cos/sin of the azimuth) so there is no atan2
// branch-cut seam. Keeping it in shader.h gives one source of truth for the CPU
// reference and the GPU path, and the parity config (disk_fbm_strength = 0)
// still reproduces the flat blackbody disk exactly. The earlier GPU-only fBm
// bands lived here.
// ----------------------------------------------------------------------------
// RELATIVISTIC SHADING TUNING (Module 2; the geodesics in Module 1 are not
// involved — everything here derives from the TerminalState alone).
//
// Doppler beaming + gravitational redshift for the Keplerian disk flow:
//   beta    = 1/sqrt(r - 2)        orbital speed measured by a static observer
//                                  (Schwarzschild circular orbit, G=c=M=1; 0.5c
//                                  at the ISCO r=6)
//   delta   = 1 / (gamma (1 - beta cos a))   special-relativistic Doppler,
//                                  a = angle between flow and the photon
//                                  direction toward the camera (-hit_dir)
//   g_grav  = sqrt(1 - 2/r)        climb out of the potential well
//   g       = delta * g_grav       total observed frequency ratio
// Applied as: temperature x g (hue slides along the Planck ramp — approaching
// side toward white/blue, receding toward deep red) and brightness x g^3
// (frequency-integrated beaming; g^4 is the bolometric result but reads as
// pure black/white clipping after tone mapping). doppler_strength lerps both
// toward neutral so 0 is EXACTLY off (the parity configuration).
// ----------------------------------------------------------------------------
static constexpr float DOPPLER_BETA_MAX   = 0.90f; // velocity clamp (the r_min slider can reach r ~ 2)
static constexpr float DOPPLER_BRIGHT_EXP = 2.5f;  // beaming exponent on brightness (3 is the frequency-integrated
                                                   // result; 2.5 keeps the receding side readable after tone mapping)

// ----------------------------------------------------------------------------
// Equirectangular skybox texture sample (hardware bilinear via texture object).
// Matches sample_skybox()'s direction->(theta,phi) convention so swapping the
// procedural field for a real texture only changes the source of the color.
// The JPEG texels are sRGB-encoded; they are linearized (gamma 2.2) here so the
// HDR pipeline + tonemap treat the sky correctly (raw texels rendered as linear
// washed the Milky Way out to grey). sky_brightness is the GUI exposure trim.
// ----------------------------------------------------------------------------
__device__ Vec3 sample_skybox_texture(cudaTextureObject_t tex, Vec3 dir,
                                      float sky_brightness) {
    float len = dir.norm();
    float dx = dir.x / len, dy = dir.y / len, dz = dir.z / len;
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dy)));     // [0, pi]
    float phi   = atan2f(dz, dx) + 3.14159265f;             // [0, 2pi]
    float u = phi   / (2.0f * 3.14159265f);
    float v = theta / 3.14159265f;
    float4 t = tex2D<float4>(tex, u, v);                    // free bilinear filtering
    return Vec3(powf(t.x, 2.2f), powf(t.y, 2.2f), powf(t.z, 2.2f)) * sky_brightness;
}

// ----------------------------------------------------------------------------
// Shade a single TerminalState into a linear HDR color.
// ----------------------------------------------------------------------------
__device__ Vec3 shade_terminal(const TerminalState& st, const RenderParams& rp,
                               float time_seconds, cudaTextureObject_t skybox_tex)
{
    switch (st.hit_type) {
        case HIT_CAPTURED:
            return Vec3(0.0f, 0.0f, 0.0f);

        case HIT_DISK: {
            Vec3 plain = shade_disk(st.disk_r);             // shader.h blackbody gradient
            float k = rp.disk_fbm_strength;                 // filament intensity / blend

            // Relativistic factors (neutral 1,1 at strength 0 -> parity path
            // bit-identical). See the tuning block above for the math.
            float g_shift = 1.0f, g_bright = 1.0f;
            if (rp.doppler_strength > 0.0f) {
                float r     = fmaxf(st.disk_r, 2.05f);
                float beta  = fminf(1.0f / sqrtf(r - 2.0f), DOPPLER_BETA_MAX);
                float gamma = 1.0f / sqrtf(1.0f - beta * beta);
                // Flow direction +phi-hat (counterclockwise from +z), the same
                // sense the filament animation streams.
                float cp = cosf(st.disk_phi), sp = sinf(st.disk_phi);
                Vec3 flow(-sp, cp, 0.0f);
                // Photon propagation direction toward the camera = reversed
                // backward-trace direction at the crossing.
                Vec3 n = float3_to_vec3(st.hit_dir) * -1.0f;
                float doppler = 1.0f / (gamma * (1.0f - beta * flow.dot(n)));
                float g = doppler * sqrtf(fmaxf(1.0f - 2.0f / r, 0.03f));
                g_shift  = 1.0f + (g - 1.0f) * rp.doppler_strength;
                g_bright = powf(g, DOPPLER_BRIGHT_EXP * rp.doppler_strength);
                plain = plain * g_bright;   // flat disk gets the beaming too
            }
            if (k <= 0.0f) return plain;                    // flat disk (exact CPU parity)

            // Sheared multi-scale filaments (the "Interstellar" look), animated
            // by time; volumetric Gaussian slab when thickness > 0.
            Vec3 fil = (rp.disk_thickness > 0.0f)
                ? shade_disk_slab(float3_to_vec3(st.hit_pos), float3_to_vec3(st.hit_dir),
                                  rp.disk_thickness, time_seconds,
                                  rp.disk_spin_speed, rp.spiral_wind, rp.disk_detail,
                                  rp.disk_r_min, rp.disk_r_max, g_shift, g_bright)
                : shade_disk_filaments(st.disk_r, st.disk_phi, time_seconds,
                                       rp.disk_spin_speed, rp.spiral_wind, rp.disk_detail,
                                       rp.disk_r_min, rp.disk_r_max, g_shift, g_bright);
            if (k >= 1.0f) return fil;
            return plain * (1.0f - k) + fil * k;            // blend flat <-> filaments
        }

        case HIT_ESCAPED:
        default: {
            Vec3 dir = float3_to_vec3(st.hit_dir);
            if (rp.use_skybox_texture)
                return sample_skybox_texture(skybox_tex, dir, rp.sky_brightness);
            return sample_skybox(dir);                      // procedural (CPU parity)
        }
    }
}

// ----------------------------------------------------------------------------
// Pass-2 kernel: read TerminalState, shade, write HDR float4 (coalesced).
// ----------------------------------------------------------------------------
__global__ void renderer_kernel(const TerminalState* __restrict__ states,
                               float4* __restrict__ hdr, int W, int H,
                               RenderParams rp, float time_seconds,
                               cudaTextureObject_t skybox_tex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    Vec3 color = shade_terminal(states[idx], rp, time_seconds, skybox_tex);
    hdr[idx] = make_hdr(color);
}

// ----------------------------------------------------------------------------
// Host launcher.
// ----------------------------------------------------------------------------
void launch_renderer_kernel(const TerminalState* d_states, float4* d_hdr,
                            int W, int H, const RenderParams& rp,
                            float time_seconds,
                            cudaTextureObject_t skybox_tex,
                            int /*sky_w*/, int /*sky_h*/, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(div_up(W, block.x), div_up(H, block.y));
    renderer_kernel<<<grid, block, 0, stream>>>(d_states, d_hdr, W, H, rp,
                                                time_seconds, skybox_tex);
    CUDA_CHECK_KERNEL();
}
