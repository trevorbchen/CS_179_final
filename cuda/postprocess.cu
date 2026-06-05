// =============================================================================
// MODULE 2 (Trevor): CUDA post-processing kernels.
//
//   * Tone mapping (Reinhard / ACES) + gamma, HDR float4 -> LDR uint32 RGBA
//   * Bloom: bright-pass extraction, separable Gaussian blur at multiple
//     scales (the shared-memory-tiled showcase kernel), and additive composite
//
// Reinhard + gamma 2.2 reproduces the CPU baseline's tone map exactly, which
// is what the parity test relies on; ACES is the default for the live viewer.
// =============================================================================
#include "cuda_renderer.h"
#include "cuda_helpers.h"

// =============================================================================
// Tone mapping
// =============================================================================

// Narkowicz 2015 ACES filmic approximation (operates per channel, linear in).
__device__ __forceinline__ float aces_curve(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Map one linear HDR channel to a display-referred [0,1] value.
__device__ __forceinline__ float tonemap_channel(float x, ToneMap tm) {
    float y = (tm == ToneMap::ACES) ? aces_curve(x)
                                    : x / (1.0f + x);   // Reinhard (CPU baseline)
    y = powf(fmaxf(y, 0.0f), 1.0f / 2.2f);              // gamma encode (gamma 2.2)
    return fminf(fmaxf(y, 0.0f), 1.0f);
}

// Pack 3 [0,1] channels into 0xAABBGGRR (R in the low byte; A=255).
__device__ __forceinline__ uint32_t pack_rgba(float r, float g, float b) {
    uint32_t R = (uint32_t)(r * 255.0f + 0.5f);
    uint32_t G = (uint32_t)(g * 255.0f + 0.5f);
    uint32_t B = (uint32_t)(b * 255.0f + 0.5f);
    return R | (G << 8) | (B << 16) | (0xFFu << 24);
}

__global__ void tonemap_kernel(const float4* __restrict__ hdr,
                               uint32_t* __restrict__ ldr,
                               int W, int H, ToneMap tm, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float4 c = hdr[idx];
    float r = tonemap_channel(c.x * exposure, tm);
    float g = tonemap_channel(c.y * exposure, tm);
    float b = tonemap_channel(c.z * exposure, tm);
    ldr[idx] = pack_rgba(r, g, b);
}

void launch_tonemap(const float4* d_hdr, uint32_t* d_ldr, int W, int H,
                    ToneMap tm, float exposure, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(div_up(W, block.x), div_up(H, block.y));
    tonemap_kernel<<<grid, block, 0, stream>>>(d_hdr, d_ldr, W, H, tm, exposure);
    CUDA_CHECK_KERNEL();
}

// =============================================================================
// Bloom — bright-pass + separable Gaussian (shared-memory tiled) + composite.
//
// This is the GPU showcase optimization. A naive Gaussian blur re-reads every
// neighbor from global memory for every output pixel: O(N * (2R+1)) global
// loads per axis. The separable + tiled version instead loads each input pixel
// into per-block SHARED memory exactly once (plus a halo of R pixels on each
// side), then every thread in the block computes its output by reading the
// (2R+1) taps it needs entirely from shared memory. Global traffic drops to
// ~one load per pixel per axis; the (2R+1)-tap reduction hits shared memory at
// ~terabytes/s instead of global DRAM.
//
// The Gaussian weights live in __constant__ memory: every thread in a warp
// reads the SAME weight for a given tap, which the constant cache broadcasts
// in a single transaction — the ideal access pattern for constant memory.
// =============================================================================

#define MAX_BLUR_RADIUS 64                 // sigma=32 -> radius clamped to 2*sigma
__constant__ float c_blur_weights[MAX_BLUR_RADIUS + 1];  // [0..R], symmetric
__constant__ int   c_blur_radius;

__device__ __forceinline__ float luma(const float4& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// --- Bright pass: keep only the over-threshold energy, preserving hue. -------
__global__ void bright_pass_kernel(const float4* __restrict__ hdr,
                                   float4* __restrict__ bright,
                                   int W, int H, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    float4 c = hdr[idx];
    float  l = luma(c);
    float  over  = fmaxf(l - threshold, 0.0f);
    float  scale = (l > 1e-6f) ? over / l : 0.0f;   // soft, hue-preserving
    bright[idx] = float4{c.x * scale, c.y * scale, c.z * scale, 1.0f};
}

// --- Horizontal separable blur (shared-memory tiled). ------------------------
// Block (BX, BY) produces a BX*BY tile of outputs. Shared holds, per row,
// (BX + 2R) pixels: the tile plus an R-wide halo on each side. The load loop is
// strided so it works for any R (including R > BX). Edge addresses are clamped
// (replicate) so the border doesn't darken.
__global__ void blur_h_kernel(const float4* __restrict__ in,
                              float4* __restrict__ out, int W, int H)
{
    extern __shared__ float4 smem[];
    const int R     = c_blur_radius;
    const int tileW = blockDim.x + 2 * R;
    const int gx0   = blockIdx.x * blockDim.x;
    const int gy    = blockIdx.y * blockDim.y + threadIdx.y;
    float4* row = &smem[threadIdx.y * tileW];

    if (gy < H) {
        for (int i = threadIdx.x; i < tileW; i += blockDim.x) {
            int sx = min(max(gx0 - R + i, 0), W - 1);   // clamp-to-edge
            row[i] = in[gy * W + sx];
        }
    }
    __syncthreads();

    int gx = gx0 + threadIdx.x;
    if (gx < W && gy < H) {
        float4 acc = float4{0, 0, 0, 0};
        for (int k = -R; k <= R; ++k) {
            float  w = c_blur_weights[k < 0 ? -k : k];
            float4 s = row[R + threadIdx.x + k];
            acc.x += w * s.x; acc.y += w * s.y; acc.z += w * s.z;
        }
        out[gy * W + gx] = float4{acc.x, acc.y, acc.z, 1.0f};
    }
}

// --- Vertical separable blur (shared-memory tiled), with optional accumulate.
// Shared layout is column-major-by-row so consecutive threads (threadIdx.x)
// read consecutive global columns -> coalesced loads. `accumulate` lets the
// caller sum multiple blur scales into one buffer; `out_scale` weights this
// scale's contribution.
__global__ void blur_v_kernel(const float4* __restrict__ in,
                              float4* __restrict__ out, int W, int H,
                              int accumulate, float out_scale)
{
    extern __shared__ float4 smem[];
    const int R     = c_blur_radius;
    const int tileH = blockDim.y + 2 * R;
    const int gy0   = blockIdx.y * blockDim.y;
    const int gx    = blockIdx.x * blockDim.x + threadIdx.x;

    if (gx < W) {
        for (int i = threadIdx.y; i < tileH; i += blockDim.y) {
            int sy = min(max(gy0 - R + i, 0), H - 1);    // clamp-to-edge
            smem[i * blockDim.x + threadIdx.x] = in[sy * W + gx];
        }
    }
    __syncthreads();

    int gy = gy0 + threadIdx.y;
    if (gx < W && gy < H) {
        float4 acc = float4{0, 0, 0, 0};
        for (int k = -R; k <= R; ++k) {
            float  w = c_blur_weights[k < 0 ? -k : k];
            float4 s = smem[(R + threadIdx.y + k) * blockDim.x + threadIdx.x];
            acc.x += w * s.x; acc.y += w * s.y; acc.z += w * s.z;
        }
        int idx = gy * W + gx;
        float4 prev = accumulate ? out[idx] : float4{0, 0, 0, 0};
        out[idx] = float4{prev.x + out_scale * acc.x,
                          prev.y + out_scale * acc.y,
                          prev.z + out_scale * acc.z, 1.0f};
    }
}

// --- Composite: HDR += intensity * bloom. ------------------------------------
__global__ void composite_kernel(float4* __restrict__ hdr,
                                 const float4* __restrict__ bloom,
                                 int W, int H, float intensity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    float4 c = hdr[idx], b = bloom[idx];
    hdr[idx] = float4{c.x + intensity * b.x,
                      c.y + intensity * b.y,
                      c.z + intensity * b.z, c.w};
}

// --- Host: fill a normalized half-Gaussian kernel [0..R]. --------------------
static int gaussian_weights(float sigma, float* w /*size MAX+1*/) {
    int R = (int)ceilf(3.0f * sigma);
    if (R > MAX_BLUR_RADIUS) R = MAX_BLUR_RADIUS;
    double norm = 0.0;
    for (int k = 0; k <= R; ++k) {
        w[k] = (float)exp(-(double)(k * k) / (2.0 * sigma * sigma));
        norm += (k == 0) ? w[k] : 2.0 * w[k];     // symmetric: count +/-k
    }
    for (int k = 0; k <= R; ++k) w[k] = (float)(w[k] / norm);
    return R;
}

// ----------------------------------------------------------------------------
// Bloom orchestration: bright-pass -> {blur at sigma 4,12,32, accumulated} ->
// composite back into the HDR buffer. Buffers (bright/tmp/accum) are owned by
// the caller (CudaRenderer) and sized W*H float4.
// ----------------------------------------------------------------------------
void launch_bloom(float4* d_hdr, float4* d_bright, float4* d_tmp,
                  float4* d_accum, int W, int H,
                  float threshold, float intensity, cudaStream_t stream)
{
    dim3 fb(16, 16), fg(div_up(W, 16), div_up(H, 16));
    bright_pass_kernel<<<fg, fb, 0, stream>>>(d_hdr, d_bright, W, H, threshold);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemsetAsync(d_accum, 0, (size_t)W * H * sizeof(float4), stream));

    const float sigmas[] = {4.0f, 12.0f, 32.0f};
    const int   nscales  = 3;
    const float wscale   = 1.0f / nscales;   // average the scales

    // Horizontal pass: block 128x2. Vertical pass: block 16x8 (keeps the
    // (BY+2R)-row tile within 48 KB shared for R up to MAX_BLUR_RADIUS).
    dim3 hb(128, 2), vb(16, 8);

    for (int s = 0; s < nscales; ++s) {
        float w[MAX_BLUR_RADIUS + 1];
        int   R = gaussian_weights(sigmas[s], w);
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_blur_weights, w,
                       (R + 1) * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_blur_radius, &R, sizeof(int),
                       0, cudaMemcpyHostToDevice, stream));

        dim3 hg(div_up(W, hb.x), div_up(H, hb.y));
        size_t hsh = (size_t)(hb.x + 2 * R) * hb.y * sizeof(float4);
        blur_h_kernel<<<hg, hb, hsh, stream>>>(d_bright, d_tmp, W, H);
        CUDA_CHECK_KERNEL();

        dim3 vg(div_up(W, vb.x), div_up(H, vb.y));
        size_t vsh = (size_t)vb.x * (vb.y + 2 * R) * sizeof(float4);
        blur_v_kernel<<<vg, vb, vsh, stream>>>(d_tmp, d_accum, W, H,
                                               /*accumulate=*/1, wscale);
        CUDA_CHECK_KERNEL();
    }

    composite_kernel<<<fg, fb, 0, stream>>>(d_hdr, d_accum, W, H, intensity);
    CUDA_CHECK_KERNEL();
}
