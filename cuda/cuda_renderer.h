#pragma once
// =============================================================================
// cuda_renderer.h — shared GPU types + host-side orchestration declarations.
//
// This header is the contract between the three CUDA modules and the host:
//   * TerminalState  : the per-pixel hand-off buffer Module 1 -> Module 2
//   * RenderParams   : runtime-tunable scene/post parameters (driven by the GUI)
//   * launch_*       : thin host launchers, each defined in its module's .cu
//   * CudaRenderer   : the host orchestration class (buffers + timing + passes)
//
// It is included by both nvcc (the .cu kernels) and the host C++ compiler
// (main_gpu.cpp, viewer). Everything here is host-callable; no <<<>>> syntax
// leaks into the header, so plain g++ translation units can use CudaRenderer.
// =============================================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "vec3.h"
#include "camera.h"
#include "geodesic.h"
#include "shader.h"         // disk filament tuning constants (DISK_SPIN_SPEED, ...)
#include "cuda_helpers.h"   // CUDA_CHECK, div_up

// ----------------------------------------------------------------------------
// Hit-type encoding for TerminalState.hit_type.
//
// NOTE: this deliberately differs from geodesic.h's RayOutcome enum
// (ESCAPED=0, CAPTURED=1, DISK=2). The TerminalState wire format follows the
// project spec (CAPTURED=0, ESCAPED=1, DISK=2); the gravity kernel maps from
// RayOutcome to these values so the renderer kernel never depends on the
// physics enum's numeric layout.
// ----------------------------------------------------------------------------
enum : uint8_t {
    HIT_CAPTURED = 0,
    HIT_ESCAPED  = 1,
    HIT_DISK     = 2,
};

// ----------------------------------------------------------------------------
// TerminalState — Module 1's per-pixel output, Module 2's per-pixel input.
//
// Layout / AoS rationale: this is Array-of-Structs (one struct per pixel)
// rather than Struct-of-Arrays. That is the right choice here because every
// field of a pixel's TerminalState is consumed together by exactly one
// renderer-kernel thread (the thread owning that pixel): the thread branches
// on hit_type, then reads disk_r/disk_phi OR hit_dir. AoS keeps all of a
// pixel's data on the same cache line, so the single thread touches one line
// instead of gathering from N separate SoA streams. The struct is padded to
// 48 bytes (12B header rounded by the two float3 + scalars); threads in a warp
// read consecutive structs, which the L2/L1 still services efficiently.
// ----------------------------------------------------------------------------
struct TerminalState {
    uint8_t hit_type;    // HIT_CAPTURED / HIT_ESCAPED / HIT_DISK
    float3  hit_pos;     // 3-D position at termination (disk crossing point for DISK)
    float3  hit_dir;     // final ray direction (skybox lookup for ESCAPED)
    float   disk_r;      // cylindrical radius at disk crossing (DISK only)
    float   disk_phi;    // azimuth at disk crossing, atan2(y,x) (DISK only)
    int     steps_taken; // RK4 steps integrated (for instrumentation / heatmaps)
};

// ----------------------------------------------------------------------------
// Tone-mapping operator selector (Module 2 postprocess).
// Reinhard reproduces the CPU baseline exactly (used by the parity test);
// ACES is the nicer-looking default for the interactive viewer.
// ----------------------------------------------------------------------------
enum class ToneMap : int { Reinhard = 0, ACES = 1 };

// ----------------------------------------------------------------------------
// RenderParams — everything the GUI can tune at runtime, shared by all passes.
// Defaults reproduce the CPU baseline (Reinhard, no bloom, no disk noise, disk
// radii = geodesic.h's R_DISK_MIN/MAX) so the parity test compares like-for-like.
// ----------------------------------------------------------------------------
struct RenderParams {
    GeodesicParams geo;                 // step_size, max_steps (Module 1)
    float   disk_r_min      = R_DISK_MIN;   // 6  — disk inner edge (gravity + shading)
    float   disk_r_max      = R_DISK_MAX;   // 24 — disk outer edge
    // Disk filaments (the "Interstellar" look). disk_fbm_strength is the master
    // intensity / blend: 0 = flat blackbody disk (exact CPU parity), 1 = full
    // filaments. spin_speed/spiral_wind back the GUI sliders (defaults = the
    // documented constants in shader.h). time_seconds is passed separately to
    // the renderer kernel (not stored here) so the parity render can force 0.
    float   disk_fbm_strength = 0.0f;           // 0 = flat disk (CPU parity); >0 -> filaments
    float   disk_spin_speed   = DISK_SPIN_SPEED; // overall Keplerian angular-speed scale
    float   spiral_wind       = SPIRAL_WIND;     // static winding -> spiral tightness at t=0
    // Detail/realism additions. ALL default to their parity-neutral setting;
    // the interactive viewer (main_gpu.cpp) switches the pretty ones on.
    float   disk_detail     = 0.65f;    // 0..1 turbulence detail (warp/braids/flicker);
                                        //   only sampled when disk_fbm_strength > 0
    float   disk_thickness  = 0.0f;     // 0 = planar disk; >0 = volumetric slab march
    float   doppler_strength = 0.0f;    // 0 = off (parity); 1 = full relativistic
                                        //   Doppler beaming + gravitational redshift
    int     supersample     = 1;        // SSAA factor 1..3: passes 1+2 render at
                                        //   (W*ss
                                        //   x H*ss), then box-downsample before post
    float   vignette        = 0.0f;     // 0 = off (parity); subtle corner falloff
    float   sky_brightness  = 1.0f;     // skybox TEXTURE exposure (GPU-only path)
    ToneMap tonemap         = ToneMap::Reinhard;
    float   exposure        = 1.0f;     // linear pre-tonemap multiplier
    bool    bloom_enabled   = false;    // off for parity; on for the pretty default
    float   bloom_threshold = 1.0f;     // HDR luminance above which pixels bloom
    float   bloom_intensity = 0.6f;     // composite weight of the blurred bloom
    bool    use_skybox_texture = false; // false = procedural starfield (CPU parity)
};

// Per-frame GPU timings (filled via cudaEvent timing in the orchestrator).
struct FrameTimings {
    float pass1_ms      = 0.0f;   // gravity kernel
    float pass2_ms      = 0.0f;   // renderer kernel
    float postprocess_ms = 0.0f;  // bright + blur + composite + tonemap
    float total_ms      = 0.0f;
};

// =============================================================================
// Host launchers — each is defined in the .cu file of its owning module.
// Kept as plain host functions (no kernel-launch syntax) so the orchestrator
// and any host TU can call them.
// =============================================================================

// --- MODULE 1 (Kevin): gravity / geodesic integration -----------------------
void launch_gravity_kernel(const Camera& cam, const GeodesicParams& gp,
                           float disk_r_min, float disk_r_max,
                           TerminalState* d_states, int W, int H,
                           cudaStream_t stream = 0);

// --- MODULE 2 (Trevor): shading ---------------------------------------------
// time_seconds animates the disk filaments (0 = frozen; the parity test passes 0).
void launch_renderer_kernel(const TerminalState* d_states, float4* d_hdr,
                            int W, int H, const RenderParams& rp,
                            float time_seconds,
                            cudaTextureObject_t skybox_tex,
                            int sky_w, int sky_h,
                            cudaStream_t stream = 0);

// --- MODULE 2 (Trevor): postprocess -----------------------------------------
// (defined in postprocess.cu; declarations added as the kernels are built)
// vignette defaults to 0 (off) so existing call sites — including the parity
// test — keep their exact behavior.
void launch_tonemap(const float4* d_hdr, uint32_t* d_ldr, int W, int H,
                    ToneMap tm, float exposure, float vignette = 0.0f,
                    cudaStream_t stream = 0);
void launch_bloom(float4* d_hdr, float4* d_bright, float4* d_tmp,
                  float4* d_accum, int W, int H,
                  float threshold, float intensity, cudaStream_t stream = 0);
// SSAA box-downsample: average ss x ss HDR texels of d_hi (W*ss x H*ss) into
// each pixel of d_lo (W x H). Used only when RenderParams::supersample > 1.
void launch_downsample(const float4* d_hi, float4* d_lo, int W, int H, int ss,
                       cudaStream_t stream = 0);

// =============================================================================
// CudaRenderer — host-side orchestration of the two-pass + postprocess pipeline.
//
// Owns every device buffer, the CUDA stream, and the cudaEvents used for
// per-pass timing. render_frame() issues Pass 1 -> Pass 2 -> (bloom) -> tonemap
// on one stream with events between passes, so the GUI gets real Pass1/Pass2/
// Postprocess millisecond numbers each frame. All methods are host-only and
// use the CUDA runtime API + the module launchers (no <<<>>> in this header),
// so this class compiles under plain g++ as well as nvcc.
// =============================================================================
class CudaRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer() { destroy(); }
    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    // (Re)allocate device buffers for a WxH output at supersample factor ss.
    // Passes 1+2 run at (W*ss x H*ss); postprocess + readback at WxH. No-op if
    // already this configuration.
    void resize(int W, int H, int ss = 1) {
        if (W == W_ && H == H_ && ss == ss_) return;
        ensure_stream();
        free_buffers();
        W_ = W; H_ = H; ss_ = ss;
        size_t n  = (size_t)W * H;            // output-resolution pixels
        size_t nh = n * ss * ss;              // internal (supersampled) pixels
        CUDA_CHECK(cudaMalloc(&d_states_, nh * sizeof(TerminalState)));
        CUDA_CHECK(cudaMalloc(&d_hdr_,    nh * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_hdr_lo_, n  * sizeof(float4)));   // downsample target
        CUDA_CHECK(cudaMalloc(&d_bright_, n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_tmp_,    n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_accum_,  n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_ldr_,    n * sizeof(uint32_t)));
        if (h_ldr_) CUDA_CHECK(cudaFreeHost(h_ldr_));
        CUDA_CHECK(cudaMallocHost(&h_ldr_, n * sizeof(uint32_t)));  // pinned for fast readback
    }

    // Render one frame. Camera carries the OUTPUT resolution (cam.width/height);
    // when rp.supersample > 1 passes 1+2 run on an ss-times-finer pixel grid
    // (same FOV — the scaled camera's pixel centers are exactly an ordered
    // ss x ss subpixel grid) and the HDR result is box-filtered down before
    // bloom/tonemap. time_seconds drives the disk animation (0 = frozen).
    void render_frame(const Camera& cam, const RenderParams& rp_in,
                      float time_seconds = 0.0f) {
        RenderParams rp = rp_in;
        int ss = rp.supersample < 1 ? 1 : (rp.supersample > 3 ? 3 : rp.supersample);
        resize(cam.width, cam.height, ss);
        if (!has_sky_) rp.use_skybox_texture = false;   // no texture -> procedural

        int Wi = W_ * ss, Hi = H_ * ss;                 // internal resolution
        Camera icam = cam;
        icam.width = Wi; icam.height = Hi;              // same FOV, finer grid

        CUDA_CHECK(cudaEventRecord(ev0_, stream_));
        // --- Pass 1: Module 1 (gravity) ---
        launch_gravity_kernel(icam, rp.geo, rp.disk_r_min, rp.disk_r_max,
                              d_states_, Wi, Hi, stream_);
        CUDA_CHECK(cudaEventRecord(ev1_, stream_));
        // --- Pass 2: Module 2 (shading) ---
        launch_renderer_kernel(d_states_, d_hdr_, Wi, Hi, rp, time_seconds,
                              sky_tex_, sky_w_, sky_h_, stream_);
        CUDA_CHECK(cudaEventRecord(ev2_, stream_));
        // --- Postprocess: Module 2 (downsample + bloom + tonemap), at WxH ---
        float4* hdr_out = d_hdr_;
        if (ss > 1) {
            launch_downsample(d_hdr_, d_hdr_lo_, W_, H_, ss, stream_);
            hdr_out = d_hdr_lo_;
        }
        if (rp.bloom_enabled)
            launch_bloom(hdr_out, d_bright_, d_tmp_, d_accum_, W_, H_,
                         rp.bloom_threshold, rp.bloom_intensity, stream_);
        launch_tonemap(hdr_out, d_ldr_, W_, H_, rp.tonemap, rp.exposure,
                       rp.vignette, stream_);
        CUDA_CHECK(cudaEventRecord(ev3_, stream_));

        CUDA_CHECK(cudaEventSynchronize(ev3_));
        CUDA_CHECK(cudaEventElapsedTime(&timings_.pass1_ms,       ev0_, ev1_));
        CUDA_CHECK(cudaEventElapsedTime(&timings_.pass2_ms,       ev1_, ev2_));
        CUDA_CHECK(cudaEventElapsedTime(&timings_.postprocess_ms, ev2_, ev3_));
        CUDA_CHECK(cudaEventElapsedTime(&timings_.total_ms,       ev0_, ev3_));
    }

    // Copy the tone-mapped LDR framebuffer to pinned host memory and return it.
    // Layout: row-major W*H uint32, 0xAABBGGRR (R in the low byte).
    const uint32_t* read_ldr() {
        CUDA_CHECK(cudaMemcpyAsync(h_ldr_, d_ldr_, (size_t)W_ * H_ * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return h_ldr_;
    }

    // Upload an equirectangular skybox (RGB8, row-major) as a bilinear-filtered
    // texture object. u (azimuth) wraps; v (polar) clamps. readMode normalized
    // -> tex2D<float4> returns [0,1] with free hardware filtering + texture cache.
    void set_skybox(const uint8_t* rgb, int sky_w, int sky_h) {
        clear_skybox();
        std::vector<uchar4> rgba((size_t)sky_w * sky_h);
        for (size_t i = 0; i < rgba.size(); ++i)
            rgba[i] = uchar4{rgb[i*3+0], rgb[i*3+1], rgb[i*3+2], 255};

        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&sky_arr_, &ch, sky_w, sky_h));
        CUDA_CHECK(cudaMemcpy2DToArray(sky_arr_, 0, 0, rgba.data(),
                       sky_w * sizeof(uchar4), sky_w * sizeof(uchar4), sky_h,
                       cudaMemcpyHostToDevice));

        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = sky_arr_;
        cudaTextureDesc tex{};
        tex.addressMode[0]   = cudaAddressModeWrap;   // phi wraps around
        tex.addressMode[1]   = cudaAddressModeClamp;  // theta clamps at poles
        tex.filterMode       = cudaFilterModeLinear;  // hardware bilinear
        tex.readMode         = cudaReadModeNormalizedFloat;
        tex.normalizedCoords = 1;
        CUDA_CHECK(cudaCreateTextureObject(&sky_tex_, &res, &tex, nullptr));
        sky_w_ = sky_w; sky_h_ = sky_h; has_sky_ = true;
    }

    bool has_skybox()             const { return has_sky_; }
    int  width()                  const { return W_; }
    int  height()                 const { return H_; }
    const FrameTimings& timings() const { return timings_; }

private:
    void ensure_stream() {
        if (stream_) return;
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreate(&ev0_));
        CUDA_CHECK(cudaEventCreate(&ev1_));
        CUDA_CHECK(cudaEventCreate(&ev2_));
        CUDA_CHECK(cudaEventCreate(&ev3_));
    }
    void free_buffers() {
        if (d_states_) cudaFree(d_states_); d_states_ = nullptr;
        if (d_hdr_)    cudaFree(d_hdr_);    d_hdr_    = nullptr;
        if (d_hdr_lo_) cudaFree(d_hdr_lo_); d_hdr_lo_ = nullptr;
        if (d_bright_) cudaFree(d_bright_); d_bright_ = nullptr;
        if (d_tmp_)    cudaFree(d_tmp_);    d_tmp_    = nullptr;
        if (d_accum_)  cudaFree(d_accum_);  d_accum_  = nullptr;
        if (d_ldr_)    cudaFree(d_ldr_);    d_ldr_    = nullptr;
    }
    void clear_skybox() {
        if (sky_tex_) { cudaDestroyTextureObject(sky_tex_); sky_tex_ = 0; }
        if (sky_arr_) { cudaFreeArray(sky_arr_); sky_arr_ = nullptr; }
        has_sky_ = false;
    }
    void destroy() {
        free_buffers();
        clear_skybox();
        if (h_ldr_) { cudaFreeHost(h_ldr_); h_ldr_ = nullptr; }
        if (stream_) {
            cudaEventDestroy(ev0_); cudaEventDestroy(ev1_);
            cudaEventDestroy(ev2_); cudaEventDestroy(ev3_);
            cudaStreamDestroy(stream_); stream_ = 0;
        }
    }

    int W_ = 0, H_ = 0, ss_ = 1;
    TerminalState* d_states_ = nullptr;
    float4 *d_hdr_ = nullptr, *d_hdr_lo_ = nullptr,
           *d_bright_ = nullptr, *d_tmp_ = nullptr, *d_accum_ = nullptr;
    uint32_t* d_ldr_ = nullptr;
    uint32_t* h_ldr_ = nullptr;             // pinned host readback buffer
    cudaStream_t stream_ = 0;
    cudaEvent_t  ev0_, ev1_, ev2_, ev3_;
    FrameTimings timings_;

    cudaArray_t        sky_arr_ = nullptr;
    cudaTextureObject_t sky_tex_ = 0;
    int  sky_w_ = 0, sky_h_ = 0;
    bool has_sky_ = false;
};
