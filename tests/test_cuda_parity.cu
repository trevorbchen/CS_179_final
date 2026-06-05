// =============================================================================
// test_cuda_parity.cu — the correctness keystone for the GPU port.
//
// Renders the SAME scene on the CPU reference path (render() in renderer.h) and
// through the full GPU pipeline (gravity -> renderer -> tonemap) in parity
// configuration (Reinhard tone map, no bloom, no disk fBm, procedural starfield
// — i.e. RenderParams defaults), then compares the tone-mapped images pixel by
// pixel. Small floating-point differences are expected (host vs device FMA
// ordering); the test passes if the MEAN absolute per-channel difference is
// below 0.01. Prints mean / 95th-percentile / max for the writeup.
//
// Build/links: cuda/{gravity,renderer,postprocess}.cu (see CMakeLists.txt).
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_util.h"          // TEST(), run_all_tests()
#include "camera.h"
#include "geodesic.h"
#include "renderer.h"           // CPU render(), Image
#include "cuda_renderer.h"
#include "cuda_helpers.h"

// Map the GPU pipeline through to a tone-mapped LDR image, in parity config.
static void gpu_render_parity(const Camera& cam, std::vector<uint32_t>& ldr) {
    int W = cam.width, H = cam.height;
    size_t n = (size_t)W * H;
    TerminalState* d_states; float4* d_hdr; uint32_t* d_ldr;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(TerminalState)));
    CUDA_CHECK(cudaMalloc(&d_hdr,    n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_ldr,    n * sizeof(uint32_t)));

    RenderParams rp;                       // defaults == CPU parity config
    GeodesicParams gp = rp.geo;
    launch_gravity_kernel(cam, gp, rp.disk_r_min, rp.disk_r_max, d_states, W, H);
    launch_renderer_kernel(d_states, d_hdr, W, H, rp, /*tex*/0, 0, 0);
    launch_tonemap(d_hdr, d_ldr, W, H, ToneMap::Reinhard, rp.exposure);
    CUDA_CHECK(cudaDeviceSynchronize());

    ldr.resize(n);
    CUDA_CHECK(cudaMemcpy(ldr.data(), d_ldr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_states); cudaFree(d_hdr); cudaFree(d_ldr);
}

TEST(CudaParity) {
    const int W = 200, H = 150;
    Camera cam = Camera::look_at(Vec3(0, -35, 1), Vec3(0, 0, 0), Vec3(0, 0, 1),
                                 55.0f, W, H);
    GeodesicParams gp;

    Image cpu = render(cam, gp);           // tone-mapped (Reinhard + gamma 2.2)
    std::vector<uint32_t> gpu;
    gpu_render_parity(cam, gpu);

    std::vector<float> diffs;
    diffs.reserve((size_t)W * H * 3);
    double sum = 0.0, maxd = 0.0;
    for (int i = 0; i < W * H; ++i) {
        uint32_t p = gpu[i];
        float gr = (p & 0xFF) / 255.f, gg = ((p >> 8) & 0xFF) / 255.f,
              gb = ((p >> 16) & 0xFF) / 255.f;
        float cr = std::fmin(std::fmax(cpu.pixels[i].x, 0.f), 1.f);
        float cg = std::fmin(std::fmax(cpu.pixels[i].y, 0.f), 1.f);
        float cb = std::fmin(std::fmax(cpu.pixels[i].z, 0.f), 1.f);
        float d0 = std::fabs(cr - gr), d1 = std::fabs(cg - gg), d2 = std::fabs(cb - gb);
        sum += d0 + d1 + d2;
        maxd = std::fmax(maxd, std::fmax(d0, std::fmax(d1, d2)));
        diffs.push_back(d0); diffs.push_back(d1); diffs.push_back(d2);
    }
    std::sort(diffs.begin(), diffs.end());
    double mean = sum / diffs.size();
    double p95  = diffs[(size_t)(0.95 * diffs.size())];

    bool ok = (mean < 0.01);
    printf("CudaParity %dx%d: mean=%.5f p95=%.5f max=%.5f (tol mean<0.01) [%s]\n",
           W, H, mean, p95, maxd, PF(ok));
    return ok;
}

int main() {
    printf("=== CUDA parity test (CPU reference vs full GPU pipeline) ===\n");
    return run_all_tests();
}
