// =============================================================================
// main_gpu.cpp — entry point for renderer_gpu (the real-time CUDA renderer).
//
// Modes:
//   (default)          : start the browser viewer (MJPEG + JSON control) and
//                        run the live render loop until Ctrl-C.
//   --benchmark [N]     : CPU-vs-GPU timing sweep across resolutions -> CSV.
//   --benchmark-fast [N]: same but only the two smallest resolutions (quick).
//   --once <file.jpg>   : render a single frame to a JPEG and exit (headless).
//   --glfw              : use the GLFW/OpenGL window (only if built with it).
//
// Orchestration is CudaRenderer (cuda/cuda_renderer.h); the server is decoupled
// via callbacks (viewer/http_server.h). This file is Module 2 (Trevor) glue.
// =============================================================================
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "camera.h"
#include "geodesic.h"
#include "renderer.h"            // CPU render() for the benchmark baseline
#include "cuda/cuda_renderer.h"
#include "viewer/image_io.h"
#include "viewer/http_server.h"

#ifdef HAVE_GLFW_VIEWER
// Defined in viewer/glfw_viewer.cpp; declared here so main can dispatch to it.
struct ViewerSharedState;
int run_glfw_viewer(int base_w, int base_h);   // see viewer/glfw_viewer.cpp
#endif

#ifndef PROJECT_SOURCE_DIR_STR
#define PROJECT_SOURCE_DIR_STR "."
#endif

// ---------------------------------------------------------------------------
// Shared state between the render loop (producer) and the HTTP server (consumer)
// ---------------------------------------------------------------------------
struct Shared {
    std::mutex m;
    // Orbit camera (around the black hole at `target`).
    double yaw = 0.0, pitch = 0.03, dist = 35.0, fov = 55.0;
    Vec3   target{0, 0, 0};
    // Render parameters (interactive defaults: ACES + bloom + a little disk fBm).
    RenderParams rp;
    // Resolution: internal = base * res_scale.
    int    base_w = 800, base_h = 600;
    double res_scale = 1.0;
    int    quality = 85;
    bool   redshift = false;     // STRETCH placeholder (no effect yet)
    bool   paused   = false;     // freeze the disk-filament animation clock
    // Outputs (produced by the render loop).
    std::vector<uint8_t> jpeg;
    FrameTimings t;
    double fps = 0.0;
    int    out_w = 800, out_h = 600;
    std::atomic<bool> running{true};
};

static Shared* g_shared = nullptr;   // for the signal handler

static void on_sigint(int) { if (g_shared) g_shared->running = false; }

// ---- minimal JSON scalar extraction (flat objects only) -------------------
static bool json_num(const std::string& s, const char* key, double& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p) return false;
    out = v;
    return true;
}

// ---- build a Camera from the orbit state ----------------------------------
static Camera make_camera(double yaw, double pitch, double dist,
                          double fov, Vec3 target, int W, int H) {
    double cp = std::cos(pitch), sp = std::sin(pitch);
    Vec3 dir((float)(std::sin(yaw) * cp),
             (float)(-std::cos(yaw) * cp),
             (float)sp);
    Vec3 pos = target + dir * (float)dist;
    return Camera::look_at(pos, target, Vec3(0, 0, 1), (float)fov, W, H);
}

static std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s((size_t)(n < 0 ? 0 : n), '\0');
    if (n > 0 && std::fread(&s[0], 1, (size_t)n, f) != (size_t)n) s.clear();
    std::fclose(f);
    return s;
}

static std::string load_index_html() {
    // Prefer the file next to the source tree, then CWD, then a minimal fallback.
    for (const std::string& p : {std::string(PROJECT_SOURCE_DIR_STR) + "/viewer/index.html",
                                 std::string("viewer/index.html")}) {
        std::string s = read_file(p);
        if (!s.empty()) return s;
    }
    return "<!doctype html><meta charset=utf-8><title>BH</title>"
           "<body style='margin:0;background:#000'>"
           "<img src='/stream' style='width:100vw;height:100vh;object-fit:contain'>"
           "</body>";
}

// Try to load data/skybox.jpg into the renderer's texture; returns true if found.
static bool try_load_skybox(CudaRenderer& r, const std::string& path) {
    std::vector<uint8_t> rgb; int w = 0, h = 0;
    if (!load_jpeg_rgb(path.c_str(), rgb, w, h)) return false;
    r.set_skybox(rgb.data(), w, h);
    std::printf("[skybox] loaded %s (%dx%d) -> CUDA texture\n", path.c_str(), w, h);
    return true;
}

// ===========================================================================
// Benchmark mode
// ===========================================================================
static int run_benchmark(int N, bool quick) {
    struct R { int w, h; };
    std::vector<R> res = {{400,300},{800,600},{1280,720},{1920,1080},{3840,2160}};
    if (quick) res = {{400,300},{800,600}};

    FILE* csv = std::fopen("benchmark_results.csv", "w");
    if (csv) std::fprintf(csv, "resolution,cpu_ms,gpu_pass1_ms,gpu_pass2_ms,"
                               "gpu_postprocess_ms,gpu_total_ms,speedup\n");

    std::printf("\n=== Benchmark: CPU baseline vs GPU (%d GPU frames/res) ===\n", N);
    std::printf("%-11s %10s %9s %9s %9s %9s %9s\n",
                "resolution","cpu_ms","p1_ms","p2_ms","post_ms","gpu_ms","speedup");

    CudaRenderer gpu;
    for (const R& r : res) {
        Camera cam = Camera::look_at(Vec3(0,-35,1), Vec3(0,0,0), Vec3(0,0,1),
                                     55.0f, r.w, r.h);
        GeodesicParams gp;

        // ---- GPU: bloom OFF, Reinhard (a fair match to the CPU baseline path) ----
        RenderParams rp;                         // defaults = parity config
        for (int i = 0; i < 3; ++i) gpu.render_frame(cam, rp);   // warm up
        double p1 = 0, p2 = 0, pp = 0, tot = 0;
        for (int i = 0; i < N; ++i) {
            gpu.render_frame(cam, rp);
            const FrameTimings& t = gpu.timings();
            p1 += t.pass1_ms; p2 += t.pass2_ms; pp += t.postprocess_ms; tot += t.total_ms;
        }
        p1 /= N; p2 /= N; pp /= N; tot /= N;

        // ---- CPU baseline: render() once (a few times at small res) ----
        int cpu_frames = (r.w * r.h >= 1280 * 720) ? 1 : 3;
        auto c0 = std::chrono::high_resolution_clock::now();
        size_t sink = 0;
        for (int i = 0; i < cpu_frames; ++i) {
            Image img = render(cam, gp);
            sink += img.pixels.size();           // keep the result live
        }
        asm volatile("" : : "r"(sink) : "memory"); // prevent dead-code elimination
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu_ms =
            std::chrono::duration<double, std::milli>(c1 - c0).count() / cpu_frames;

        double speedup = cpu_ms / tot;
        char name[32]; std::snprintf(name, sizeof(name), "%dx%d", r.w, r.h);
        std::printf("%-11s %10.2f %9.3f %9.3f %9.3f %9.3f %8.1fx\n",
                    name, cpu_ms, p1, p2, pp, tot, speedup);
        if (csv) std::fprintf(csv, "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                              name, cpu_ms, p1, p2, pp, tot, speedup);
    }
    if (csv) { std::fclose(csv); std::printf("\nWrote benchmark_results.csv\n"); }
    return 0;
}

// ===========================================================================
// Single-frame headless render (for tests / quick checks)
// ===========================================================================
static int run_once(int W, int H, const std::string& out, const std::string& skybox) {
    CudaRenderer gpu;
    bool sky = try_load_skybox(gpu, skybox);
    RenderParams rp;
    rp.tonemap = ToneMap::ACES;
    rp.bloom_enabled = true; rp.bloom_intensity = 0.6f;
    rp.disk_fbm_strength = 1.0f;            // full thin-filament disk
    rp.use_skybox_texture = sky;
    Camera cam = make_camera(0.0, 0.03, 35.0, 55.0, Vec3(0,0,0), W, H);
    gpu.render_frame(cam, rp, 0.0f);        // static frame (t=0)
    const uint32_t* ldr = gpu.read_ldr();
    std::vector<uint8_t> jpeg = encode_jpeg_rgba(ldr, W, H, 92);
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::fwrite(jpeg.data(), 1, jpeg.size(), f);
    std::fclose(f);
    const FrameTimings& t = gpu.timings();
    std::printf("Wrote %s (%dx%d). Pass1 %.2fms  Pass2 %.2fms  Post %.2fms  Total %.2fms\n",
                out.c_str(), W, H, t.pass1_ms, t.pass2_ms, t.postprocess_ms, t.total_ms);
    return 0;
}

// ===========================================================================
// Browser viewer mode (default)
// ===========================================================================
static int run_server(Shared& sh, int port, const std::string& skybox) {
    g_shared = &sh;
    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);

    CudaRenderer gpu;
    bool sky = try_load_skybox(gpu, skybox);
    { std::lock_guard<std::mutex> lk(sh.m); sh.rp.use_skybox_texture = sky; }

    ViewerServer server(port, load_index_html());
    server.set_frame_provider([&sh]() {
        std::lock_guard<std::mutex> lk(sh.m);
        return sh.jpeg;
    });
    server.set_stats_provider([&sh]() {
        std::lock_guard<std::mutex> lk(sh.m);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"fps\":%.1f,\"pass1\":%.3f,\"pass2\":%.3f,\"post\":%.3f,"
            "\"total\":%.3f,\"w\":%d,\"h\":%d}",
            sh.fps, sh.t.pass1_ms, sh.t.pass2_ms, sh.t.postprocess_ms,
            sh.t.total_ms, sh.out_w, sh.out_h);
        return std::string(buf);
    });
    server.set_control_handler([&sh](const std::string& body) {
        std::lock_guard<std::mutex> lk(sh.m);
        double v;
        if (json_num(body, "reset", v) && v != 0) {
            sh.yaw = 0; sh.pitch = 0.03; sh.dist = 35; sh.fov = 55;
        }
        if (json_num(body, "yaw", v))            sh.yaw = v;
        if (json_num(body, "pitch", v))          sh.pitch = std::fmax(-1.5, std::fmin(1.5, v));
        if (json_num(body, "dist", v))           sh.dist = std::fmax(3.0, std::fmin(200.0, v));
        if (json_num(body, "fov", v))            sh.fov = std::fmax(10.0, std::fmin(120.0, v));
        if (json_num(body, "disk_r_min", v))     sh.rp.disk_r_min = (float)v;
        if (json_num(body, "disk_r_max", v))     sh.rp.disk_r_max = (float)v;
        if (json_num(body, "fbm", v))            sh.rp.disk_fbm_strength = (float)v;
        if (json_num(body, "disk_spin_speed", v))sh.rp.disk_spin_speed = (float)std::fmax(0.0, v);
        if (json_num(body, "spiral_wind", v))    sh.rp.spiral_wind = (float)std::fmax(0.0, v);
        if (json_num(body, "paused", v))         sh.paused = (v != 0);
        if (json_num(body, "step_size", v))      sh.rp.geo.step_size = (float)std::fmax(0.02, v);
        if (json_num(body, "exposure", v))       sh.rp.exposure = (float)v;
        if (json_num(body, "bloom", v))          sh.rp.bloom_enabled = (v != 0);
        if (json_num(body, "bloom_threshold", v))sh.rp.bloom_threshold = (float)v;
        if (json_num(body, "bloom_intensity", v))sh.rp.bloom_intensity = (float)v;
        if (json_num(body, "tonemap", v))        sh.rp.tonemap = (v != 0) ? ToneMap::ACES : ToneMap::Reinhard;
        if (json_num(body, "res_scale", v))      sh.res_scale = std::fmax(0.1, std::fmin(2.0, v));
        if (json_num(body, "skybox", v) && sh.rp.use_skybox_texture == false) { /* only if loaded */ }
        if (json_num(body, "use_skybox", v))     sh.rp.use_skybox_texture = (v != 0);
        if (json_num(body, "redshift", v))       sh.redshift = (v != 0);
    });
    server.start();

    std::printf("Open the forwarded port in your browser:  http://localhost:%d/\n", port);
    std::printf("(Ctrl-C to stop.)\n");

    // ---- render loop (main thread owns the CUDA context) ----
    const double target_ms = 1000.0 / 60.0;   // cap ~60 fps
    double fps_ema = 0.0;
    // Disk-filament animation clock: advanced by REAL wall-clock elapsed time per
    // frame, so the filaments stream at the same rate regardless of frame rate.
    // Frozen while paused.
    double anim_seconds = 0.0;
    auto   anim_prev    = std::chrono::high_resolution_clock::now();
    while (sh.running) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Advance the animation clock by real elapsed time (unless paused).
        double anim_dt = std::chrono::duration<double>(t0 - anim_prev).count();
        anim_prev = t0;

        double yaw, pitch, dist, fov, scale; Vec3 target; RenderParams rp; int bw, bh, q;
        bool paused;
        {
            std::lock_guard<std::mutex> lk(sh.m);
            yaw = sh.yaw; pitch = sh.pitch; dist = sh.dist; fov = sh.fov;
            target = sh.target; rp = sh.rp; scale = sh.res_scale;
            bw = sh.base_w; bh = sh.base_h; q = sh.quality; paused = sh.paused;
        }
        if (!paused) anim_seconds += anim_dt;

        int W = std::max(16, (int)std::lround(bw * scale));
        int H = std::max(16, (int)std::lround(bh * scale));
        Camera cam = make_camera(yaw, pitch, dist, fov, target, W, H);

        gpu.render_frame(cam, rp, (float)anim_seconds);
        const uint32_t* ldr = gpu.read_ldr();
        std::vector<uint8_t> jpeg = encode_jpeg_rgba(ldr, W, H, q);
        FrameTimings t = gpu.timings();

        auto t1 = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double inst_fps = frame_ms > 0 ? 1000.0 / frame_ms : 0.0;
        fps_ema = (fps_ema == 0.0) ? inst_fps : (0.9 * fps_ema + 0.1 * inst_fps);

        {
            std::lock_guard<std::mutex> lk(sh.m);
            sh.jpeg = std::move(jpeg);
            sh.t = t; sh.fps = fps_ema; sh.out_w = W; sh.out_h = H;
        }
        if (frame_ms < target_ms)
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(target_ms - frame_ms));
    }

    std::printf("\nShutting down...\n");
    server.stop();
    return 0;
}

// ===========================================================================
int main(int argc, char** argv) {
    int  base_w = 800, base_h = 600, port = 8080, bench_n = 100;
    bool do_bench = false, bench_fast = false, do_glfw = false, do_once = false;
    std::string once_out = "frame.jpg";
    std::string skybox = std::string(PROJECT_SOURCE_DIR_STR) + "/milkyway.jpg";

    // Positional WxH first (two integers), then flags.
    int pos = 1;
    if (argc >= 3 && argv[1][0] != '-' && argv[2][0] != '-') {
        base_w = std::atoi(argv[1]); base_h = std::atoi(argv[2]); pos = 3;
    }
    for (int i = pos; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--benchmark")       { do_bench = true; if (i+1 < argc && argv[i+1][0] != '-') bench_n = std::atoi(argv[++i]); }
        else if (a == "--benchmark-fast") { do_bench = true; bench_fast = true; if (i+1 < argc && argv[i+1][0] != '-') bench_n = std::atoi(argv[++i]); }
        else if (a == "--port")       { if (i+1 < argc) port = std::atoi(argv[++i]); }
        else if (a == "--glfw")       { do_glfw = true; }
        else if (a == "--once")       { do_once = true; if (i+1 < argc && argv[i+1][0] != '-') once_out = argv[++i]; }
        else if (a == "--skybox")     { if (i+1 < argc) skybox = argv[++i]; }
        else if (a == "--help" || a == "-h") {
            std::printf("usage: renderer_gpu [W H] [--benchmark[-fast] [N]] [--once file.jpg]\n"
                        "                    [--port P] [--skybox path.jpg] [--glfw]\n");
            return 0;
        }
    }

    int dev = 0; cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        std::printf("GPU: %s  (sm_%d%d, %.1f GB)\n", prop.name, prop.major, prop.minor,
                    prop.totalGlobalMem / 1e9);

    if (do_bench) return run_benchmark(bench_n, bench_fast);
    if (do_once)  return run_once(base_w, base_h, once_out, skybox);

    if (do_glfw) {
#ifdef HAVE_GLFW_VIEWER
        return run_glfw_viewer(base_w, base_h);
#else
        std::fprintf(stderr, "renderer_gpu was built without the GLFW viewer "
                             "(reconfigure with -DBUILD_GLFW_VIEWER=ON).\n");
        return 1;
#endif
    }

    Shared sh;
    sh.base_w = base_w; sh.base_h = base_h;
    sh.rp.tonemap = ToneMap::ACES;          // pretty interactive defaults
    sh.rp.bloom_enabled = true; sh.rp.bloom_intensity = 0.6f;
    sh.rp.disk_fbm_strength = 1.0f;         // full thin-filament accretion disk
    return run_server(sh, port, skybox);
}
