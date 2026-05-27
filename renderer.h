#pragma once
#include <vector>
#include <cstdio>
#include <cmath>
#include "camera.h"
#include "geodesic.h"
#include "shader.h"
#include "config.h"

// -----------------------------------------------------------------------
// Trevor's module: CPU renderer + post-processing.
//
// Pipeline (mirrors the planned GPU passes):
//   1. trace + shade  → linear HDR framebuffer        (one kernel)
//   2. bright-pass + 3 Gaussian blurs → bloom buffer   (kernel pairs)
//   3. hdr += bloom, then tone-map + gamma → LDR       (one kernel)
//
// CUDA port plan:
//   - Pass 1: move the pixel loop into a __global__ kernel; replace the
//     std::vector<Vec3> with a device float array (3 floats/pixel).
//   - Pass 2: each Gaussian blur becomes a horizontal + vertical kernel
//     pair using shared-memory tiling per block; the three scales run as
//     three independent kernel pairs.
//   - Pass 3: a final kernel adds bloom and tone-maps in place.
// -----------------------------------------------------------------------

struct Image {
    int width, height;
    std::vector<Vec3> pixels;   // row-major [y*width + x]
};

// === Tone-mapping operators ==============================================
inline float reinhard(float x) { return x / (1.0f + x); }

// Narkowicz ACES filmic approximation (operates per channel, linear in).
inline float aces(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float y = (x * (a * x + b)) / (x * (c * x + d) + e);
    return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
}

inline float tone_map(float x) {
    return (cfg::TONE_MAP == cfg::ToneMap::ACES) ? aces(x) : reinhard(x);
}

// === Bloom ===============================================================
// Separable Gaussian blur of an HDR buffer (horizontal then vertical).
// Edges clamp.  GPU: each pass is a shared-memory-tiled kernel.
inline std::vector<Vec3> blur_separable(const std::vector<Vec3>& src,
                                        int W, int H, float sigma) {
    int radius = (int)ceilf(3.0f * sigma);
    std::vector<float> kern(radius + 1);
    float inv2s2 = 1.0f / (2.0f * sigma * sigma), ksum = 0.0f;
    for (int i = 0; i <= radius; ++i) {
        kern[i] = expf(-(float)(i * i) * inv2s2);
        ksum += (i == 0) ? kern[i] : 2.0f * kern[i];
    }
    for (float& k : kern) k /= ksum;

    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

    std::vector<Vec3> tmp(src.size()), out(src.size());
    // Horizontal
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            Vec3 acc = src[y * W + x] * kern[0];
            for (int i = 1; i <= radius; ++i) {
                acc += src[y * W + clampi(x - i, 0, W - 1)] * kern[i];
                acc += src[y * W + clampi(x + i, 0, W - 1)] * kern[i];
            }
            tmp[y * W + x] = acc;
        }
    // Vertical
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            Vec3 acc = tmp[y * W + x] * kern[0];
            for (int i = 1; i <= radius; ++i) {
                acc += tmp[clampi(y - i, 0, H - 1) * W + x] * kern[i];
                acc += tmp[clampi(y + i, 0, H - 1) * W + x] * kern[i];
            }
            out[y * W + x] = acc;
        }
    return out;
}

// Bright-pass + three-scale Gaussian bloom, combined by weight.
inline std::vector<Vec3> compute_bloom(const std::vector<Vec3>& hdr, int W, int H) {
    std::vector<Vec3> bright(hdr.size());
    for (size_t i = 0; i < hdr.size(); ++i) {
        const Vec3& c = hdr[i];
        float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        float over = fmaxf(lum - cfg::BLOOM_THRESHOLD, 0.0f);
        bright[i] = (lum > 1e-4f) ? c * (over / lum) : Vec3(0, 0, 0);
    }

    std::vector<Vec3> b1 = blur_separable(bright, W, H, cfg::BLOOM_SIGMA_1);
    std::vector<Vec3> b2 = blur_separable(bright, W, H, cfg::BLOOM_SIGMA_2);
    std::vector<Vec3> b3 = blur_separable(bright, W, H, cfg::BLOOM_SIGMA_3);

    std::vector<Vec3> bloom(hdr.size());
    for (size_t i = 0; i < hdr.size(); ++i)
        bloom[i] = b1[i] * cfg::BLOOM_WEIGHT_1
                 + b2[i] * cfg::BLOOM_WEIGHT_2
                 + b3[i] * cfg::BLOOM_WEIGHT_3;
    return bloom;
}

inline Image render(const Camera& cam, const GeodesicParams& params = {}) {
    Image img;
    img.width  = cam.width;
    img.height = cam.height;
    img.pixels.resize((size_t)img.width * img.height);

    // --- Pass 1: trace + shade into a linear HDR framebuffer -------------
    // This loop is the direct target for a CUDA __global__ kernel.
    std::vector<Vec3> hdr((size_t)img.width * img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            Ray         ray    = cam.generate_ray(x, y);
            TraceResult result = trace_geodesic(ray.origin, ray.direction, params);
            hdr[(size_t)y * img.width + x] = shade(result);   // HDR, no tone map yet
        }
        if (y % (img.height / 20 + 1) == 0)
            printf("\r  trace %.0f%%", 100.0f * y / img.height), fflush(stdout);
    }
    printf("\r  trace 100%%\n");

    // --- Pass 2: bloom (computed and added in HDR, before tone mapping) --
    printf("  bloom...\n");
    std::vector<Vec3> bloom = compute_bloom(hdr, img.width, img.height);

    // --- Pass 3: add bloom, tone-map, gamma-correct ----------------------
    float inv_gamma = 1.0f / cfg::GAMMA;
    for (size_t i = 0; i < hdr.size(); ++i) {
        Vec3 c = hdr[i] + bloom[i] * cfg::BLOOM_STRENGTH;
        c.x = powf(tone_map(c.x), inv_gamma);
        c.y = powf(tone_map(c.y), inv_gamma);
        c.z = powf(tone_map(c.z), inv_gamma);
        img.pixels[i] = c;
    }

    return img;
}

// Write a PPM P6 (binary) image.
inline void save_ppm(const Image& img, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    for (const Vec3& px : img.pixels) {
        unsigned char rgb[3] = {
            (unsigned char)(fminf(fmaxf(px.x, 0.0f), 1.0f) * 255.0f + 0.5f),
            (unsigned char)(fminf(fmaxf(px.y, 0.0f), 1.0f) * 255.0f + 0.5f),
            (unsigned char)(fminf(fmaxf(px.z, 0.0f), 1.0f) * 255.0f + 0.5f)
        };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}
