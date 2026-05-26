#pragma once
#include <vector>
#include <cstdio>
#include <cmath>
#include "camera.h"
#include "geodesic.h"
#include "shader.h"

// -----------------------------------------------------------------------
// CPU renderer: iterates over pixels, calls geodesic + shader per ray.
//
// CUDA port plan:
//   - Move this loop body into a __global__ kernel.
//   - Replace std::vector<Vec3> with a device float array (3 floats/pixel).
//   - Each thread handles one pixel: blockIdx and threadIdx replace (x, y).
//   - cam, params can be passed by value or stored in __constant__ memory.
// -----------------------------------------------------------------------

struct Image {
    int width, height;
    std::vector<Vec3> pixels;   // row-major [y*width + x]
};

// Tone-map a linear HDR value to [0,1] with a simple Reinhard operator.
inline float reinhard(float x) { return x / (1.0f + x); }

inline Image render(const Camera& cam, const GeodesicParams& params = {}) {
    Image img;
    img.width  = cam.width;
    img.height = cam.height;
    img.pixels.resize(img.width * img.height);

    // This loop is the direct target for a CUDA __global__ kernel.
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            Ray         ray    = cam.generate_ray(x, y);
            TraceResult result = trace_geodesic(ray.origin, ray.direction, params);
            Vec3        color  = shade(result);

            // Reinhard tone-map + gamma correction (γ = 2.2 approximated)
            color.x = powf(reinhard(color.x), 1.0f / 2.2f);
            color.y = powf(reinhard(color.y), 1.0f / 2.2f);
            color.z = powf(reinhard(color.z), 1.0f / 2.2f);

            img.pixels[y * img.width + x] = color;
        }

        // Progress bar (one dot per row)
        if (y % (img.height / 20 + 1) == 0)
            printf("\r  %.0f%%", 100.0f * y / img.height), fflush(stdout);
    }
    printf("\r  100%%\n");

    return img;
}

// Write a PPM P6 (binary) image.
inline void save_ppm(const Image& img, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    for (const Vec3& px : img.pixels) {
        unsigned char rgb[3] = {
            (unsigned char)(fminf(px.x, 1.0f) * 255.0f + 0.5f),
            (unsigned char)(fminf(px.y, 1.0f) * 255.0f + 0.5f),
            (unsigned char)(fminf(px.z, 1.0f) * 255.0f + 0.5f)
        };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}
