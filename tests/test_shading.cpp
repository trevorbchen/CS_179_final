// -----------------------------------------------------------------------
// Shading / camera tests: tone mapping, disk falloff, camera, skybox.
// -----------------------------------------------------------------------
#include "test_util.h"
#include "vec3.h"
#include "geodesic.h"
#include "shader.h"
#include "camera.h"
#include "renderer.h"   // reinhard()

// 9. Tone mapping is bounded to [0,1] for any (non-negative) HDR input.
// NOTE: the renderer uses the Reinhard operator x/(1+x), not ACES.
TEST(ToneMappingBounded) {
    const float inputs[] = {0.0f, 1e-4f, 0.1f, 0.5f, 1.0f, 4.0f, 50.0f,
                            1e3f, 1e6f, 1e9f};
    float lo = 1e30f, hi = -1e30f;
    bool ok = true;
    for (float x : inputs) {
        float y = reinhard(x);
        lo = std::fmin(lo, y); hi = std::fmax(hi, y);
        if (!(y >= 0.0f && y <= 1.0f)) ok = false;
    }
    printf("ToneMappingBounded (Reinhard): output range [%.4f, %.4f] "
           "subset of [0,1] [%s]\n", lo, hi, PF(ok));
    return ok;
}

// 10. Disk temperature falloff: inner edge is more luminous than outer edge.
TEST(DiskTemperatureFalloff) {
    float lum_in  = luminance(shade_disk(R_DISK_MIN));   // ISCO, hottest
    float lum_out = luminance(shade_disk(R_DISK_MAX));    // outer edge, coolest
    bool ok = (lum_in > lum_out) && (lum_out >= 0.0f);
    printf("DiskTemperatureFalloff: L(r_in=%.1f)=%.3f > L(r_out=%.1f)=%.3f [%s]\n",
           R_DISK_MIN, lum_in, R_DISK_MAX, lum_out, PF(ok));
    return ok;
}

// 11. Center pixel of an odd-sized image shoots exactly along camera forward.
TEST(CameraCenterPixelForward) {
    const int W = 201, H = 201;   // odd -> a true center pixel exists
    Camera cam = Camera::look_at(Vec3(0,-35,1), Vec3(0,0,0), Vec3(0,0,1),
                                 55.0f, W, H);
    Ray r = cam.generate_ray(W/2, H/2);
    float dev = angle_between(r.direction, cam.forward);
    bool ok = (dev < 1e-4f);
    printf("CameraCenterPixelForward (%dx%d): center ray vs forward = %.2e rad "
           "(tol 1.0e-04) [%s]\n", W, H, dev, PF(ok));
    return ok;
}

// 12. Skybox sampling is deterministic: same direction -> identical color.
TEST(SkyboxDeterminism) {
    Vec3 dirs[] = { Vec3(1,0,0), Vec3(0.3f,0.8f,-0.5f), Vec3(-2,1,4),
                    Vec3(0,-1,0) };
    bool ok = true;
    for (Vec3 d : dirs) {
        Vec3 a = sample_skybox(d);
        Vec3 b = sample_skybox(d);
        if (!(a.x==b.x && a.y==b.y && a.z==b.z)) ok = false;
    }
    printf("SkyboxDeterminism: %zu directions, repeated calls bit-identical [%s]\n",
           sizeof(dirs)/sizeof(dirs[0]), PF(ok));
    return ok;
}

int main() {
    printf("=== Shading / camera tests ===\n");
    return run_all_tests();
}
