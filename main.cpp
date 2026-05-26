#include <cstdio>
#include "renderer.h"

int main(int argc, char* argv[]) {
    // Resolution
    int W = 800, H = 600;
    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }

    // Camera: slightly above the equatorial (accretion) disk, looking at the BH.
    // Units: geometric (G = c = M = 1), so RS = 2 and the disk spans [6, 24].
    Vec3 cam_pos(0.0f, -35.0f, 5.0f);
    Vec3 target  (0.0f,   0.0f, 0.0f);
    Vec3 world_up(0.0f,   0.0f, 1.0f);   // z is the disk normal

    Camera cam = Camera::look_at(cam_pos, target, world_up, 55.0f, W, H);

    GeodesicParams params;
    params.step_size = 0.4f;
    params.max_steps = 5000;

    printf("Schwarzschild black hole renderer\n");
    printf("  Resolution : %d x %d\n", W, H);
    printf("  Camera pos : (%.1f, %.1f, %.1f)\n",
           cam_pos.x, cam_pos.y, cam_pos.z);
    printf("  RS=%.1f  disk=[%.1f, %.1f]  R_inf=%.0f\n",
           RS, R_DISK_MIN, R_DISK_MAX, R_INF);
    printf("Tracing...\n");

    Image img = render(cam, params);

    const char* out = "output.ppm";
    save_ppm(img, out);
    printf("Saved %s\n", out);
    return 0;
}
