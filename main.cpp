#include <cstdio>
#include <cstdlib>
#include "renderer.h"

int main(int argc, char* argv[]) {
    // Usage: ./renderer [W H [cam_x cam_y cam_z [target_x target_y target_z [fov]]]]

    // Resolution
    int W = 800, H = 600;
    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }

    // Camera defaults: slightly above the equatorial (accretion) disk, looking at the BH.
    // Units: geometric (G = c = M = 1), so RS = 2 and the disk spans [6, 24].
    Vec3  cam_pos(0.0f, -35.0f, 1.0f);
    Vec3  target (0.0f,   0.0f, 0.0f);
    Vec3  world_up(0.0f,  0.0f, 1.0f);   // z is the disk normal
    float fov = 55.0f;

    if (argc >= 6) {
        cam_pos.x = atof(argv[3]);
        cam_pos.y = atof(argv[4]);
        cam_pos.z = atof(argv[5]);
    }
    if (argc >= 9) {
        target.x = atof(argv[6]);
        target.y = atof(argv[7]);
        target.z = atof(argv[8]);
    }
    if (argc >= 10) {
        fov = atof(argv[9]);
    }

    Camera cam = Camera::look_at(cam_pos, target, world_up, fov, W, H);

    GeodesicParams params;
    params.step_size = 0.4f;
    params.max_steps = 5000;

    printf("Schwarzschild black hole renderer\n");
    printf("  Resolution : %d x %d\n", W, H);
    printf("  Camera pos : (%.1f, %.1f, %.1f)\n",
           cam_pos.x, cam_pos.y, cam_pos.z);
    printf("  Target     : (%.1f, %.1f, %.1f)\n",
           target.x, target.y, target.z);
    printf("  FOV        : %.1f deg\n", fov);
    printf("  RS=%.1f  disk=[%.1f, %.1f]  R_inf=%.0f\n",
           RS, R_DISK_MIN, R_DISK_MAX, R_INF);
    printf("Tracing...\n");

    Image img = render(cam, params);

    save_ppm(img, "output.ppm");
    printf("Saved output.ppm\n");
    save_jpg(img, "output.jpg");
    printf("Saved output.jpg\n");
    return 0;
}
