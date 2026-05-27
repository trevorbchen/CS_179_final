#include <cstdio>
#include <cstdlib>
#include "renderer.h"
#include "config.h"

// Usage:
//   ./renderer [W H] [camX camY camZ] [outfile.ppm]
// With no extra args the defaults from config.h are used. The camera/output
// overrides let us batch-render multiple perspectives without recompiling.
int main(int argc, char* argv[]) {
    int W = 800, H = 600;
    Vec3 cam_pos   = cfg::CAM_POS;
    const char* out = "output.ppm";

    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }
    if (argc >= 6) { cam_pos = Vec3((float)atof(argv[3]),
                                    (float)atof(argv[4]),
                                    (float)atof(argv[5])); }
    if (argc >= 7) { out = argv[6]; }

    // z is the disk normal; the disk lies in the z = 0 plane. A small |z|
    // offset is a near-edge-on tilt (Interstellar framing).
    Camera cam = Camera::look_at(cam_pos, cfg::CAM_TARGET, cfg::CAM_UP,
                                 cfg::CAM_FOV, W, H);

    GeodesicParams params;
    params.step_size = 0.4f;
    params.max_steps = 5000;

    // Try to load an equirectangular skybox; falls back to procedural stars.
    load_skybox(cfg::SKYBOX_PATH);

    printf("Schwarzschild black hole renderer\n");
    printf("  Resolution : %d x %d\n", W, H);
    printf("  Camera pos : (%.1f, %.1f, %.1f)\n",
           cam_pos.x, cam_pos.y, cam_pos.z);
    printf("  RS=%.1f  disk=[%.1f, %.1f]  R_inf=%.0f\n",
           RS, R_DISK_MIN, R_DISK_MAX, R_INF);
    printf("  Tone map   : %s\n",
           cfg::TONE_MAP == cfg::ToneMap::ACES ? "ACES" : "Reinhard");
    printf("Tracing...\n");

    Image img = render(cam, params);

    save_ppm(img, out);
    printf("Saved %s\n", out);
    return 0;
}
