#pragma once
#include "vec3.h"
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// -----------------------------------------------------------------------
// Trevor's module: pinhole camera ray generator
// -----------------------------------------------------------------------

struct Ray {
    Vec3 origin;
    Vec3 direction;   // unit vector
};

struct Camera {
    Vec3  pos;
    Vec3  forward;    // unit, points toward scene
    Vec3  right;      // unit, screen-right
    Vec3  up;         // unit, screen-up
    float half_h;     // tan(fov_v / 2)
    float half_w;     // half_h * aspect
    int   width, height;

    // Build a camera looking from pos toward target.
    // world_up: hint vector for orientation (typically (0,0,1) since disk is z=0).
    // fov_deg : vertical field of view in degrees.
    static Camera look_at(
        Vec3 pos, Vec3 target, Vec3 world_up,
        float fov_deg, int w, int h)
    {
        Camera cam;
        cam.pos    = pos;
        cam.width  = w;
        cam.height = h;
        float fov_v = fov_deg * (3.14159265358979f / 180.0f);
        cam.half_h  = tanf(fov_v * 0.5f);
        cam.half_w  = cam.half_h * ((float)w / (float)h);
        cam.forward = (target - pos).normalized();
        cam.right   = cam.forward.cross(world_up).normalized();
        cam.up      = cam.right.cross(cam.forward);
        return cam;
    }

    // Generate the ray for pixel (px, py) in [0,width) x [0,height).
    // Pixel centers sit at (px+0.5, py+0.5); y=0 is the top row.
    __host__ __device__ Ray generate_ray(int px, int py) const {
        float u = (2.0f * ((float)px + 0.5f) / (float)width  - 1.0f) * half_w;
        float v = (1.0f - 2.0f * ((float)py + 0.5f) / (float)height) * half_h;

        Ray ray;
        ray.origin    = pos;
        ray.direction = (forward + right * u + up * v).normalized();
        return ray;
    }
};
