#pragma once
#include <cmath>

// Simple 3-vector. __host__ __device__ annotations added as stubs for easy CUDA port.
// Replace the macros with real CUDA qualifiers when porting.
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    __host__ __device__ Vec3 operator*(float t)        const { return {x*t,   y*t,   z*t};   }
    __host__ __device__ Vec3 operator/(float t)        const { return {x/t,   y/t,   z/t};   }
    __host__ __device__ Vec3  operator-()               const { return {-x, -y, -z}; }
    __host__ __device__ Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    __host__ __device__ Vec3& operator*=(float t)       { x*=t;   y*=t;   z*=t;   return *this; }

    __host__ __device__ float dot(const Vec3& b)  const { return x*b.x + y*b.y + z*b.z; }
    __host__ __device__ Vec3  cross(const Vec3& b) const {
        return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x};
    }
    __host__ __device__ float norm2() const { return x*x + y*y + z*z; }
    __host__ __device__ float norm()  const { return sqrtf(norm2()); }
    __host__ __device__ Vec3  normalized() const { return (*this) / norm(); }
};

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) { return v * t; }

// -----------------------------------------------------------------------
// Vec3 <-> CUDA float3 bridges (device build only).
//
// The GPU port stores per-pixel results (TerminalState, HDR framebuffer)
// using CUDA's built-in float3/float4 vector types: they have the natural
// 12/16-byte layout the hardware expects and let the renderer kernel read
// the gravity kernel's output with coalesced, aligned loads.  Vec3 keeps
// the shared host/device math operators used by the physics + shading.
// These thin wrappers convert between the two without copying semantics.
//
// Guarded by __CUDACC__ so the pure-C++ CPU baseline (which never includes
// the CUDA headers) is completely unaffected.
// -----------------------------------------------------------------------
#ifdef __CUDACC__
__host__ __device__ inline float3 vec3_to_float3(const Vec3& v) { return float3{v.x, v.y, v.z}; }
__host__ __device__ inline Vec3   float3_to_vec3(const float3& v) { return Vec3(v.x, v.y, v.z); }
#endif
