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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

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
