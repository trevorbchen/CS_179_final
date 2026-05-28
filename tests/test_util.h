#pragma once
// -----------------------------------------------------------------------
// Minimal header-only test framework (no GoogleTest dependency).
//
//   TEST(MyTest) { ...; return pass; }
//   int main() { return run_all_tests(); }
//
// Each TEST is auto-registered and must print its own one-line PASS/FAIL
// summary (use PF() for the tag) and return whether it passed.
// -----------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include "vec3.h"

typedef bool (*TestFn)();

struct TestCase { const char* name; TestFn fn; };

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int register_test(const char* name, TestFn fn) {
    test_registry().push_back({name, fn});
    return 0;
}

#define TEST(name)                                                       \
    static bool name();                                                  \
    static int _reg_##name = register_test(#name, name);                 \
    static bool name()

inline int run_all_tests() {
    int passed = 0;
    auto& reg = test_registry();
    for (const TestCase& tc : reg)
        if (tc.fn()) ++passed;
    printf("\n%d/%d tests passed.\n", passed, (int)reg.size());
    return (passed == (int)reg.size()) ? 0 : 1;
}

// ---- assertion / formatting helpers -------------------------------------

inline const char* PF(bool b) { return b ? "PASS" : "FAIL"; }

inline bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

inline float rel_err(float measured, float predicted) {
    return std::fabs((measured - predicted) / predicted);
}

// Angle (radians) between two vectors.
inline float angle_between(Vec3 a, Vec3 b) {
    float d = a.normalized().dot(b.normalized());
    if (d >  1.0f) d =  1.0f;
    if (d < -1.0f) d = -1.0f;
    return std::acos(d);
}

// Rotate v about a unit axis by angle (radians) — Rodrigues' formula.
inline Vec3 rotate_axis(Vec3 v, Vec3 axis, float angle) {
    Vec3  k = axis.normalized();
    float c = std::cos(angle), s = std::sin(angle);
    return v * c + k.cross(v) * s + k * (k.dot(v) * (1.0f - c));
}

// Rec. 709 relative luminance of a linear RGB color.
inline float luminance(Vec3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }
