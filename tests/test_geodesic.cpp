// -----------------------------------------------------------------------
// Physics + symmetry tests for the Schwarzschild geodesic integrator.
// Build:  see `make test`.  Units: G = c = M = 1  (RS = 2).
// -----------------------------------------------------------------------
#include "test_util.h"
#include "vec3.h"
#include "geodesic.h"
#include "camera.h"
#include "shader.h"
#include "renderer.h"   // reinhard() tone map, for the symmetry render

// Fire a photon that starts at (b, -X0, 0) heading +y, i.e. a straight line
// x = b in the z = 0 plane.  Its impact parameter (perpendicular distance
// from the BH at the origin) is exactly b, and L = b for a unit-speed ray.
static DebugTrace shoot(float b, float X0, GeodesicParams p, float r_inf) {
    Vec3 pos(b, -X0, 0.0f);
    Vec3 dir(0.0f, 1.0f, 0.0f);
    return trace_ray_debug(pos, dir, p, r_inf);
}

static Vec3 final_dir(const DebugTrace& t) {
    return t.samples.back().vel.normalized();
}

// 1. Light deflection vs the GR prediction.
// The leading term is 4M/b, but the integrator solves the exact geodesic so
// it also captures the O(M^2/b^2) term: alpha = 4M/b + (15*pi/4)(M/b)^2 + ...
// That second term is ~6% of the total at b=50, so we compare against the
// 2nd-order GR value (and still print the bare 4M/b for reference).
TEST(WeakFieldDeflection) {
    GeodesicParams p; p.step_size = 1.0f; p.max_steps = 20000;
    const float r_inf = 4000.0f;          // wide domain so the arc is ~asymptotic
    const float bs[] = {50.0f, 100.0f, 200.0f};
    const float k2 = 15.0f * 3.14159265f / 4.0f;   // 2nd-order coefficient
    Vec3 incoming(0.0f, 1.0f, 0.0f);
    bool all = true;
    for (float b : bs) {
        DebugTrace t = shoot(b, 3500.0f, p, r_inf);
        float measured   = angle_between(incoming, final_dir(t));
        float leading     = 4.0f / b;                     // 4M/b
        float predicted  = leading + k2 / (b * b);        // + O(M^2/b^2)
        float err = rel_err(measured, predicted);
        bool ok = (t.outcome == RayOutcome::ESCAPED) && (err < 0.02f);
        all = all && ok;
        printf("WeakFieldDeflection b=%-3.0f: 4M/b=%.4f, GR(2nd)=%.4f, "
               "measured %.4f, err vs GR %.2f%% [%s]\n",
               b, leading, predicted, measured, err*100.0f, PF(ok));
    }
    return all;
}

// 2. Photon-sphere capture threshold: b_crit = 3*sqrt(3) ~ 5.196.
TEST(PhotonSphereCapture) {
    GeodesicParams p; p.step_size = 0.2f; p.max_steps = 60000;
    const float b_crit = 3.0f * std::sqrt(3.0f);

    DebugTrace below = shoot(b_crit - 0.1f, 60.0f, p, R_INF);
    DebugTrace above = shoot(b_crit + 0.1f, 60.0f, p, R_INF);

    bool ok_below = (below.outcome == RayOutcome::CAPTURED);
    bool ok_above = (above.outcome == RayOutcome::ESCAPED);
    bool ok = ok_below && ok_above;

    auto name = [](RayOutcome o) {
        return o == RayOutcome::CAPTURED ? "CAPTURED"
             : o == RayOutcome::ESCAPED  ? "ESCAPED" : "DISK";
    };
    printf("PhotonSphereCapture b_crit=%.3f: b-0.1 -> %s (want CAPTURED), "
           "b+0.1 -> %s (want ESCAPED) [%s]\n",
           b_crit, name(below.outcome), name(above.outcome), PF(ok));
    return ok;
}

// 3. Radial plunge (b = 0) is captured as it crosses the horizon.
// NOTE: the integrator's absorbing radius R_HORIZON = 1.02 sits just inside
// the physical Schwarzschild radius r_s = 2, so termination is at r ~ 1.02.
TEST(HorizonLocation) {
    GeodesicParams p; p.max_steps = 5000;
    DebugTrace t = trace_ray_debug(Vec3(0,-40,0), Vec3(0,1,0), p, R_INF);
    float r_final = t.samples.back().r;
    bool ok = (t.outcome == RayOutcome::CAPTURED) &&
              (r_final <= RS) && (r_final >= 1.0f);
    printf("HorizonLocation b=0: outcome=%s, final r=%.3f "
           "(R_HORIZON=%.2f, r_s=%.1f) [%s]\n",
           t.outcome == RayOutcome::CAPTURED ? "CAPTURED" : "other",
           r_final, R_HORIZON, RS, PF(ok));
    return ok;
}

// 4. Flat-space limit: a ray that stays far from the BH travels straight.
TEST(FlatSpaceLimit) {
    GeodesicParams p; p.step_size = 0.4f; p.max_steps = 20000;
    Vec3 pos(1000.0f, 0.0f, 0.0f);
    Vec3 dir = Vec3(1.0f, 0.05f, 0.0f).normalized();  // outward, slightly angled
    DebugTrace t = trace_ray_debug(pos, dir, p, 2000.0f);
    float dev = angle_between(dir, final_dir(t));
    bool ok = (dev < 1e-3f);
    printf("FlatSpaceLimit (r>=1000): direction drift %.2e rad "
           "(tol 1.0e-03) [%s]\n", dev, PF(ok));
    return ok;
}

// 5. Conservation of E and L along the integration (clean escaping ray).
TEST(ConservationLaws) {
    GeodesicParams p; p.step_size = 0.4f; p.max_steps = 20000;
    DebugTrace t = shoot(20.0f, 98.0f, p, R_INF);   // b=20, escapes cleanly

    float e0 = std::sqrt(t.samples.front().E2);
    float l0 = t.samples.front().Lmeas;
    float emin = e0, emax = e0, lmin = l0, lmax = l0;
    for (const TraceSample& s : t.samples) {
        float e = std::sqrt(s.E2);
        emin = std::fmin(emin, e); emax = std::fmax(emax, e);
        lmin = std::fmin(lmin, s.Lmeas); lmax = std::fmax(lmax, s.Lmeas);
    }
    float e_drift = (emax - emin) / e0;
    float l_drift = (lmax - lmin) / l0;
    bool ok = (t.outcome == RayOutcome::ESCAPED) &&
              (e_drift < 1e-3f) && (l_drift < 1e-3f);
    printf("ConservationLaws (b=20, %d steps): E drift %.3e, L drift %.3e "
           "(tol 1.0e-03) [%s]\n", t.steps, e_drift, l_drift, PF(ok));
    return ok;
}

// 6. Time reversal: trace A->B, then B with reversed direction retraces to A.
TEST(TimeReversal) {
    GeodesicParams p; p.step_size = 0.4f; p.max_steps = 20000;
    Vec3 A(20.0f, -98.0f, 0.0f);
    Vec3 dirA(0.0f, 1.0f, 0.0f);

    DebugTrace fwd = trace_ray_debug(A, dirA, p, R_INF);
    Vec3 B    = fwd.samples.back().pos;
    Vec3 dirB = fwd.samples.back().vel.normalized();

    DebugTrace rev = trace_ray_debug(B, -dirB, p, R_INF);
    float closest = 1e30f;
    for (const TraceSample& s : rev.samples)
        closest = std::fmin(closest, (s.pos - A).norm());

    float r_A = A.norm();
    bool ok = (closest / r_A < 0.01f);     // within 1% of the start radius
    printf("TimeReversal: reverse ray passes within %.4f of A (|A|=%.1f, "
           "%.3f%%) [%s]\n", closest, r_A, 100.0f*closest/r_A, PF(ok));
    return ok;
}

// 7. Spherical symmetry: rotating (pos, dir) rotates the whole trajectory.
TEST(SphericalSymmetry) {
    GeodesicParams p; p.step_size = 0.4f; p.max_steps = 20000;
    // b = 30 > R_DISK_MAX/2... actually > R_DISK_MAX(=24): closest approach
    // exceeds the disk, so the ray escapes regardless of orientation.
    Vec3 pos(30.0f, -120.0f, 0.0f);
    Vec3 dir(0.0f, 1.0f, 0.0f);
    Vec3 axis(1.0f, 2.0f, 3.0f);
    float ang = 0.6f;   // arbitrary rotation

    DebugTrace base = trace_ray_debug(pos, dir, p, R_INF);
    DebugTrace rot  = trace_ray_debug(rotate_axis(pos, axis, ang),
                                      rotate_axis(dir, axis, ang), p, R_INF);

    Vec3 expected = rotate_axis(final_dir(base), axis, ang);
    Vec3 got      = final_dir(rot);
    Vec3 d = got - expected;
    float maxd = std::fmax(std::fabs(d.x), std::fmax(std::fabs(d.y), std::fabs(d.z)));
    bool ok = (base.outcome == RayOutcome::ESCAPED) &&
              (rot.outcome == RayOutcome::ESCAPED) && (maxd < 1e-4f);
    printf("SphericalSymmetry: max|R(dir_base) - dir_rot| = %.2e "
           "(tol 1.0e-04) [%s]\n", maxd, PF(ok));
    return ok;
}

// 8. Equatorial-plane symmetry: an edge-on (camera z=0) render is mirror
// symmetric top<->bottom for the BH shadow and disk.  The procedural
// starfield is NOT z-flip symmetric, so escaped-ray colors are excluded;
// we still require escaped *outcomes* to mirror.
TEST(EquatorialPlaneSymmetry) {
    const int W = 64, H = 64;
    Camera cam = Camera::look_at(Vec3(0,-35,0), Vec3(0,0,0), Vec3(0,0,1),
                                 55.0f, W, H);
    GeodesicParams params;

    auto tonemapped = [](Vec3 c) {
        return Vec3(std::pow(reinhard(c.x), 1.0f/2.2f),
                    std::pow(reinhard(c.y), 1.0f/2.2f),
                    std::pow(reinhard(c.z), 1.0f/2.2f));
    };

    int   outcome_mismatch = 0, structural = 0;
    float max_diff = 0.0f;       // max channel diff over non-skybox mirror pairs
    for (int y = 0; y < H/2; ++y) {
        for (int x = 0; x < W; ++x) {
            Ray ra = cam.generate_ray(x, y);
            Ray rb = cam.generate_ray(x, H-1-y);
            TraceResult ta = trace_geodesic(ra.origin, ra.direction, params);
            TraceResult tb = trace_geodesic(rb.origin, rb.direction, params);
            if (ta.outcome != tb.outcome) { ++outcome_mismatch; continue; }
            if (ta.outcome == RayOutcome::ESCAPED) continue;   // starfield excluded
            ++structural;
            Vec3 ca = tonemapped(shade(ta));
            Vec3 cb = tonemapped(shade(tb));
            max_diff = std::fmax(max_diff,
                std::fmax(std::fabs(ca.x-cb.x),
                std::fmax(std::fabs(ca.y-cb.y), std::fabs(ca.z-cb.z))));
        }
    }
    bool ok = (outcome_mismatch == 0) && (max_diff <= 1.0f/255.0f) && (structural > 0);
    printf("EquatorialPlaneSymmetry %dx%d: %d structural pairs, "
           "outcome mismatches=%d, max color diff=%.4f/255 (tol 1.0) [%s]\n",
           W, H, structural, outcome_mismatch, max_diff*255.0f, PF(ok));
    return ok;
}

int main() {
    printf("=== Geodesic / physics tests (M=1, RS=2) ===\n");
    return run_all_tests();
}
