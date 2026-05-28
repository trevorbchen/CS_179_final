# CODE INSTRUCTIONS (run brew install jpeg-turbo if dependency missing)
make
./renderer (default resolution: 800 x 600, default camera position: (0, -35, 1), )
./renderer [W H [cam_x cam_y cam_z [target_x target_y target_z [fov]]]]

Defaults
Parameter	    Default 	Notes
Width × Height	800 600	
Camera position	(0, -35, 1)	35 units away from the black hole, 1 unit above the disk
Target	        (0, 0, 0)	The black hole
FOV	            55°	        Vertical field of view
# Default
./renderer
# Higher resolution, same camera
./renderer 1920 1080
# Overhead view looking straight down at the disk
./renderer 1920 1080  0 0 60  0 0 0  70
# Side view (edge-on to the disk), lower FOV
./renderer 800 600  35 0 0  0 0 0  40

Output:
kpeng2@titan:~/CS_179_final$ make
g++ -O2 -std=c++17 -Wall -Wextra -I/opt/homebrew/opt/jpeg-turbo/include -o renderer main.cpp -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg
kpeng2@titan:~/CS_179_final$ ./renderer
Schwarzschild black hole renderer
  Resolution : 800 x 600
  Camera pos : (0.0, -35.0, 1.0)
  Target     : (0.0, 0.0, 0.0)
  FOV        : 55.0 deg
  RS=2.0  disk=[6.0, 24.0]  R_inf=500
Tracing...
  100%
Saved output.ppm
Saved output.jpg


# TESTS
make test

Geodesic / physics suite — tests/test_geodesic.cpp

1. WeakFieldDeflection — Fires photons at impact parameters b=50, 100, 200 (r_inf=4000) and compares the measured bending to GR's α = 4M/b + (15π/4)(M/b)², tol 2%.
2/ PhotonSphereCapture — Tests b_crit = 3√3 ≈ 5.196: b−0.1 must be CAPTURED, b+0.1 must ESCAPE.
3. HorizonLocation — Radial ray (b=0) terminates as CAPTURED with final r ≤ r_s=2 (actual ≈ R_HORIZON=1.02).
4. FlatSpaceLimit — Ray at r≥1000 angled outward; final direction matches initial within 1e-3 rad.
5. ConservationLaws — Tracks E and L over the full trajectory of a b=20 escape; drift in each must be < 0.1%.
6. TimeReversal — Trace A→B, reverse direction at B, retrace; closest pass to A must be within 1% of |A|.
7. SphericalSymmetry — Rotating (pos, dir) by an arbitrary axis rotates the final direction by the same rotation, max component diff < 1e-4.
8. EquatorialPlaneSymmetry — Renders 64×64 edge-on (camera z=0); mirror pixel pairs have identical outcomes and bit-identical disk/shadow colors. Starfield colors excluded (not z-flip symmetric).
Shading / camera suite — tests/test_shading.cpp

9. ToneMappingBounded — reinhard(x) output stays in [0,1] for x ∈ {0, 1e-4, …, 1e9}.
10. DiskTemperatureFalloff — luminance(shade_disk(R_DISK_MIN)) > luminance(shade_disk(R_DISK_MAX)).
11. CameraCenterPixelForward — On a 201×201 (odd-sized) camera, the center pixel's ray direction equals camera forward within 1e-4 rad.
12. SkyboxDeterminism — sample_skybox(d) returns bit-identical RGB on repeated calls, across 4 directions.

Output:
=== Geodesic / physics tests (M=1, RS=2) ===
WeakFieldDeflection b=50 : 4M/b=0.0800, GR(2nd)=0.0847, measured 0.0851, err vs GR 0.44% [PASS]
WeakFieldDeflection b=100: 4M/b=0.0400, GR(2nd)=0.0412, measured 0.0412, err vs GR 0.09% [PASS]
WeakFieldDeflection b=200: 4M/b=0.0200, GR(2nd)=0.0203, measured 0.0203, err vs GR 0.00% [PASS]
PhotonSphereCapture b_crit=5.196: b-0.1 -> CAPTURED (want CAPTURED), b+0.1 -> ESCAPED (want ESCAPED) [PASS]
HorizonLocation b=0: outcome=CAPTURED, final r=1.015 (R_HORIZON=1.02, r_s=2.0) [PASS]
FlatSpaceLimit (r>=1000): direction drift 0.00e+00 rad (tol 1.0e-03) [PASS]
ConservationLaws (b=20, 1501 steps): E drift 7.155e-07, L drift 1.431e-06 (tol 1.0e-03) [PASS]
TimeReversal: reverse ray passes within 0.1621 of A (|A|=100.0, 0.162%) [PASS]
SphericalSymmetry: max|R(dir_base) - dir_rot| = 1.79e-07 (tol 1.0e-04) [PASS]
EquatorialPlaneSymmetry 64x64: 386 structural pairs, outcome mismatches=0, max color diff=0.0000/255 (tol 1.0) [PASS]

8/8 tests passed.

=== Shading / camera tests ===
ToneMappingBounded (Reinhard): output range [0.0000, 1.0000] subset of [0,1] [PASS]
DiskTemperatureFalloff: L(r_in=6.0)=1.000 > L(r_out=24.0)=0.028 [PASS]
CameraCenterPixelForward (201x201): center ray vs forward = 0.00e+00 rad (tol 1.0e-04) [PASS]
SkyboxDeterminism: 4 directions, repeated calls bit-identical [PASS]

4/4 tests passed.

# GPU Implementation Plan

The main change required is replacing the double `for` loop in `renderer.h::render()` with a render_kernel

Basic Sketch:
__global__ void render_kernel(Camera cam, GeodesicParams p, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    Ray r = cam.generate_ray(x, y);
    TraceResult tr = trace_geodesic(r.origin, r.direction, p);
    Vec3 c = shade(tr);
    // tone-map and write to out buffer
}

geodesic.h, camera.h, and shader.h don't require modifications to be used with CUDA

Parallelization strategy:
The renderer will assign one GPU thread per pixel. Each thread independently generates its camera ray, inputs it into the full RK4 geodesic integration gravity simulator to determine whether the ray is captured by the black hole, escapes to the skybox, or hits the accretion disk, then shades and tone-maps the result, writing three floats to a flat cudaMalloc-ed output buffer. Because every pixel's ray is completely independent, this an embarrassingly parallel task.

