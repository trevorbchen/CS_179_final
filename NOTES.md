# Schwarzschild Black Hole Renderer — CS 179 Final Project

## Build (CPU baseline)

```bash
# Linux / macOS / MSYS2 with MinGW
make
./renderer            # 800×600
./renderer 1920 1080  # custom resolution

# Windows MSVC
cl /O2 /std:c++17 /EHsc main.cpp /Fe:renderer.exe
```

Output: `output.ppm` (view with any image viewer; convert with ImageMagick).

---

## Module ownership

| File | Owner | Role |
|------|-------|------|
| `vec3.h` | shared | 3-D float vector, `__host__`/`__device__` stubs |
| `geodesic.h` | **Kevin** | RK4 null-geodesic integrator |
| `camera.h` | **Trevor** | Pinhole camera, ray generation |
| `shader.h` | **Trevor** | Starfield skybox, disk blackbody, tone map |
| `renderer.h` | **Trevor** | Pixel loop (→ CUDA kernel) |
| `main.cpp` | shared | Entry point, argument parsing |

---

## Physics summary

### Coordinates
Geometric units G = c = M = 1, so the Schwarzschild radius r_s = 2.
The black hole sits at the origin; the accretion disk is the z = 0 plane.

### Why orbital-plane integration?
Schwarzschild is spherically symmetric, so every photon path lies in a
fixed plane (the orbital plane).  We project to polar coords (r, φ) in
that plane and integrate a 3-component ODE:

```
dr/dλ   = v_r
dv_r/dλ = L²(r − 3) / r⁴      ← exact Schwarzschild geodesic equation
dφ/dλ   = L / r²
```

L = r · v_tangential is conserved (angular momentum).  E (energy) drops
out when you differentiate the null condition, so it never appears.
Integrating 3 floats per ray instead of 8 (full 4-D phase space) halves
memory bandwidth on the GPU.

### RK4 with adaptive step
Near the photon sphere (r = 3) the curvature is high.  We scale h by
(r / 5r_s)² when r < 5r_s, floored at 0.005.

### Termination
- r ≤ R_HORIZON (= 1.02) → **CAPTURED** (black pixel)
- r ≥ R_INF (= 500) → **ESCAPED** → skybox lookup on asymptotic direction
- z changes sign with ρ ∈ [6, 24] → **DISK** → blackbody shade

---

## CUDA porting plan

1. Rename `.cpp` → `.cu`; remove the `#ifndef __CUDACC__` stubs (nvcc
   provides the real `__host__`/`__device__`).

2. Replace the double `for` loop in `renderer.h::render()` with:

   ```cuda
   __global__ void render_kernel(Camera cam, GeodesicParams p,
                                 float* out, int W, int H) {
       int x = blockIdx.x * blockDim.x + threadIdx.x;
       int y = blockIdx.y * blockDim.y + threadIdx.y;
       if (x >= W || y >= H) return;
       Ray r = cam.generate_ray(x, y);
       TraceResult tr = trace_geodesic(r.origin, r.direction, p);
       Vec3 c = shade(tr);
       // tone-map and write to out[3*(y*W+x) + {0,1,2}]
   }
   ```

3. All functions already have `__host__ __device__` qualifiers; no other
   changes are needed in `geodesic.h`, `camera.h`, or `shader.h`.

4. Use `dim3 block(16, 16)` and `dim3 grid(ceil(W/16), ceil(H/16))`.

5. `std::vector` in `Image` becomes a `cudaMalloc`-ed float array;
   copy back with `cudaMemcpy` before saving the PPM.

---

## Camera setup

- Position: (0, −35, 5) — to the side and slightly above the disk
- Target: origin (black hole)
- World up: +z (disk normal)
- FoV: 55° vertical

The central pixel shoots straight at the BH → captured (black shadow).
Off-center rays curve around the BH; highly bent ones can complete
partial orbits and hit the disk from the far side (Einstein ring, photon
ring visible as a bright arc near the shadow boundary).
