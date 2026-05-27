# Schwarzschild Black Hole Renderer — CS 179 Final Project

## Build (CPU baseline)

```bash
# Linux / macOS / MSYS2 with MinGW
make
./renderer            # 800×600
./renderer 1920 1080  # custom resolution

# Windows MSVC
cl /O2 /std:c++17 /EHsc main.cpp stb_image_impl.cpp /Fe:renderer.exe
```

Output: `output.ppm` (view with any image viewer; convert with ImageMagick or
`python3 -c "from PIL import Image; Image.open('output.ppm').save('out.png')"`).

The build links two TUs: `main.cpp` (our code) and `stb_image_impl.cpp` (the
vendored `stb_image.h` implementation, compiled with warnings off).

---

## Module ownership

| File | Owner | Role |
|------|-------|------|
| `vec3.h` | shared | 3-D float vector, `__host__`/`__device__` stubs |
| `geodesic.h` | **Kevin** | RK4 null-geodesic integrator |
| `camera.h` | **Trevor** | Pinhole camera, ray generation |
| `shader.h` | **Trevor** | Procedural disk texture, skybox sampling, starfield |
| `renderer.h` | **Trevor** | Pixel loop + post-processing (bloom, tone map) |
| `config.h` | **Trevor** | All tunable "look" constants (`cfg::…`) |
| `stb_image.h` | vendored | Single-header image loader (public domain) |
| `main.cpp` | shared | Entry point, argument parsing, skybox load |

---

## Visual features

The shading/post pipeline (Trevor, Module 2) aims for the Interstellar look.
All "look" parameters live in **`config.h`** (`namespace cfg`) and can be
retuned without touching algorithm code:

- **Procedural disk texture** (`shader.h::shade_disk`): Shakura-Sunyaev
  temperature `T ∝ r^{-3/4}` mapped through a warm 4-color gradient (deep red →
  orange → tan-white → pinkish white, no blue), modulated by two layers of
  value-noise fBm (gain 0.5, lacunarity 2.0), subtle Keplerian banding, and a
  handful of Gaussian "hot spots" at fixed `(r, φ)`. The turbulence floor/gain
  carve dark filament lanes that stay visible even where the lensed ring
  saturates. Output is HDR (`DISK_HDR_SCALE`). The disk has finite thickness
  (`R_DISK_H`), so it reads as a band even exactly edge-on.
- **Skybox** (`shader.h`): loads an equirectangular image into a linear float
  buffer and bilinearly samples it for escaped rays (wraps in `u`, clamps in
  `v`). Falls back to the procedural starfield if the file is missing.
- **Multi-scale bloom** (`renderer.h`): bright-pass at `BLOOM_THRESHOLD`, three
  separable Gaussian blurs (σ = 4 / 12 / 32) combined by weight, added to the
  HDR framebuffer **before** tone mapping.
- **Tone mapping** (`renderer.h`): ACES filmic by default; Reinhard still
  available via `cfg::TONE_MAP`. Gamma 2.2.
- **Camera** (`config.h`): exactly edge-on (in the disk plane) so the lensed
  disk wraps symmetrically over the top and under the bottom of the shadow.

### Skybox image (optional)

For a real Milky Way background, download a free **equirectangular** panorama
and save it as **`data/skybox.jpg`** (path set by `cfg::SKYBOX_PATH`):

- ESO — "The Milky Way panorama" (eso0932a): https://www.eso.org/public/images/eso0932a/
- NASA/Goddard — "Deep Star Maps" / Gaia all-sky maps

8-bit JPG/PNG is fine — `stbi_loadf` converts it to linear. Without the file
the renderer prints a warning and uses the procedural starfield.

### Tunable constants (`config.h`)

Camera pose/FoV, tone-map operator + gamma, disk HDR scale, fBm frequencies and
weights, turbulence floor/gain, banding, hot-spot positions/sizes, skybox
path/intensity, and all bloom parameters (threshold, three sigmas, weights,
final strength). Disk geometry (`R_DISK_MIN/MAX`, `R_DISK_H` half-thickness)
lives in `geodesic.h` with the other scene constants.

> Note — small Module 1 (geodesic.h) deviations from the "don't touch" rule,
> both scene-geometry only (the RK4 integrator and the geodesic ODE are
> unchanged):
> - `TraceResult` carries an extra `disk_phi` (crossing azimuth) so the surface
>   texture has a `(r, φ)` coordinate — just records an already-computed value.
> - the disk hit-test is a finite-thickness **slab** instead of a plane, so the
>   disk is visible edge-on.

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
- ray enters disk slab (ρ ∈ [6, 24] and |z| ≤ R_DISK_H) → **DISK** → procedural shade

The disk is modeled as an opaque **slab** of half-thickness `R_DISK_H`
(geodesic.h), not a zero-thickness plane — so it stays visible exactly edge-on
(a plane would project to a sub-pixel line). The first integration step landing
inside the slab volume is the surface hit.

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

Defined in `config.h` (`cfg::CAM_*`):

- Position: (0, −30, 0) — exactly edge-on, in the disk plane
- Target: origin (black hole)
- World up: +z (disk normal)
- FoV: 55° vertical

At exactly edge-on the image is perfectly top/bottom (and left/right)
symmetric: the lensed disk wraps the shadow into a clean ring, the
finite-thickness disk reads as a horizontal band through the middle, and there
is no asymmetric "chamfer" at the band/arc junction.

Because the disk now has thickness (`R_DISK_H`), `z = 0` keeps both the band
and perfect symmetry. A nonzero `CAM_POS.z` tilts the view to show more of the
disk face, but note the result is very sensitive near edge-on — even ≈1 unit
off-plane (≈2° tilt here) makes the upper/lower arcs visibly uneven. Preview
any angle without recompiling: `./renderer 800 600 <camX> <camY> <camZ>
out.ppm`.

The central pixel shoots straight at the BH → captured (black shadow).
Off-center rays curve around the BH; highly bent ones can complete
partial orbits and hit the disk from the far side (Einstein ring, photon
ring visible as a bright arc near the shadow boundary).
