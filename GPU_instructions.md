Build:
cmake -S . -B build
cmake --build build -j

Launch the browser GUI:
./build/renderer_gpu

Output:
GPU: NVIDIA RTX A5000  (sm_86, 25.3 GB)
[viewer] serving on http://localhost:8080/
Open the forwarded port in your browser:  http://localhost:8080/
(Ctrl-C to stop.)

 ./build/renderer_gpu 1280 720          # change resolution
./build/renderer_gpu --port 8081       # different port
./build/renderer_gpu --skybox data/skybox.jpg   # use a custom skybox image
./build/renderer_gpu --once frame.jpg  # render one frame to a file, no server
./build/renderer_gpu --benchmark       # Test CPU vs GPU output (saved to CSV), no GUI

GUI controls:
drag = orbit, scroll = zoom, sliders for disk radii, FOV, RK4 step, exposure, bloom intensity, fBm, a Reinhard/ACES selector, resolution-scale buttons (0.25×–2×), bloom/Doppler toggles, reset, screenshot to PNG, and FPS/performance metrics

