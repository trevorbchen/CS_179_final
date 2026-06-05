cmake_minimum_required(VERSION 3.18)

# =============================================================================
# Schwarzschild Black Hole Renderer — CS 179 GPU final project
#
# Two executables:
#   renderer_cpu : header-only CPU reference (verification baseline)
#   renderer_gpu : real-time CUDA renderer with a browser (and optional GLFW) GUI
#
# The same CMakeLists drives every step of the port: GPU/test targets are
# guarded with `if(EXISTS ...)` so the project configures and builds the CPU
# baseline even before the CUDA sources exist.
# =============================================================================
project(black_hole_renderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# A5000 = Ampere, compute capability 8.6.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_GLFW_VIEWER "Build the GLFW+OpenGL+ImGui viewer (for ssh -X / X11)" OFF)

# --- Dependencies ------------------------------------------------------------
find_package(JPEG REQUIRED)            # libjpeg-turbo (already used by the CPU path)
find_package(CUDAToolkit REQUIRED)     # cudart for host-side buffer management
find_package(Threads REQUIRED)         # cpp-httplib server threads

set(THIRD_PARTY ${CMAKE_SOURCE_DIR}/third_party)

# -----------------------------------------------------------------------------
# renderer_cpu — existing header-only CPU baseline (unchanged physics)
# -----------------------------------------------------------------------------
add_executable(renderer_cpu main.cpp)
target_include_directories(renderer_cpu PRIVATE ${CMAKE_SOURCE_DIR} ${JPEG_INCLUDE_DIRS})
target_link_libraries(renderer_cpu PRIVATE ${JPEG_LIBRARIES})
target_compile_options(renderer_cpu PRIVATE -O2)

# -----------------------------------------------------------------------------
# renderer_gpu — real-time CUDA renderer (the main deliverable)
#   main_gpu.cpp           : host entry point / CLI / GUI selection
#   cuda/gravity_kernel.cu : Module 1 (Kevin) — geodesic integration
#   cuda/renderer_kernel.cu: Module 2 (Trevor) — shading
#   cuda/postprocess.cu    : Module 2 (Trevor) — bloom + tone map
#   viewer/http_server.cpp : browser MJPEG/control server (default GUI)
# -----------------------------------------------------------------------------
set(GPU_SOURCES
    main_gpu.cpp
    cuda/gravity_kernel.cu
    cuda/renderer_kernel.cu
    cuda/postprocess.cu
    viewer/http_server.cpp)

set(_gpu_sources_present TRUE)
foreach(src ${GPU_SOURCES})
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${src})
    set(_gpu_sources_present FALSE)
  endif()
endforeach()

if(_gpu_sources_present)
  add_executable(renderer_gpu ${GPU_SOURCES})
  target_include_directories(renderer_gpu PRIVATE
      ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cuda ${THIRD_PARTY} ${JPEG_INCLUDE_DIRS})
  target_link_libraries(renderer_gpu PRIVATE
      ${JPEG_LIBRARIES} CUDA::cudart Threads::Threads)
  set_target_properties(renderer_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # Embed the path to the repo root so the binary can find viewer/index.html
  # and data/skybox.jpg when launched from a build directory.
  target_compile_definitions(renderer_gpu PRIVATE
      PROJECT_SOURCE_DIR_STR="${CMAKE_SOURCE_DIR}")

  if(BUILD_GLFW_VIEWER AND EXISTS ${CMAKE_SOURCE_DIR}/viewer/glfw_viewer.cpp)
    find_package(glfw3 QUIET)
    find_package(OpenGL QUIET)
    if(glfw3_FOUND AND OpenGL_FOUND AND EXISTS ${THIRD_PARTY}/imgui)
      file(GLOB IMGUI_SRC ${THIRD_PARTY}/imgui/*.cpp
                          ${THIRD_PARTY}/imgui/backends/imgui_impl_glfw.cpp
                          ${THIRD_PARTY}/imgui/backends/imgui_impl_opengl3.cpp)
      target_sources(renderer_gpu PRIVATE viewer/glfw_viewer.cpp ${IMGUI_SRC})
      target_include_directories(renderer_gpu PRIVATE
          ${THIRD_PARTY}/imgui ${THIRD_PARTY}/imgui/backends)
      target_link_libraries(renderer_gpu PRIVATE glfw OpenGL::GL CUDA::cuda_driver)
      target_compile_definitions(renderer_gpu PRIVATE HAVE_GLFW_VIEWER=1)
      message(STATUS "GLFW viewer: ENABLED")
    else()
      message(WARNING "BUILD_GLFW_VIEWER=ON but glfw3/OpenGL/third_party/imgui not found; skipping.")
    endif()
  endif()
else()
  message(STATUS "renderer_gpu: CUDA sources not present yet; skipping (CPU baseline still builds).")
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()

add_executable(test_geodesic tests/test_geodesic.cpp)
target_include_directories(test_geodesic PRIVATE ${CMAKE_SOURCE_DIR} ${JPEG_INCLUDE_DIRS})
target_link_libraries(test_geodesic PRIVATE ${JPEG_LIBRARIES})
add_test(NAME geodesic COMMAND test_geodesic)

add_executable(test_shading tests/test_shading.cpp)
target_include_directories(test_shading PRIVATE ${CMAKE_SOURCE_DIR} ${JPEG_INCLUDE_DIRS})
target_link_libraries(test_shading PRIVATE ${JPEG_LIBRARIES})
add_test(NAME shading COMMAND test_shading)

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_cuda_parity.cu AND _gpu_sources_present)
  add_executable(test_cuda_parity
      tests/test_cuda_parity.cu
      cuda/gravity_kernel.cu
      cuda/renderer_kernel.cu
      cuda/postprocess.cu)
  target_include_directories(test_cuda_parity PRIVATE
      ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cuda ${JPEG_INCLUDE_DIRS})
  target_link_libraries(test_cuda_parity PRIVATE ${JPEG_LIBRARIES} CUDA::cudart)
  set_target_properties(test_cuda_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME cuda_parity COMMAND test_cuda_parity)
endif()
