#pragma once
// =============================================================================
// cuda_helpers.h — small shared utilities for the GPU renderer.
//   - CUDA_CHECK / CUDA_CHECK_KERNEL error-checking macros
//   - div_up grid-size helper
// Used by both Module 1 (gravity) and Module 2 (renderer / postprocess).
// =============================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wrap every CUDA runtime call: abort loudly with file:line on failure.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err),             \
                    cudaGetErrorName(_err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Check for launch/async errors after a kernel. Synchronizes in debug-ish use.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
    } while (0)

// Ceiling division — number of blocks needed to cover n threads given block dim.
__host__ __device__ inline int div_up(int n, int d) { return (n + d - 1) / d; }
