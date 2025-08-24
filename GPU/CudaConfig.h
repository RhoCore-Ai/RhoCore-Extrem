/*
 * CUDA Configuration Header for KeyHunt-Cuda-2
 * This file contains CUDA configuration settings optimized for RTX 4090 on Ubuntu 22.04
 */

#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

#include <cstddef>  // Add this for size_t

// CUDA Version Check
#if CUDART_VERSION >= 12000
#define CUDA_12_OR_LATER 1
#else
#define CUDA_12_OR_LATER 0
#endif

// GPU Architecture Settings for RTX 4090 (Compute Capability 8.9)
#define TARGET_CUDA_ARCH 89
#define TARGET_CUDA_MAJOR 8
#define TARGET_CUDA_MINOR 9

// Memory Settings for RTX 4090 (24GB VRAM per GPU)
#define DEFAULT_GPU_MEMORY_ALLOC_SIZE (1024 * 1024 * 1024) // 1GB initial allocation
#define MAX_GPU_MEMORY_USAGE 0.85f // Use up to 85% of available GPU memory

// Thread Block Settings optimized for RTX 4090
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_GRID_SIZE_MULTIPLE 8

// Shared Memory Settings
#define SHARED_MEMORY_BANKS 32
#define SHARED_MEMORY_BANK_WIDTH 4

// Warp Size (standard for all modern GPUs)
#define WARP_SIZE 32

// PTX Architecture Settings
#define PTX_ARCH_SM89 89
#define PTX_ARCH_SM86 86
#define PTX_ARCH_SM80 80

// Compiler Flags for RTX 4090
#define CUDA_COMPILER_FLAGS \
    "--ptxas-options=-v " \
    "--maxrregcount=255 " \
    "--use_fast_math " \
    "-lineinfo "

// Optimization Flags
#define CUDA_OPTIMIZATION_FLAGS \
    "-O3 " \
    "--extra-device-vectorization " \
    "--relocatable-device-code=false "

// Memory Access Patterns
#define COALESCED_ACCESS_ALIGNMENT 128
#define MEMORY_ACCESS_STRIDE 4

// Error Checking Macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device Query Functions
#ifdef __cplusplus
extern "C" {
#endif

int get_gpu_count();
int get_gpu_compute_capability(int gpu_id, int* major, int* minor);
size_t get_gpu_memory_size(int gpu_id);
const char* get_gpu_name(int gpu_id);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CONFIG_H