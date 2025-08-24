/*
 * CUDA Configuration Implementation for KeyHunt-Cuda-2
 * This file implements CUDA configuration functions optimized for RTX 4090 on Ubuntu 22.04
 */

#include "CudaConfig.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int get_gpu_count()
{
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    return device_count;
}

int get_gpu_compute_capability(int gpu_id, int* major, int* minor)
{
    if (gpu_id < 0) {
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, gpu_id);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for GPU %d: %s\n", gpu_id, cudaGetErrorString(error));
        return -1;
    }
    
    if (major) *major = prop.major;
    if (minor) *minor = prop.minor;
    
    return 0;
}

size_t get_gpu_memory_size(int gpu_id)
{
    if (gpu_id < 0) {
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, gpu_id);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for GPU %d: %s\n", gpu_id, cudaGetErrorString(error));
        return 0;
    }
    
    return prop.totalGlobalMem;
}

const char* get_gpu_name(int gpu_id)
{
    if (gpu_id < 0) {
        return NULL;
    }
    
    static char name[256];
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, gpu_id);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for GPU %d: %s\n", gpu_id, cudaGetErrorString(error));
        return NULL;
    }
    
    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}

void print_gpu_info()
{
    int device_count = get_gpu_count();
    if (device_count <= 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    
    printf("Found %d CUDA device(s):\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Shared Memory Per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("\n");
    }
}

int set_gpu_device(int gpu_id)
{
    if (gpu_id < 0) {
        return -1;
    }
    
    int device_count = get_gpu_count();
    if (gpu_id >= device_count) {
        fprintf(stderr, "Invalid GPU ID %d. Only %d GPU(s) available.\n", gpu_id, device_count);
        return -1;
    }
    
    cudaError_t error = cudaSetDevice(gpu_id);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device to %d: %s\n", gpu_id, cudaGetErrorString(error));
        return -1;
    }
    
    // Set device flags for optimal performance
    error = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to set device flags for GPU %d: %s\n", gpu_id, cudaGetErrorString(error));
        // Continue anyway as this might not be critical
    }
    
    return 0;
}

int initialize_cuda_context(int gpu_id)
{
    if (set_gpu_device(gpu_id) != 0) {
        return -1;
    }
    
    // Initialize CUDA context by allocating a small amount of memory
    void* dummy_ptr;
    cudaError_t error = cudaMalloc(&dummy_ptr, 1);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to initialize CUDA context on GPU %d: %s\n", gpu_id, cudaGetErrorString(error));
        return -1;
    }
    
    cudaFree(dummy_ptr);
    return 0;
}