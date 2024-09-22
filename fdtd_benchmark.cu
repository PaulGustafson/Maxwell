#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include "main_correct.h" // Include the header file

// Utility function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Structure to hold benchmark results
struct BenchmarkResult {
    double totalTime;
    double kernelTime;
    double memoryBandwidth;
    double computeEfficiency;
};

// Function to measure kernel execution time and performance
BenchmarkResult measureKernelPerformance(int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t,
                                         float* ez, float* dz, float* hx, float* hy, const float* er, const float* mh) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total = nx*ny;
    // Warm-up run
    update_h_kernel<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, ez, mh, hx, hy, total);
    update_e_kernel<<<(nx * ny + 255) / 256, 256>>>(nx, source_position, t, dx,dy, C0_p_dt, ez, dz, er, hx, hy, total);

    // Timed run
    cudaEventRecord(start);
    update_h_kernel<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, ez, mh, hx, hy, total);
    update_e_kernel<<<(nx * ny + 255) / 256, 256>>>(nx, source_position, t, dx,dy, C0_p_dt, ez, dz, er, hx, hy, total);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate memory bandwidth
    size_t bytesRead = nx * ny * sizeof(float) * 5;  // ez, hx, hy, er, mh
    size_t bytesWritten = nx * ny * sizeof(float) * 3;  // ez, hx, hy
    double gigaBytesPerSecond = (bytesRead + bytesWritten) / (milliseconds * 1e-3) / 1e9;

    // Calculate compute efficiency
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    double flops = nx * ny * 26;  // Approximate FLOPs per kernel execution
    double teraFlopsPerSecond = flops / (milliseconds * 1e-3) / 1e12;
    double peakTeraFlops = 2.0 * deviceProp.clockRate * deviceProp.multiProcessorCount * 32 / 1e6;  // Assuming 32 CUDA cores per SM
    double computeEfficiency = teraFlopsPerSecond / peakTeraFlops * 100.0;

    return {milliseconds / 1000.0, milliseconds / 1000.0, gigaBytesPerSecond, computeEfficiency};
}

// Function to run benchmarks
void runBenchmarks() {
    std::vector<std::pair<int, int>> gridSizes = {{100, 100}, {500, 500}, {1000, 1000}, {2000, 2000}};
    int steps = 50;

    for (const auto& size : gridSizes) {
        int nx = size.first;
        int ny = size.second;
        int source_position = (nx / 2) * nx + (ny / 2);

        // Allocate memory on device
        float *ez, *dz, *hx, *hy, *er, *mh;
        cudaMalloc(&ez, nx * ny * sizeof(float));
        cudaMalloc(&dz, nx * ny * sizeof(float));
        cudaMalloc(&hx, nx * ny * sizeof(float));
        cudaMalloc(&hy, nx * ny * sizeof(float));
        cudaMalloc(&er, nx * ny * sizeof(float));
        cudaMalloc(&mh, nx * ny * sizeof(float));

        // Initialize fields on the device
        init_fields<<<(nx * ny + 255) / 256, 256>>>(nx, ny, ez, dz, hx, hy, er, mh);
        cudaDeviceSynchronize();

        // Measure total execution time
        auto start = std::chrono::high_resolution_clock::now();

        BenchmarkResult totalResult = {0.0, 0.0, 0.0, 0.0};
        for (int step = 0; step < steps; ++step) {
            float t = step * dt;
            BenchmarkResult stepResult = measureKernelPerformance(nx, ny, dx, dy, C0_p_dt, source_position, t, ez, dz, hx, hy, er, mh);
            totalResult.kernelTime += stepResult.kernelTime;
            totalResult.memoryBandwidth += stepResult.memoryBandwidth;
            totalResult.computeEfficiency += stepResult.computeEfficiency;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        totalResult.totalTime = diff.count();

        // Calculate average metrics
        totalResult.memoryBandwidth /= steps;
        totalResult.computeEfficiency /= steps;

        // Print performance metrics
        std::cout << "Grid size: " << nx << "x" << ny << std::endl;
        std::cout << "Total time: " << totalResult.totalTime << " seconds" << std::endl;
        std::cout << "Total kernel time: " << totalResult.kernelTime << " seconds" << std::endl;
        std::cout << "Average memory bandwidth: " << totalResult.memoryBandwidth << " GB/s" << std::endl;
        std::cout << "Average compute efficiency: " << totalResult.computeEfficiency << "%" << std::endl;
        std::cout << std::endl;

        // Free allocated memory
        cudaFree(ez);
        cudaFree(dz);
        cudaFree(hx);
        cudaFree(hy);
        cudaFree(er);
        cudaFree(mh);
    }
}

int main() {
    runBenchmarks();
    return 0;
}