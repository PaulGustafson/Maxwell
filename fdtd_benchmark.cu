#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Constants
constexpr float C0 = 299792458.0f;
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-9f;
constexpr float C0_p_dt = C0 * dt;

// Utility function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernels (from the original code)
__device__ float update_curl_ex(int nx, int cell_x, int cell_y, int cell_id, float dy, const float* ez) {
    const int top_neighbor_id = nx * (cell_y + 1) + cell_x;
    return (ez[top_neighbor_id] - ez[cell_id]) / dy;
}

__device__ float update_curl_ey(int nx, int cell_x, int cell_y, int cell_id, float dx, const float* ez) {
    const int right_neighbor_id = cell_x == nx - 1 ? cell_y * nx + 0 : cell_id + 1;
    return -(ez[right_neighbor_id] - ez[cell_id]) / dx;
}

__device__ void update_h(int nx, int cell_id, float dx, float dy, const float* ez, const float* mh, float* hx, float* hy) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float cex = update_curl_ex(nx, cell_x, cell_y, cell_id, dy, ez);
    const float cey = update_curl_ey(nx, cell_x, cell_y, cell_id, dx, ez);
    hx[cell_id] -= mh[cell_id] * cex;
    hy[cell_id] -= mh[cell_id] * cey;
}

__device__ float update_curl_h(int nx, int cell_id, int cell_x, int cell_y, float dx, float dy, const float* hx, const float* hy) {
    const int left_neighbor_id = cell_x == 0 ? cell_y * nx + nx - 1 : cell_id - 1;
    const int bottom_neighbor_id = nx * (cell_y - 1) + cell_x;
    return (hy[cell_id] - hy[left_neighbor_id]) / dx - (hx[cell_id] - hx[bottom_neighbor_id]) / dy;
}

__device__ float gaussian_pulse(float t, float t_0, float tau) {
    return __expf(-(((t - t_0) / tau) * ((t - t_0) / tau)));
}

__device__ float calculate_source(float t, float frequency) {
    const float tau = 0.5f / frequency;
    const float t_0 = 6.0f * tau;
    return gaussian_pulse(t, t_0, tau);
}

__device__ void update_e(int nx, int cell_id, int own_in_process_begin, int source_position,
                         float t, float dx, float dy, float C0_p_dt,
                         float* ez, float* dz, const float* er, const float* hx, const float* hy) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float chz = update_curl_h(nx, cell_id, cell_x, cell_y, dx, dy, hx, hy);
    dz[cell_id] += C0_p_dt * chz;
    if ((own_in_process_begin + cell_y) * nx + cell_x == source_position)
        dz[cell_id] += calculate_source(t, 5E+7);
    ez[cell_id] = dz[cell_id] / er[cell_id];
}

// Kernel to initialize the arrays
__global__ void init_fields(int nx, int ny, float* ez, float* dz, float* hx, float* hy, float* er, float* mh) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < nx * ny) {
        ez[cell_id] = 0.0f;
        dz[cell_id] = 0.0f;
        hx[cell_id] = 0.0f;
        hy[cell_id] = 0.0f;
        er[cell_id] = 1.0f;  // Free space
        mh[cell_id] = 1.0f;
    }
}

// Kernel for running FDTD updates
__global__ void fdtd_update(int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t,
                            float* ez, float* dz, float* hx, float* hy, const float* er, const float* mh) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < nx * ny) {
        update_h(nx, cell_id, dx, dy, ez, mh, hx, hy);
        update_e(nx, cell_id, 0, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
    }
}

// Structure to hold benchmark results
struct BenchmarkResult {
    double totalTime;
    double kernelTime;
    double memoryBandwidth;
    double computeEfficiency;
};

// Function to measure kernel execution time and performance
BenchmarkResult measureKernelPerformance(void (*kernel)(int, int, float, float, float, int, float, float*, float*, float*, float*, const float*, const float*),
                                         int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t,
                                         float* ez, float* dz, float* hx, float* hy, const float* er, const float* mh) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, C0_p_dt, source_position, t, ez, dz, hx, hy, er, mh);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    kernel<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, C0_p_dt, source_position, t, ez, dz, hx, hy, er, mh);
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
    int steps = 1000;

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
            BenchmarkResult stepResult = measureKernelPerformance(fdtd_update, nx, ny, dx, dy, C0_p_dt, source_position, t, ez, dz, hx, hy, er, mh);
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