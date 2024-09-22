#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "waves.cuh"

void allocate_memory(int nx, int ny, float **u, float **u_new) {
    cudaMalloc(u, nx * ny * sizeof(float));
    cudaMalloc(u_new, nx * ny * sizeof(float));
}

void initialize_fields(int nx, int ny, float *u, float *u_new) {
    init_fields<<<(nx * ny + 255) / 256, 256>>>(nx, ny, u, u_new);
    cudaDeviceSynchronize();
}

void free_memory(float *u, float *u_new) {
    cudaFree(u);
    cudaFree(u_new);
}

void write_state(int nx, int ny, float *u, int step) {
    float *h_u = new float[nx * ny];  // Allocate host memory
    cudaMemcpy(h_u, u, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);  // Copy data from device to host

    std::ofstream file("data/u_step_" + std::to_string(step) + ".txt");
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            file << h_u[y * nx + x] << " ";
        }
        file << "\n";
    }
    file.close();
}

__global__ void init_fields(int nx, int ny, float *u, float *u_new) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx && idy < ny) {
        int cell_id = idy * nx + idx;
        // Initial conditions for u and u_new
        // We are starting with a sin wave based on x and y directions.
        u[cell_id] = __sinf(idx * dx) * __sinf(idy * dy);
        u_new[cell_id] = 0.0f;
    }
}
        
    

