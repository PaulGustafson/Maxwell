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
    std::ofstream file("data/u_step_" + std::to_string(step) + ".txt");
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            file << u[y * nx + x] << " ";
        }
        file << "\n";
    }
    file.close();
}

// void run_fdtd_step(int nx, int ny, float dx, 
//         float dy, float c_wave, float t, float *u, float *u_new) {
//     fdtd_update<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, c_wave, t, u, u_new);
//     cudaDeviceSynchronize();
// }

__global__ void init_fields(int nx, int ny, float *u, float *u_new) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        int index = idx * nx + idy;
        // Initial conditions for u and u_new
        // We are starting with a sin wave based on x and y directions.
        u[index] = sin(idx * dx) * sin(idy * dy);
        u_new[index] = 0.0f;
    }
}
        
    

