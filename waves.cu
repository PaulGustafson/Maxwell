#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "waves.cuh"

void allocate_memory(int nx, int ny, float **u, float **u_new) {
    cudaMalloc(u, nx * ny * sizeof(float));
    cudaMalloc(u_new, nx * ny * sizeof(float));
}

void initialize_fields(int nx, int ny, float *u_old, float *u_curr, float *u_new) {
    init_fields<<<(nx * ny + 255) / 256, 256>>>(nx, ny, u_old, u_curr, u_new);
    cudaDeviceSynchronize();
}

void free_memory(float *u_old, float *u_curr, float *u_new) {
    cudaFree(u_old);
    cudaFree(u_curr);
    cudaFree(u_new);
}

void write_state(int nx, int ny, float *u_curr, int step) {
    float *h_u = new float[nx * ny];  // Allocate host memory
    cudaMemcpy(h_u, u_curr, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);  // Copy data from device to host

    std::ofstream file("data/u_step_" + std::to_string(step) + ".txt");
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            file << h_u[y * nx + x] << " ";
        }
        file << "\n";
    }
    file.close();
}

__global__ void init_fields(int nx, int ny, float *u_old, float *u_curr, float *u_new) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_id < nx * ny) {
        float x = cell_id % nx;
        float y = cell_id / nx;
        // Initial conditions for u and u_new
        // We are starting with a sin wave based on x and y directions.
        u_old[cell_id] = __sinf(2 * M_PI * x / nx) * __sinf(2 * M_PI * y / ny);
        u_curr[cell_id] = 0.0f;
        u_new[cell_id] = 0.0f;
    }
}

void run_fdtd_step(int nx, int ny, float dx, float dy, float c_p_dt, float t,
                   float *u_old, float *u_curr, float *u_new) {
    fdtd_update<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, c_p_dt, t, u_old, u_curr, u_new);
    cudaDeviceSynchronize();
}

__global__ void fdtd_update(int nx, int ny, float dx, float dy, float c_p_dt,float *u_old, float *u_curr, float *u_new) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.y;

    float alpha_x = c_p_dt / dx;
    float alpha_y = c_p_dt / dy;

    int x = cell_id % nx;
    int y = cell_id / nx;

}
    

