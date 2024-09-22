#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "waves.cuh"

__device__ float source_term(float t) {
    float freq = 10.0f; // Adjust frequency as needed
    return sinf(2.0f * M_PI * freq * t);
}

__global__ void initialize_fields_kernel(int nx, int ny, float *u_old, float *u_curr, float *u_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        int idx = i + j * nx;
        
        // Set initial conditions (e.g., Gaussian pulse)
        float x = i - nx / 2.0f;
        float y = j - ny / 2.0f;
        float r2 = x*x + y*y;
        float sigma = 10.0f;
        
        u_old[idx] = expf(-r2 / (2.0f * sigma * sigma));
        u_curr[idx] = u_old[idx];
        u_new[idx] = 0.0f;
    }
}

void allocate_memory(int nx, int ny, float **u_old, float **u_curr, float **u_new) {
    cudaMalloc(u_old, nx * ny * sizeof(float));
    cudaMalloc(u_curr, nx * ny * sizeof(float));
    cudaMalloc(u_new, nx * ny * sizeof(float));
}

void initialize_fields(int nx, int ny, float *u_old, float *u_curr, float *u_new) {
    dim3 block_size(16, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
    
    initialize_fields_kernel<<<grid_size, block_size>>>(nx, ny, u_old, u_curr, u_new);
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

__global__ void fdtd_update(int nx, int ny, float dx, float dy, float c_p_dt, float t, float *u_old, float *u_curr, float *u_new, int source_position) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < nx * ny) {
        int i = cell_id % nx;
        int j = cell_id / nx;

        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            // Implement absorbing boundary condition
            u_new[cell_id] = u_curr[cell_id];
        } else {
            float d2x = (u_curr[cell_id + 1] - 2.0f * u_curr[cell_id] + u_curr[cell_id - 1]) / (dx * dx);
            float d2y = (u_curr[cell_id + nx] - 2.0f * u_curr[cell_id] + u_curr[cell_id - nx]) / (dy * dy);
            
            // Update wave equation
            u_new[cell_id] = 2.0f * u_curr[cell_id] - u_old[cell_id] + c_p_dt * c_p_dt * (d2x + d2y);

            // Apply source term
            if (cell_id == source_position) {
                u_new[cell_id] += c_p_dt * c_p_dt * source_term(t);
            }
        }
    }
}

void run_fdtd_step(int nx, int ny, float dx, float dy, float c_p_dt, float t, int source_position,
                   float *u_old, float *u_curr, float *u_new) {
    // The function parameters should be float *u_old, *u_curr, *u_new, not float **
    // So we need to remove the dereferencing (*) when passing to the kernel
    fdtd_update<<<(nx * ny + 255) / 256, 256>>>(nx, ny, dx, dy, c_p_dt, t, u_old, u_curr, u_new, source_position);
    cudaDeviceSynchronize();

    // Debug print before cycling pointers
    static float last_print_time = -10.0f;  // Initialize to -10 to ensure first print
    if (t - last_print_time >= 10.0f) {
        float old_value, curr_value, new_value;
        cudaMemcpy(&old_value, u_old + 275, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&curr_value, u_curr + 275, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&new_value, u_new + 275, sizeof(float), cudaMemcpyDeviceToHost);
        printf("After step %f: u_old = %f, u_curr = %f, u_new = %f\n", t, old_value, curr_value, new_value);
        last_print_time = t;
    }

    // Cycle the pointers
    float *temp = u_old;
    u_old = u_curr;
    u_curr = u_new;
    u_new = temp;
}

