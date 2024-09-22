#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

constexpr float C0 = 299792458.0f; 
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-9f;
constexpr float C0_p_dt = C0 * dt;

__device__ float update_curl_ex(int nx, int cell_x, int cell_y, int cell_id, float dy, const float * ez) {
  const int top_neighbor_id = nx * (cell_y + 1) + cell_x;
  return (ez[top_neighbor_id] - ez[cell_id]) / dy;
}

__device__ float update_curl_ey(int nx, int cell_x, int cell_y, int cell_id, float dx, const float * ez) {
  const int right_neighbor_id = cell_x == nx - 1 ? cell_y * nx + 0 : cell_id + 1;
  return -(ez[right_neighbor_id] - ez[cell_id]) / dx;
}

__device__ float update_curl_h(int nx, int cell_id, int cell_x, int cell_y, float dx, float dy,
                               const float *hx, const float *hy) {
  const int left_neighbor_id = cell_x == 0 ? cell_y * nx + nx - 1 : cell_id - 1;
  const int bottom_neighbor_id = nx * (cell_y - 1) + cell_x;
  return (hy[cell_id] - hy[left_neighbor_id]) / dx
       - (hx[cell_id] - hx[bottom_neighbor_id]) / dy;
}

__device__ float gaussian_pulse(float t, float t_0, float tau) {
  return __expf(-(((t - t_0) / tau) * (t - t_0) / tau));
}

__device__ float calculate_source(float t, float frequency) {
  const float tau = 0.5f / frequency;
  const float t_0 = 6.0f * tau;
  return gaussian_pulse(t, t_0, tau);
}

__global__ void update_h_kernel(int nx, int ny, float dx, float dy,
                                const float *ez, const float *mh,
                                float *hx, float *hy, int total) {
  const int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id < total) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float cex = update_curl_ex(nx, cell_x, cell_y, cell_id, dy, ez);
    const float cey = update_curl_ey(nx, cell_x, cell_y, cell_id, dx, ez);

    const float mh_id = mh[cell_id];
    hx[cell_id] -= mh_id * cex;
    hy[cell_id] -= mh_id * cey;
  }
}

__global__ void update_e_kernel(int nx, int source_position,
                                float t, float dx, float dy, float C0_p_dt,
                                float *ez, float *dz,
                                const float *er, const float *hx, const float *hy, int total) {
  const int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id < total) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float chz = update_curl_h(nx, cell_id, cell_x, cell_y, dx, dy, hx, hy);
    dz[cell_id] += C0_p_dt * chz;
    if (cell_id == source_position)
      dz[cell_id] += calculate_source(t, 5E+7);
    ez[cell_id] = dz[cell_id] / er[cell_id];
  }
}

__global__ void init_fields(int nx, int ny, float *ez, float *dz, float *hx, float *hy, float *er, float *mh) {
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

int main_old(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " nx ny steps inc" << std::endl;
        return 1;
    }

    int nx = std::atoi(argv[1]);
    int ny = std::atoi(argv[2]);
    int steps = std::atoi(argv[3]);
    int increment = std::atoi(argv[4]);

    int source_position = (nx / 2) * nx + (ny / 2);  // Center of the grid
    
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

    int total = nx * ny;
    
    // Time-stepping loop
    for (int step = 0; step < steps; ++step) {
        float t = step * dt;

        update_h_kernel<<<(total + 255) / 256, 256>>>(
            nx, ny, dx, dy, ez, mh, hx, hy, total);
        
        update_e_kernel<<<(total + 255) / 256, 256>>>(
            nx, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy, total);

        cudaDeviceSynchronize();

        // Output and file writing code
        if (step % increment == 0) {
            float ez_center;
            cudaMemcpy(&ez_center, &ez[source_position], sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Step " << step << ", t = " << t << ", Ez at center: " << ez_center << std::endl;

            float *h_ez = (float*)malloc(nx * ny * sizeof(float));
            cudaMemcpy(h_ez, ez, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
            std::ofstream file("data/ez_step_" + std::to_string(step) + ".txt");
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    file << h_ez[y * nx + x] << " ";
                }
                file << "\n";
            }
            file.close();
            free(h_ez);
        }
    }

    // Free memory
    cudaFree(ez);
    cudaFree(dz);
    cudaFree(hx);
    cudaFree(hy);
    cudaFree(er);
    cudaFree(mh);

    return 0;
}