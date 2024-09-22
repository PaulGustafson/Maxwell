#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

constexpr float C0 = 299792458.0f; 
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-9;
constexpr float C0_p_dt = C0 * dt;

__device__ float update_curl_ex (int nx, int cell_x, int cell_y, int cell_id, float dy, const float * ez) {
    const int top_neighbor_id = nx * (cell_y + 1) + cell_x;
    return (ez[top_neighbor_id] - ez[cell_id]) / dy;
}

__device__ float update_curl_ey (int nx, int cell_x, int cell_y, int cell_id, float dx, const float * ez) {
    const int right_neighbor_id = cell_x == nx - 1 ? cell_y * nx + 0 : cell_id + 1;
    return -(ez[right_neighbor_id] - ez[cell_id]) / dx;
}

__device__ void update_h (int nx, int cell_id, float dx, float dy, const float *ez, const float *mh, float *hx, float *hy) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float cex = update_curl_ex(nx, cell_x, cell_y, cell_id, dy, ez);
    const float cey = update_curl_ey(nx, cell_x, cell_y, cell_id, dx, ez);
    hx[cell_id] -= mh[cell_id] * cex;
    hy[cell_id] -= mh[cell_id] * cey;
}

__device__ float update_curl_h (int nx, int cell_id, int cell_x, int cell_y, float dx, float dy, const float *hx, const float *hy) {
    const int left_neighbor_id = cell_x == 0 ? cell_y * nx + nx - 1 : cell_id - 1;
    const int bottom_neighbor_id = nx * (cell_y - 1) + cell_x;
    return (hy[cell_id] - hy[left_neighbor_id]) / dx - (hx[cell_id] - hx[bottom_neighbor_id]) / dy;
}

__device__ float gaussian_pulse (float t, float t_0, float tau) {
    return __expf(-(((t - t_0) / tau) * (t - t_0) / tau));
}

__device__ float calculate_source(float t, float frequency) {
    const float tau = 0.5f / frequency;
    const float t_0 = 6.0f * tau;
    return gaussian_pulse(t, t_0, tau);
}

__device__ void update_e(int nx, int cell_id, int own_in_process_begin, int source_position, float t, float dx, float dy, float C0_p_dt, float *ez, float *dz, const float *er, const float *hx, const float *hy) {
    const int cell_x = cell_id % nx;
    const int cell_y = cell_id / nx;
    const float chz = update_curl_h(nx, cell_id, cell_x, cell_y, dx, dy, hx, hy);
    dz[cell_id] += C0_p_dt * chz;
    if ((own_in_process_begin + cell_y) * nx + cell_x == source_position)
        dz[cell_id] += calculate_source(t, 5E+7);
    ez[cell_id] = dz[cell_id] / er[cell_id];
}

__global__ void init_fields(int nx, int ny, float *ez, float *dz, float *hx, float *hy, float *er, float *mh) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < nx * ny) {
        ez[cell_id] = 0.0f;
        dz[cell_id] = 0.0f;
        hx[cell_id] = 0.0f;
        hy[cell_id] = 0.0f;
        er[cell_id] = 1.0f;
        mh[cell_id] = 1.0f;
    }
}

__global__ void fdtd_update(int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t, float *ez, float *dz, float *hx, float *hy, const float *er, const float *mh) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < nx * ny) {
        update_h(nx, cell_id, dx, dy, ez, mh, hx, hy);
        update_e(nx, cell_id, 0, source_position, t, dx, dy, C0_p_dt, ez, dz, er, hx, hy);
    }
}

void check_cuda_error(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " nx ny steps inc" << std::endl;
        return 1;
    }

    int nx = 0, ny = 0, steps = 0, increment = 0;
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--grid_size_x") nx = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--grid_size_y") ny = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--steps") steps = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--increment") increment = std::atoi(argv[i + 1]);
    }

    int source_position = (nx / 2) * nx + (ny / 2);

    int num_devices = 2;
    check_cuda_error(cudaSetDevice(0), "Unable to set device 0");
    float *ez_0, *dz_0, *hx_0, *hy_0, *er_0, *mh_0;
    check_cuda_error(cudaMalloc(&ez_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating ez_0");
    check_cuda_error(cudaMalloc(&dz_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating dz_0");
    check_cuda_error(cudaMalloc(&hx_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating hx_0");
    check_cuda_error(cudaMalloc(&hy_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating hy_0");
    check_cuda_error(cudaMalloc(&er_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating er_0");
    check_cuda_error(cudaMalloc(&mh_0, (nx * ny / 2 + 1) * sizeof(float)), "Allocating mh_0");

    check_cuda_error(cudaSetDevice(1), "Unable to set device 1");
    float *ez_1, *dz_1, *hx_1, *hy_1, *er_1, *mh_1;
    check_cuda_error(cudaMalloc(&ez_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating ez_1");
    check_cuda_error(cudaMalloc(&dz_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating dz_1");
    check_cuda_error(cudaMalloc(&hx_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating hx_1");
    check_cuda_error(cudaMalloc(&hy_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating hy_1");
    check_cuda_error(cudaMalloc(&er_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating er_1");
    check_cuda_error(cudaMalloc(&mh_1, (nx * ny / 2 + 1) * sizeof(float)), "Allocating mh_1");

    check_cuda_error(cudaSetDevice(0), "Unable to set device 0");
    init_fields<<<(nx * ny / 2 + 256) / 256, 256>>>(nx, ny / 2, ez_0, dz_0, hx_0, hy_0, er_0, mh_0);
    cudaDeviceSynchronize();

    check_cuda_error(cudaSetDevice(1), "Unable to set device 1");
    init_fields<<<(nx * ny / 2 + 256) / 256, 256>>>(nx, ny / 2, ez_1, dz_1, hx_1, hy_1, er_1, mh_1);
    cudaDeviceSynchronize();

    // Time-stepping loop
    for (int step = 0; step < steps; ++step) {
        float t = step * dt;

        // GPU 0 update
        check_cuda_error(cudaSetDevice(0), "Unable to set device 0");
        fdtd_update<<<(nx * (ny / 2) + 256) / 256, 256>>>(nx, ny / 2, dx, dy, C0_p_dt, source_position, t, ez_0, dz_0, hx_0, hy_0, er_0, mh_0);
        
        // GPU 1 update
        check_cuda_error(cudaSetDevice(1), "Unable to set device 1");
        fdtd_update<<<(nx * (ny / 2) + 256) / 256, 256>>>(nx, ny / 2, dx, dy, C0_p_dt, source_position - (nx * (ny / 2)), t, ez_1, dz_1, hx_1, hy_1, er_1, mh_1);
        
        // Synchronize updates
        cudaDeviceSynchronize();

        // // Exchange boundary data (between GPU 0 and GPU 1)
        // // Copy the last row from GPU 0 to GPU 1 and the first row from GPU 1 to GPU 0
        float *boundary_row_0 = (float*)malloc(nx * sizeof(float));
        float *boundary_row_1 = (float*)malloc(nx * sizeof(float));
        
        // Copy last row of ez_0 (the boundary) to host
        check_cuda_error(cudaMemcpy(boundary_row_0, &ez_0[(ny / 2 - 1) * nx], nx * sizeof(float), cudaMemcpyDeviceToHost), "Copy boundary row 0 to host");
        
        // Copy first row of ez_1 (the boundary) to host
        check_cuda_error(cudaSetDevice(1), "Unable to set device 1");
        check_cuda_error(cudaMemcpy(boundary_row_1, &ez_1[0], nx * sizeof(float), cudaMemcpyDeviceToHost), "Copy boundary row 1 to host");

        for (int i = 0; i < nx; i++) {
            boundary_row_1[i] += boundary_row_0[i];
        }

        // Copy to GPU 1
        check_cuda_error(cudaSetDevice(1), "Unable to set device 1");
        check_cuda_error(cudaMemcpy(&ez_1[0], boundary_row_1, nx * sizeof(float), cudaMemcpyHostToDevice), "Copy boundary row 0 to GPU 1");

        // Copy to GPU 0
        check_cuda_error(cudaSetDevice(0), "Unable to set device 0");
        check_cuda_error(cudaMemcpy(&ez_0[(ny / 2 - 1) * nx], boundary_row_1, nx * sizeof(float), cudaMemcpyHostToDevice), "Copy boundary row 1 to GPU 0");

        free(boundary_row_0);
        free(boundary_row_1);

        // Optionally print or log values of the field at the center of the grid
        if (step % increment == 0) {
            float ez_center_0, ez_center_1;
            check_cuda_error(cudaMemcpy(&ez_center_0, &ez_0[source_position - (ny / 2) * nx], sizeof(float), cudaMemcpyDeviceToHost), "Copy ez_center_0");
            check_cuda_error(cudaMemcpy(&ez_center_1, &ez_1[source_position - (ny / 2) * nx], sizeof(float), cudaMemcpyDeviceToHost), "Copy ez_center_1");
            std::cout << "Step " << step << ", t = " << t << ", Ez at center GPU 0: " << ez_center_0 << ", Ez at center GPU 1: " << ez_center_1 << std::endl;

            float *h_ez_0 = (float*)malloc(nx * (ny / 2) * sizeof(float));  // Host-side copy for GPU 0
            float *h_ez_1 = (float*)malloc(nx * (ny / 2) * sizeof(float));  // Host-side copy for GPU 1

            // Copy data from both GPUs
            check_cuda_error(cudaMemcpy(h_ez_0, ez_0, nx * (ny / 2) * sizeof(float), cudaMemcpyDeviceToHost), "Copy ez_0 to host");
            check_cuda_error(cudaMemcpy(h_ez_1, ez_1, nx * (ny / 2) * sizeof(float), cudaMemcpyDeviceToHost), "Copy ez_1 to host");

            // Print to file
            std::ofstream file("data/ez_step_" + std::to_string(step) + ".txt");
            if (file.is_open()) {
                // Write data from GPU 0
                for (int y = 0; y < ny / 2; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        file << h_ez_0[y * nx + x] << " ";
                    }
                    file << "\n";
                }
                
                // Write data from GPU 1
                for (int y = 0; y < ny / 2; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        file << h_ez_1[y * nx + x] << " ";
                    }
                    file << "\n";
                }
                
                file.close();
            } else {
                std::cerr << "Unable to open file for writing" << std::endl;
            }

            // Free host memory
            free(h_ez_0);
            free(h_ez_1);
        }

    }

    // Free memory
    cudaFree(ez_0);
    cudaFree(ez_1);
    cudaFree(dz_0);
    cudaFree(dz_1);
    cudaFree(hx_0);
    cudaFree(hx_1);
    cudaFree(hy_0);
    cudaFree(hy_1);
    cudaFree(er_0);
    cudaFree(er_1);
    cudaFree(mh_0);
    cudaFree(mh_1);

    return 0;
}