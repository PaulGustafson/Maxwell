#ifndef FDTD_KERNELS_H
#define FDTD_KERNELS_H

#include <cuda_runtime.h>

// Constants
constexpr float C0 = 299792458.0f;
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-9f;
constexpr float C0_p_dt = C0 * dt;

// CUDA device functions
__device__ float update_curl_ex(int nx, int cell_x, int cell_y, int cell_id, float dy, const float* ez);
__device__ float update_curl_ey(int nx, int cell_x, int cell_y, int cell_id, float dx, const float* ez);
__device__ void update_h(int nx, int cell_id, float dx, float dy, const float* ez, const float* mh, float* hx, float* hy);
__device__ float update_curl_h(int nx, int cell_id, int cell_x, int cell_y, float dx, float dy, const float* hx, const float* hy);
__device__ float gaussian_pulse(float t, float t_0, float tau);
__device__ float calculate_source(float t, float frequency);
__device__ void update_e(int nx, int cell_id, int own_in_process_begin, int source_position,
                         float t, float dx, float dy, float C0_p_dt,
                         float* ez, float* dz, const float* er, const float* hx, const float* hy);

// CUDA kernels
__global__ void init_fields(int nx, int ny, float* ez, float* dz, float* hx, float* hy, float* er, float* mh);
__global__ void fdtd_update(int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t,
                            float* ez, float* dz, float* hx, float* hy, const float* er, const float* mh);

#endif // FDTD_KERNELS_H