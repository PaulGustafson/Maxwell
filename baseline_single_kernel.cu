#include "baseline_single_kernel.hh"
#include "gpu_utils.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ float gaussian_pulse (float t, float t_0, float tau) {
  return __expf (-(((t - t_0) / tau) * (t - t_0) / tau));
}

__device__ float calculate_source (float t, float frequency) {
  const float tau = 0.5f / frequency;
  const float t_0 = 6.0f * tau;
  return gaussian_pulse (t, t_0, tau);
}

__global__ void fdtd_cg (
    // param to scale problem size
    const int nx, const int ny, const int n_cells, 
    float * __restrict__ ez, 
    const float * __restrict__ mh,
    // ez and mh update hx and hy
    float * __restrict__ hx, float * __restrict__ hy,
    // hx and hy update dz
    float * __restrict__ dz,
    // dz and er update ez
    const float * __restrict__ er)
{
   const int tid = blockIdx.x * blockDim.x + threadIdx.x;

   cg::grid_group grid = cg::this_grid ();

   float t = initial_t;

    for (int cell_id = tid; cell_id < n_cells; cell_id += blockDim.x * gridDim.x)
    {
        // update H
        const float cez = ez[cell_id];
        const int cell_x = cell_id % nx;
        const int cell_y = cell_id / nx;

        const int top_neighbor_id = nx * (cell_y == ny - 1 ? 0 : cell_y + 1) + cell_x;
        const int right_neighbor_id = cell_x == nx - 1 ? cell_y * nx + 0 : cell_id + 1;

        const float cex =  (ez[top_neighbor_id]   - cez) / dy;
        const float cey = -(ez[right_neighbor_id] - cez) / dx;

        hx[cell_id] -= mh[cell_id] * cex;
        hy[cell_id] -= mh[cell_id] * cey;
    }
    
    grid.sync ();

    for (int cell_id = tid; cell_id < n_cells; cell_id += blockDim.x * gridDim.x)
    {
        const int cell_x = cell_id % nx;
        const int cell_y = cell_id / nx;

        // update E
        const float chx = hx[cell_id];
        const float chy = hy[cell_id];

        const int left_neighbor_id = cell_x == 0 ? cell_y * nx + nx - 1 : cell_id - 1;
        const int bottom_neighbor_id = nx * (cell_y == ny - 1 ? ny - 1 : cell_y - 1) + cell_x
        const float chz = (chy - hy[left_neighbor_id])   / dx
                        - (chx - hx[bottom_neighbor_id]) / dy;

        dz[cell_id] += C0_p_dt * chz;

        if (cell_id == source_position)
        {
            dz[cell_id] += calculate_source (t, 5E+7);
            t += dt;
        }

        ez[cell_id] = dz[cell_id] * er[cell_id];
    }
    grid.sync ();
}


void baseline_fdtd_cg(
const int n_steps,
// param to scale problem size
const int nx, const int ny, const int n_cells, 
float *  ez, 
const float * mh,
// ez and mh update hx and hy
float * hx, float * hy,
// hx and hy update dz
float * dz,
// dz and er update ez
const float * er
) {
    int num_blocks;
    int num_threads_per_block;

    cudaOccupancyMaxPotentialBlockSize(&num_blocks, &num_threads_per_block, fdtd_cg);
    dim3 dim_block (num_threads_per_block, 1, 1);
    dim3 dim_grid (num_blocks, 1, 1);

    printf("nx: %d, ny:%d, n_cells: %d \n", nx, ny, n_cells);
    void *kernelArgs[] = { 
        (void*)&nx, (void*)&ny, (void*)&n_cells, 
        (void*)&ez, 
        (void*)&mh, 
        (void*)&hx, (void*)&hy,
        (void*)&dz,
        (void*)&er,
    };

    for (int step_idx = 0; step_idx < n_steps; step_idx++)
    {
        cudaLaunchCooperativeKernel((void*)fdtd_cg, dim_grid, dim_block, kernelArgs);
        throw_on_error(cudaGetLastError(), __FILE__, __LINE__);
    }

    throw_on_error(cudaStreamSynchronize(0), __FILE__, __LINE__);

}