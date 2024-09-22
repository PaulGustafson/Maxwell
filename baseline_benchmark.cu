
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "baseline_single_kernel.hh"

int main() {

    int nx = 1024;
    int ny = 1024;
    int n_cells = nx * ny;
    int n_steps = 5;

    // Allocate memory on device
    // Device arrays
    float *ez, *dz, *hx, *hy, *er, *mh;
    cudaMalloc(&ez, n_cells * sizeof(float));
    cudaMalloc(&dz, n_cells * sizeof(float));
    cudaMalloc(&hx, n_cells * sizeof(float));
    cudaMalloc(&hy, n_cells * sizeof(float));
    cudaMalloc(&er, n_cells * sizeof(float));
    cudaMalloc(&mh, n_cells * sizeof(float));

    // warm up
    {
        NvtxScope scope("warmup");
        baseline_fdtd_cg(
        n_steps,
        nx, ny, n_cells, 
        ez, 
        mh,
        hx, hy,
        dz,
        er);
    }

    {
        NvtxScope scope("warmup");
        // actual running
        baseline_fdtd_cg(
        n_steps,
        nx, ny, n_cells, 
        ez, 
        mh,
        hx, hy,
        dz,
        er);
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