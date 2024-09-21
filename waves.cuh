#ifndef WAVES_H
#define WAVES_H

#include <cuda_runtime.h>

constexpr float c_wave = 10.0f;
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-3f;
constexpr float c_p_dt = c_wave * dt;

extern float *u, *u_new;

void allocate_memory(int nx, int ny, float **u, float **u_new);
void initialize_fields(int nx, int ny, float *u, float *u_new);
void write_state(int nx, int ny, float *u, int step);
void run_fdtd_step(int nx, int ny, float dx, float dy, float c_p_dt, int source_position, float t,
                   float *u, float *u_new);
void free_memory(float *u, float *u_new);

__global__ void init_fields(int nx, int ny, float *u, float *u_new);
__global__ void fdtd_update(int nx, int ny, float dx, float dy, float c_p_dt, int source_position, float t,
                            float *u, float *u_new);

#endif // WAVES_H
