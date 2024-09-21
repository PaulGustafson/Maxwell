#ifndef MAXWELLS_H
#define MAXWELLS_H

#include <cuda_runtime.h>

// Constants
constexpr float C0 = 299792458.0f;
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float dt = 1e-9f;
constexpr float C0_p_dt = C0 * dt;

// Public function declarations
void allocate_memory(int nx, int ny, float **ez, float **dz, float **hx, float **hy, float **er, float **mh);
void initialize_fields(int nx, int ny, float *ez, float *dz, float *hx, float *hy, float *er, float *mh);
void run_fdtd_step(int nx, int ny, float dx, float dy, float C0_p_dt, int source_position, float t,
                   float *ez, float *dz, float *hx, float *hy, float *er, float *mh);
void free_memory(float *ez, float *dz, float *hx, float *hy, float *er, float *mh);

#endif // MAXWELLS_H