#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include "maxwells.h"

constexpr int nx = 1000;  // Grid size
constexpr int ny = 1000;
constexpr int increment = 10;   // Steps between data logging event
constexpr int steps = increment*15*10;   // Number of time steps
constexpr int source_position = (nx / 2) * nx + (ny / 2);  // Center of the grid

int main() {
    float *ez, *dz, *hx, *hy, *er, *mh;

    // Allocate memory
    allocate_memory(nx, ny, &ez, &dz, &hx, &hy, &er, &mh);

    // Initialize fields
    initialize_fields(nx, ny, ez, dz, hx, hy, er, mh);

    // Time-stepping loop
    for (int step = 0; step < steps; ++step) {
        float t = step * dt;
        run_fdtd_step(nx, ny, dx, dy, C0_p_dt, source_position, t, ez, dz, hx, hy, er, mh);

        // Optionally print or log values of the field at the center of the grid
        if (step % increment == 0) {
            float ez_center;
            cudaMemcpy(&ez_center, &ez[source_position], sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Step " << step << ", t = " << t << ", Ez at center: " << ez_center << std::endl;

            // Print to file
            float *h_ez = new float[nx * ny];  // Host-side copy of ez
            cudaMemcpy(h_ez, ez, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
            std::ofstream file("data/ez_step_" + std::to_string(step) + ".txt");
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    file << h_ez[y * nx + x] << " ";
                }
                file << "\n";
            }
            file.close();
            delete[] h_ez;
        }
    }

    // Free memory
    free_memory(ez, dz, hx, hy, er, mh);

    return 0;
}
