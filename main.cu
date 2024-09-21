#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include "maxwells.cuh"

// Add these definitions at the global scope in main.cu
float *ez, *dz, *hx, *hy, *er, *mh;

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " nx ny steps inc" << std::endl;
        return 1;
    }

    // Parse command line arguments
    int nx = 0, ny = 0, steps = 0, increment = 0;
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--grid_size_x") nx = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--grid_size_y") ny = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--steps") steps = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--increment") increment = std::atoi(argv[i + 1]);
    }
    
    // Check if all required parameters were provided
    if (nx == 0 || ny == 0 || steps == 0 || increment == 0) {
        std::cerr << "Error: Missing or invalid command line arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " --grid_size_x <nx> --grid_size_y <ny> --steps <steps> --increment <increment>" << std::endl;
        return 1;
    }
    // Print out the args for debugging
    std::cout << "Debugging: Command line arguments" << std::endl;
    std::cout << "grid_size_x (nx): " << nx << std::endl;
    std::cout << "grid_size_y (ny): " << ny << std::endl;
    std::cout << "steps: " << steps << std::endl;
    std::cout << "increment: " << increment << std::endl;
    std::cout << std::endl;

    // Grid dimensions
    int source_position = (nx / 2) * nx + (ny / 2);  
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
