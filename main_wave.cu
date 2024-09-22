#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include "waves.cuh"

// Add these definitions at the global scope in main.cu
float *u, *u_new;

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
    allocate_memory(nx, ny, &u, &u_new);

    // Initialize fields
    initialize_fields(nx, ny, u, u_new);

    // Write state to a file.
    write_state(nx, ny, &u, 0);

    // Free memory
    free_memory(u, u_new);

    return 0;
}
