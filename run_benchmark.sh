#!/bin/bash

# Function to run benchmark and save results
run_benchmark() {
    local grid_size_x=$1
    local grid_size_y=$2
    local steps=$3
    local output_file=$4

    echo "Running benchmark with grid size ${grid_size_x}x${grid_size_y}, steps: $steps"
    ./fdtd_benchmark $grid_size_x $grid_size_y $steps | tee -a $output_file
    echo "----------------------------------------" >> $output_file
}

# Create build directory and compile
rm -rf build && mkdir -p build
cd build
cmake ..
make

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Create results directory
mkdir -p benchmark_results

# Set benchmark parameters
steps=1000
output_file="benchmark_results/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

# Run benchmarks for different grid sizes
echo "Benchmark Results" > $output_file
echo "==================" >> $output_file
echo "" >> $output_file

run_benchmark 100 100 $steps $output_file
run_benchmark 500 500 $steps $output_file
run_benchmark 1000 1000 $steps $output_file
run_benchmark 2000 2000 $steps $output_file

echo "Benchmarks completed. Results saved to $output_file"

# Return to the original directory
cd ..