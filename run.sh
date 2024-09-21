#!/bin/bash

# Check if a YAML file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_yaml_file>"
    exit 1
fi

yaml_file="$1"

# Extract parameters from YAML file
grid_size_x=$(grep 'grid_size_x:' "$yaml_file" | awk '{print $2}')
grid_size_y=$(grep 'grid_size_y:' "$yaml_file" | awk '{print $2}')
steps=$(grep 'steps:' "$yaml_file" | awk '{print $2}')
increment=$(grep 'increment:' "$yaml_file" | awk '{print $2}')
# Add more parameter extractions as needed

# Build and run the project with parameters
rm -rf build && mkdir -p build/data && cd build && cmake .. && make && \
./cuda_project --grid_size_x "$grid_size_x" --grid_size_y "$grid_size_y" --steps "$steps" --increment "$increment" && cd ..
# Add more parameters to the ./cuda_project call as needed