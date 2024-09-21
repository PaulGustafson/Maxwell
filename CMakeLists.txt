cmake_minimum_required(VERSION 3.10)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(maxwells main.cu maxwells.cu)
add_executable(waves main_wave.cu waves.cu)
add_executable(fdtd_benchmark fdtd_benchmark.cu)

set_target_properties(maxwells PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(waves PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fdtd_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)