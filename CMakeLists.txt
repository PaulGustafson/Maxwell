cmake_minimum_required(VERSION 3.10)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_project main.cu maxwells.cu)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
