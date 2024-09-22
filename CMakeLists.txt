cmake_minimum_required(VERSION 3.10)
project(cuda_project LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_90,code=sm_90")
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -g")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

add_executable(cuda_project main.cu)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(baseline_benchmark baseline_single_kernel.hh baseline_single_kernel.cu baseline_benchmark.cu gpu_utils.h)
target_include_directories(baseline_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})