cmake_minimum_required(VERSION 3.10)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Adding Raylib
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE) # don't build the supplied example games

FetchContent_Declare(
    raylib
    GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
    GIT_TAG "master"
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(raylib)

add_executable(maxwells main.cu maxwells.cu)
add_executable(waves main_wave.cu waves.cu)
add_executable(fdtd_benchmark fdtd_benchmark.cu)
# find_package(raylib REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

set_target_properties(maxwells PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(waves PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fdtd_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)