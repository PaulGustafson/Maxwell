
#ifndef MULTIGPUFDTD_GPU_UTILS_H
#define MULTIGPUFDTD_GPU_UTILS_H

#include <sstream>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

inline void throw_on_error (cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
    {
      std::stringstream ss;
      ss << cudaGetErrorString (code);
      std::cout << ss.str() << std::endl;
    //   throw std::runtime_error (file_and_line);
    }
}


#endif //MULTIGPUFDTD_GPU_UTILS_H
