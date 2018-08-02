#pragma once

#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

static inline void CUDA_CHECK(cudaError_t err) {
  if(err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda failed: ") + cudaGetErrorString(err));
  }
}

using cast_fn_t = void(*)(float*, const float*);
