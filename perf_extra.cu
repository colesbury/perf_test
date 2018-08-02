#include "timestamps.cuh"
#include "common.cuh"
#include "offset_info.cuh"

#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

using cast_fn_t = void(*)(float*, const float*);

__device__ __noinline__ void cast3(float* a, const float *b) {
  *a = *b;
}

__constant__ cast_fn_t function_table[] = {cast3};

cast_fn_t get_cast_fn() {
  cast_fn_t fn = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&fn, function_table, sizeof(cast_fn_t)));
  std::cout << "got " << fn << "\n";
  return fn;
}
