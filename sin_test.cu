#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void sin_kernel(int* __restrict__ out, const float* __restrict__ x) {
  *out = std::pow(4,3);
}

//__global__
//void sinf_kernel(float* __restrict__ out, const float* __restrict__ x) {
//  *out = sinf(*x);
//}

int main(int argc, char* argv[]) {
  return 0;
}
