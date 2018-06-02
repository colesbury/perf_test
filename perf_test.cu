#include "timestamps.cuh"
#include "common.cuh"
#include "offset_info.cuh"

#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

void* cuda_malloc(size_t size) {
  void* devPtr;
  CUDA_CHECK(cudaMalloc(&devPtr, size));
  return devPtr;
}

static const int NT = 256;
static const int VT = 4;

// static const int NT = 512;
// static const int VT = 1;


__launch_bounds__(NT, 4)
__global__
void add_kernel(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ y, int N,
                OffsetInfo o1, StrideInfo s1, StrideInfo s2, StrideInfo s3) {
  int tid = threadIdx.x;
  int cta = blockIdx.x;
  int lane = tid % 32;
  int warp = tid / 32;
  int nv = NT * VT;
  int start = nv * cta;
  int end = min(N, nv * (cta + 1));
  int count = end - start;
  if (count >= NT * VT) {
    int linearIndex = start + warp * VT * 32 + lane;

    int counter0;
    int idx1, idx2, idx3;
    o1.get(linearIndex, &counter0, s1, s2, s3, &idx1, &idx2, &idx3);
    counter0 = __shfl_sync(0xFFFFFFFF, counter0, 32);
    if (counter0 + VT * 32 < o1.sizes_[0].divisor) {
      #pragma unroll
      for (int i = 0; i < VT; i++) {
        out[idx1] = x[idx2] + y[idx3];
        idx1 += s1.strides[0] * 32;
        idx2 += s2.strides[0] * 32;
        idx3 += s3.strides[0] * 32;
      }
      return;
    }

    #pragma unroll
    for (int i = 0; i < VT; i++) {
      int idx1, idx2, idx3;
      o1.get(linearIndex, &counter0, s1, s2, s3, &idx1, &idx2, &idx3);

      out[idx1] = x[idx2] + y[idx3];
      linearIndex += 32;
    }
  } else {
    // assert(0);
  }
}

static void verify(float* out_cuda, float* x_cuda, float* y_cuda, int N) {
  float* x = (float*)malloc(N * sizeof(float));
  float* y = (float*)malloc(N * sizeof(float));
  float* out = (float*)malloc(N * sizeof(float));
  CUDA_CHECK(cudaMemcpy(x, x_cuda, N * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(y, y_cuda, N * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(out, out_cuda, N * sizeof(float), cudaMemcpyDefault));
  bool non_zero = false;
  for (int i = 0; i < N; i++) {
    if (out[i] != x[i] + y[i]) {
      throw std::runtime_error(std::string("error at ") + std::to_string(i));
    }
    if (x[i] != 0 && y[i] != 0) {
      non_zero = true;
    }
  }
  if (!non_zero) {
    throw std::runtime_error("all zero");
  }
  std::cout << "OK\n";
}

static uint64_t x = 7; /* The state can be seeded with any value. */

uint64_t next() {
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

float next_float() {
  return (float)((next() >> 11) * (1. / (UINT64_C(1) << 53)));
}

static void fill_random(float* out_cuda, int N) {
  float* cpu = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    cpu[i] = next_float();
  }
  CUDA_CHECK(cudaMemcpy(out_cuda, cpu, N * sizeof(float), cudaMemcpyDefault));
  free(cpu);
}

int main(int argc, char* argv[]) {
  static const int N = 1024 * 1024 * 10;
  int64_t sizes[] = {10 * 32, 32, 32, 32, 1};
  int64_t strides[] = {1, 320, 10240, 327680, 327680};

  auto offset = OffsetInfo(5, sizes);
  auto stride_info = StrideInfo(5, strides);

  auto x = (float*)cuda_malloc(N * 2 * sizeof(float));
  auto y = (float*)cuda_malloc(N * 2 * sizeof(float));
  auto res = (float*)cuda_malloc(N * 2 * sizeof(float));

  fill_random(x, N);
  fill_random(y, N);

  cudaDeviceProp deviceProperties;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

  std::cout << "multiProcessorCount: " << deviceProperties.multiProcessorCount << "\n";

  dim3 block(NT);
  dim3 grid(N / block.x / VT);

  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < 10; i++) {
    cuda_timestamp start;
    add_kernel<<<grid, block>>>(res, x, y, N, offset, stride_info, stride_info, stride_info);
    std::cout << "time " << start.elapsed_time() << "\n";
  }

  verify(res, x, y, N);

  // cuda_timestamp end;
  // int64_t ts[10];
  // for (int i = 0; i < 10; i++) {
  //   timestamp ss;
  //   // start.record();
  //   // end.record();
  //   ts[i] = ss.elapsed_time();
  // }
  // for (int i = 0; i < 10; i++) {
  //   std::cout << ts[i] << "\n";
  // }

  return 0;
}
