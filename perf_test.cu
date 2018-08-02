#include "timestamps.cuh"
#include "common.cuh"
#include "offset_info.cuh"

#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

#define MGPU_DEVICE __device__
#define MGPU_LAMBDA __device__

void* cuda_malloc(size_t size) {
  void* devPtr;
  CUDA_CHECK(cudaMalloc(&devPtr, size));
  return devPtr;
}

static const int NT = 128;
static const int VT = 4;

// static const int NT = 512;
// static const int VT = 1;

enum ScalarType {
  Byte,
  Char,
  Short,
  Int,
  Long,
  Float,
  Double
};

static const int BYTE = 0x001;
static const int CHAR = 0x002;
static const int SHORT = 0x004;
static const int INT = 0x1;
static const int LONG = 0x1;
static const int FLOAT = 0x1;
static const int DOUBLE = 0x1;


// __device__ __noinline__ void cast(char* out, const char* in, ScalarType tout, ScalarType tin) {
//   int64_t intval;
//   float floatval;
//   double doubleval;
//   switch (tin) {
//     case Byte: intval = *(uint8_t*)in; break;
//     case Char: intval = *(int8_t*)in; break;
//     case Short: intval = *(int16_t*)in; break;
//     case Int: intval = *(int32_t*)in; break;
//     case Long: intval = *(int64_t*)in; break;
//     case Float: floatval = *(float*)in; break;
//     case Double: doubleval = *(double*)in; break;
//   }
//   *(float*)out = (float)intval;
// }
// __global__
// void sam_kernel(int N, int* out) {
//   *out = N;
// }

// template<typename func_t>
// __launch_bounds__(NT, 4)
// __global__
// void generic_kernel(int N, func_t f) {
//   int tid = threadIdx.x;
//   int cta = blockIdx.x;
//   int nv = NT * VT;
//   int start = nv * cta;
//   int end = min(N, nv * (cta + 1));
//   int count = end - start;
//
//   if (count >= NT * VT) {
//     int idx = start + tid;
//
//     #pragma unroll
//     for (int i = 0; i < VT; i++) {
//       f(idx);
//       idx += NT;
//     }
//   } else {
//     assert(0);
//   }
// }

template<typename arg1_t_, typename arg2_t_=arg1_t_, typename arg3_t_=arg2_t_>
struct binary_args_t {
  using arg1_t = arg1_t_;
  using arg2_t = arg2_t_;
  using arg3_t = arg3_t_;
};

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    typedef std::tuple<Args...> args;

    template <size_t i>
    struct arg
    {
        typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

template <typename T>
struct binary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  // using arg2_t = arg<1>::type;
};


template<typename args, typename func_t>
void launch_kernel(int N, const int64_t* sizes, const int64_t* strides, char** data_ptrs, func_t f) {
  std::array<const int64_t*, 3> all_strides = {strides, strides, strides};
  int ndim = 5;
  auto offset_calc = OffsetCalculator<3>(ndim, sizes, all_strides);

  using arg1_t = float;
  using arg2_t = float;
  using arg3_t = float;
  arg1_t* out = (arg1_t*)data_ptrs[0];
  arg2_t* in1 = (arg2_t*)data_ptrs[1];
  arg3_t* in2 = (arg3_t*)data_ptrs[2];

  dim3 block(NT);
  dim3 grid(N / block.x / VT);
  generic_kernel<<<grid, block>>>(N, [=]MGPU_DEVICE(int idx) {


    auto offsets = offset_calc.get(idx);
    auto res = f(in1[offsets[1]], in2[offsets[2]]);
    using tmp = decltype(res);

    out[offsets[0]] = (tmp)res;
  });
}

template <int nt, int vt>
__global__ void bandwidth_kernel(int N, const float* input, float* output) {
  int idx = threadIdx.x + blockIdx.x * nt;

  float buffer[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    buffer[i] = input[idx + i * nt * gridDim.x];
  }
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (buffer[i] == 76.5) {
      output[0] = 1;
    }
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
      throw std::runtime_error(std::string("sum incorrect at ") + std::to_string(i));
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

void* global;

int main(int argc, char* argv[]) {
  static const int N = 1024 * 1024 * 10;
  int64_t sizes[] = {10 * 32, 32, 32, 32, 1};
  int64_t strides[] = {1, 320, 10240, 327680, 327680};

  // auto offset = OffsetCalculator(5, sizes);
  // auto stride_info = StrideInfo(5, strides);

  auto x = (float*)cuda_malloc(N * sizeof(float));
  auto y = (float*)cuda_malloc(N * sizeof(float));
  auto res = (float*)cuda_malloc(N * sizeof(float));

  fill_random(x, N);
  fill_random(y, N);

  cudaDeviceProp deviceProperties;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

  std::cout << "multiProcessorCount: " << deviceProperties.multiProcessorCount << "\n";

  static constexpr int nt = 512;
  static constexpr int vt = 4;
  dim3 block(512);
  dim3 grid(N / nt / vt);

  std::cout << grid.x << " x " << block.x << "\n";

  CUDA_CHECK(cudaDeviceSynchronize());
  char* data[3] = {(char*)res, (char*)x, (char*)y};
  for (int i = 0; i < 10; i++) {
    cuda_timestamp start;
    bandwidth_kernel<nt, vt><<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    // bandwidth_kernel<<<grid, block>>>(N, x, y);
    std::cout << "time " << start.elapsed_time() << "\n";
  }

  // verify(res, x, y, N);

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
