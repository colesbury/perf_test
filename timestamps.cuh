#pragma once

#include "common.cuh"

#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>


struct timestamp {
  timestamp() {
    clock_gettime(CLOCK_MONOTONIC, &tv);
  }
  int64_t elapsed_time() {
    return elapsed_time(timestamp());
  }
  int64_t elapsed_time(const timestamp& other) {
    int64_t ds = (other.tv.tv_sec - tv.tv_sec);
    ds *= 1000000000;
    ds += (other.tv.tv_nsec - tv.tv_nsec);
    return ds;
  }
  struct timespec tv;
};

struct cuda_timestamp {
  cuda_timestamp() {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDefault));
    record();
  }
  ~cuda_timestamp() {
    CUDA_CHECK(cudaEventDestroy(event));
  }
  int64_t elapsed_time() {
    return elapsed_time(cuda_timestamp());
  }
  void record() {
    CUDA_CHECK(cudaEventRecord(event));
  }
  int64_t elapsed_time(const cuda_timestamp& other) {
    CUDA_CHECK(cudaEventSynchronize(other.event));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, event, other.event));
    return (int64_t)((double)ms * 1000000);
  }
  cudaEvent_t event;
};
