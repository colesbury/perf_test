#pragma once

#include "integer_divider.cuh"

static const int MAX_DIMS = 25;

struct StrideInfo {
  explicit StrideInfo(int Dims, const int64_t* strides) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < Dims) {
        this->strides[i] = strides[i];
      } else {
        this->strides[i] = 0;
      }
    }
  }
  unsigned int strides[MAX_DIMS];
};

struct OffsetInfo {
  explicit OffsetInfo(int Dims, const int64_t* sizes) : Dims(Dims) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < Dims) {
        sizes_[i] = IntDivider<unsigned int>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<unsigned int>(1);
      }
    }
  }

  __host__ __device__ void get(unsigned int linearIndex,
                               const StrideInfo& s1,
                               const StrideInfo& s2,
                               const StrideInfo& s3,
                               int* idx1,
                               int* idx2,
                               int* idx3) const {
    *idx1 = 0;
    *idx2 = 0;
    *idx3 = 0;

    #pragma unroll
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i == Dims) {
        break;
      }
      DivMod<unsigned int> divmod = sizes_[i].divmod(linearIndex);
      linearIndex = divmod.div;
      *idx1 += divmod.mod * s1.strides[i];
      *idx2 += divmod.mod * s2.strides[i];
      *idx3 += divmod.mod * s3.strides[i];
    }
  }

  int Dims;
  IntDivider<unsigned int> sizes_[MAX_DIMS];
};
