#pragma once

#include "integer_divider.cuh"

static const int MAX_DIMS = 25;

template<int NARGS>
struct OffsetCalculator {
  struct offsets_t {
    __device__ unsigned int& operator[](int idx) { return values[idx]; }
    unsigned int values[NARGS];
  };

  explicit OffsetCalculator(int Dims, const int64_t* sizes, std::array<const int64_t*, NARGS> instrides) : Dims(Dims) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < Dims) {
        sizes_[i] = IntDivider<unsigned int>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<unsigned int>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        strides[i][arg] =  i < Dims ? instrides[arg][i] : 0;
      }
    }
  }


  __host__ __device__
  offsets_t get(unsigned int linearIndex) const {
    offsets_t offsets;
    #pragma unroll
    for (int j = 0; j < NARGS; j++) {
      offsets.values[j] = 0;
    }

    #pragma unroll
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i == Dims) {
        break;
      }
      DivMod<unsigned int> divmod = sizes_[i].divmod(linearIndex);
      linearIndex = divmod.div;

      #pragma unroll
      for (int j = 0; j < NARGS; j++) {
        offsets.values[j] += divmod.mod * strides[i][j];
      }

    }
    return offsets;
  }

  int Dims;
  IntDivider<unsigned int> sizes_[MAX_DIMS];
  unsigned int strides[MAX_DIMS][NARGS];
};


template<int N>
static OffsetCalculator<N> make_offset_calc(int Dims, const int64_t* sizes, std::array<const int64_t*, N> instrides) {
  return OffsetCalculator<N>(Dims, sizes, instrides);
}
