#pragma once
#include "strata_hip.h"
#include <hip/hip_bf16.h>
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
static __host__ __device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
    return __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
}
