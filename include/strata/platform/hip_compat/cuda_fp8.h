#pragma once
// the OCP E4M3 type and conversion under their CUDA names (hip_fp8.h converts in software where the GPU has no FP8)
#include "strata_hip.h"
#include <hip/hip_fp8.h>
#define __nv_fp8_e4m3 __hip_fp8_e4m3
#define __nv_fp8_storage_t __hip_fp8_storage_t
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3
