#pragma once
#include "strata_hip.h"
#include <hip/hip_fp4.h>
#define __nv_cvt_float_to_fp4 __hip_cvt_float_to_fp4
#define __NV_E2M1 __HIP_E2M1
#define cudaRoundNearest hipRoundNearest
