// nemo_ptx.hpp - the cp.async group fences of the Nemotron kernels, and on the HIP build the PTX stand-ins.
// HIP copies synchronously (platform/hip_compat/ptx_emu.h), so there is no group to commit or wait for.
#pragma once

// the dynamic shared memory a block may ask for: 99 KB on NVIDIA, 64 KB of LDS a workgroup on AMD
#ifdef STRATA_HIP_GFX906
#define NEMO_SMEM_MAX (64 * 1024)
#else
#define NEMO_SMEM_MAX (99 * 1024)
#endif

#ifdef STRATA_HIP_GFX906
#include "strata/platform/hip_compat/ptx_emu.h"
#define NEMO_CP_COMMIT() ((void) 0)
#define NEMO_CP_WAIT(n) ((void) 0)
#else
#define NEMO_CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define NEMO_CP_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#endif
