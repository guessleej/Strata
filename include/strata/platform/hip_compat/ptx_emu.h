// ptx_emu.h - the PTX the Nemotron kernels use (cp.async, ldmatrix, mma.sync), written out for the HIP build.
//
// Each function keeps the PTX instruction's lane contract: the same thread receives the same fragment and returns
// the same accumulator registers as on NVIDIA, so the kernels around them stay as they are.  A "warp" is a 32-lane
// group (wave32 on RDNA; the gfx906 compat layer's logical half-wave on wave64), and every lane of it must call.
//
// - cp.async is a plain load and store: the copy is complete when the call returns, so commit_group and wait_group
//   have nothing to wait for.  The src-size form zero-fills the bytes past `bytes`, as PTX does.
// - ldmatrix reads the 8 row addresses of each 8 x 8 b16 matrix from the lanes that supplied them (a shuffle) and
//   then reads LDS directly.
// - mma.sync gathers the rows of A and the columns of B that a lane's outputs need by shuffles and does the dot
//   products itself.  f16 and bf16 accumulate in FP32 (the f16-accumulate form rounds once at the end); s8 is exact.
//   The sums run in k order within each 8-wide group, which is not the tensor core's order: results match to FP32
//   rounding, not bit for bit.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace strata::ptx {

typedef __attribute__((address_space(3))) uint8_t lds_u8;
typedef __attribute__((address_space(3))) uint16_t lds_u16;
typedef __attribute__((address_space(3))) uint32_t lds_u32;

/// the 32-bit LDS address of a generic pointer into shared memory (PTX cvta.to.shared)
__device__ __forceinline__ uint32_t cvta_shared(const void* p) {
    return (uint32_t) (uintptr_t) (const __attribute__((address_space(3))) void*) p;
}

__device__ __forceinline__ int lane32() { return (int) (__lane_id() & 31u); }

__device__ __forceinline__ uint32_t shfl32(uint32_t v, int src) { return (uint32_t) __shfl((int) v, src, 32); }

/// cp.async.{ca,cg}.shared.global [dst], [src], N, bytes: N bytes into LDS, the ones past `bytes` zero
template <int N>
__device__ __forceinline__ void cp_async(uint32_t dst, const void* src, int bytes = N) {
    static_assert(N == 4 || N == 8 || N == 16, "cp.async copies 4, 8 or 16 bytes");
    if (bytes >= N) {
        // word by word: HIP's vector types have no assignment into another address space (the stores still merge)
        lds_u32* d = (lds_u32*) (uintptr_t) dst;
        const uint32_t* s = (const uint32_t*) src;
#pragma unroll
        for (int i = 0; i < N / 4; ++i) d[i] = s[i];
        return;
    }
    const uint8_t* s = (const uint8_t*) src;
    lds_u8* d = (lds_u8*) (uintptr_t) dst;
#pragma unroll
    for (int i = 0; i < N; ++i) d[i] = i < bytes ? s[i] : (uint8_t) 0;
}

/// ldmatrix.sync.aligned.m8n8.x4.shared.b16: lane l gives the address of row l % 8 of matrix l / 8; r[j] is row
/// l / 4 of matrix j, b16 values 2 (l % 4) and 2 (l % 4) + 1
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
    const int l = lane32();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t row = shfl32(addr, 8 * j + (l >> 2));
        r[j] = *(lds_u32*) (uintptr_t) (row + 4u * (uint32_t) (l & 3));
    }
}

/// ldmatrix ... .trans: r[j] holds matrix j's values (2 (l % 4), l / 4) low and (2 (l % 4) + 1, l / 4) high
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, uint32_t addr) {
    const int l = lane32(), g = l >> 2, t = l & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t r0 = shfl32(addr, 8 * j + 2 * t), r1 = shfl32(addr, 8 * j + 2 * t + 1);
        const uint32_t lo = *(lds_u16*) (uintptr_t) (r0 + 2u * (uint32_t) g);
        const uint32_t hi = *(lds_u16*) (uintptr_t) (r1 + 2u * (uint32_t) g);
        r[j] = lo | (hi << 16);
    }
}

__device__ __forceinline__ float2 f16x2(uint32_t u) {
    __half2 h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return __half22float2(h);
}
__device__ __forceinline__ float2 bf16x2(uint32_t u) {
    return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}

/// m16n8k16, A row-major / B column-major, 16-bit inputs (Cvt: a packed pair to two floats), FP32 accumulators.
/// Thread (g, t) = (l / 4, l % 4) holds A rows g, g + 8 at k 2t.. (a0, a1) and 2t + 8.. (a2, a3), B column g at
/// k 2t.. (b0) and 2t + 8.. (b1), and outputs (g, 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1).
template <float2 (*Cvt)(uint32_t)>
__device__ __forceinline__ void mma_m16n8k16_f32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
    const int l = lane32(), g = l >> 2, t = l & 3;
    float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int la = 4 * g + s, lb0 = 4 * (2 * t) + s, lb1 = 4 * (2 * t + 1) + s;
        const float2 r0lo = Cvt(shfl32(a0, la)), r1lo = Cvt(shfl32(a1, la));
        const float2 r0hi = Cvt(shfl32(a2, la)), r1hi = Cvt(shfl32(a3, la));
        const float2 c0lo = Cvt(shfl32(b0, lb0)), c0hi = Cvt(shfl32(b1, lb0));
        const float2 c1lo = Cvt(shfl32(b0, lb1)), c1hi = Cvt(shfl32(b1, lb1));
        d0 += r0lo.x * c0lo.x + r0lo.y * c0lo.y + r0hi.x * c0hi.x + r0hi.y * c0hi.y;
        d1 += r0lo.x * c1lo.x + r0lo.y * c1lo.y + r0hi.x * c1hi.x + r0hi.y * c1hi.y;
        d2 += r1lo.x * c0lo.x + r1lo.y * c0lo.y + r1hi.x * c0hi.x + r1hi.y * c0hi.y;
        d3 += r1lo.x * c1lo.x + r1lo.y * c1lo.y + r1hi.x * c1hi.x + r1hi.y * c1hi.y;
    }
    d[0] = d0;
    d[1] = d1;
    d[2] = d2;
    d[3] = d3;
}

/// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__ void mma_bf16_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    mma_m16n8k16_f32<bf16x2>(d, a[0], a[1], a[2], a[3], b0, b1);
}

/// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16: c[0] = (g, 2t), (g, 2t + 1); c[1] = rows g + 8
__device__ __forceinline__ void mma_f16_f16(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const float2 lo = f16x2(c[0]), hi = f16x2(c[1]);
    float d[4] = {lo.x, lo.y, hi.x, hi.y};
    mma_m16n8k16_f32<f16x2>(d, a[0], a[1], a[2], a[3], b0, b1);
    const __half2 h0 = __floats2half2_rn(d[0], d[1]), h1 = __floats2half2_rn(d[2], d[3]);
    __builtin_memcpy(&c[0], &h0, 4);
    __builtin_memcpy(&c[1], &h1, 4);
}

/// four signed bytes times four signed bytes, plus c
__device__ __forceinline__ int dot4_s8(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t* va = reinterpret_cast<const int8_t*>(&a);
    const int8_t* vb = reinterpret_cast<const int8_t*>(&b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
}

/// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.  Thread (g, t) holds A rows g, g + 8 at k 4t.. (a0, a1) and
/// 16 + 4t.. (a2, a3), B column g at k 4t.. (b0) and 16 + 4t.. (b1); outputs as in the 16-bit form.
__device__ __forceinline__ void mma_s8_s32(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const int l = lane32(), g = l >> 2, t = l & 3;
    int d0 = c[0], d1 = c[1], d2 = c[2], d3 = c[3];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int la = 4 * g + s, lb0 = 4 * (2 * t) + s, lb1 = 4 * (2 * t + 1) + s;
        const int r0lo = (int) shfl32(a[0], la), r1lo = (int) shfl32(a[1], la);
        const int r0hi = (int) shfl32(a[2], la), r1hi = (int) shfl32(a[3], la);
        const int c0lo = (int) shfl32(b0, lb0), c0hi = (int) shfl32(b1, lb0);
        const int c1lo = (int) shfl32(b0, lb1), c1hi = (int) shfl32(b1, lb1);
        d0 = dot4_s8(r0hi, c0hi, dot4_s8(r0lo, c0lo, d0));
        d1 = dot4_s8(r0hi, c1hi, dot4_s8(r0lo, c1lo, d1));
        d2 = dot4_s8(r1hi, c0hi, dot4_s8(r1lo, c0lo, d2));
        d3 = dot4_s8(r1hi, c1hi, dot4_s8(r1lo, c1lo, d3));
    }
    c[0] = d0;
    c[1] = d1;
    c[2] = d2;
    c[3] = d3;
}

}  // namespace strata::ptx
