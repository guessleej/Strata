# strata-nemotron on AMD Strix Halo (gfx1151)

The Nemotron 3.5 Super engine of this fork (`strata-nemotron`) built for the Ryzen AI Max "Strix Halo" APU (Radeon 8060S,
RDNA3.5, `gfx1151`). The engine was written for one RTX 5090 (sm_120a): its kernels use PTX directly (`cp.async`, `ldmatrix`,
`mma.sync`, FP4 block-scaled mma). This port keeps those kernels and gives them HIP stand-ins with the same lane contract, so
the kernel code itself is unchanged. **Status: experimental.** Every kernel passes the fork's own parity tests on gfx1151
against the same float64 references as on the 5090, and the whole 71.6 GB model loads and answers on one Strix Halo box:
10–16 tok/s decode, ~170 tok/s prefill (see [The whole model](#the-whole-model)).

Ported by xCloudinfo. Branch `hip-gfx1151` of [guessleej/Strata](https://github.com/guessleej/Strata/tree/hip-gfx1151).

## The machine

HP Z2 Mini G1a: Ryzen AI Max+ PRO 395, Radeon 8060S, 128 GB LPDDR5X with 96 GB carved out for the GPU in the BIOS (31 GB left to
Linux). Ubuntu 24.04.4, kernel 6.17.0-35, amdgpu DKMS 6.16.6. Compiled with ROCm 7.14.1 from AMD's TheRock builds (clang 23), run
on the system's ROCm 7.1.1.

## What the port does

It takes the upstream `STRATA_HIP_GFX906` mode, which compiles the whole CUDA tree as HIP through
`include/strata/platform/hip_compat/`, and points it at gfx1151 (`CMAKE_HIP_ARCHITECTURES=gfx1151`). On top of that mode:

- **PTX stand-ins** (`hip_compat/ptx_emu.h`, namespace `strata::ptx`), by wave32 shuffles and direct LDS reads:
  - `cp.async` (16/8/4 bytes, with the zero fill past `src-size`): a synchronous copy, so `cp.async.commit_group` / `wait_group`
    are no-ops (`NEMO_CP_COMMIT` / `NEMO_CP_WAIT` in `kernels/nemo_ptx.hpp`);
  - `ldmatrix.x4` and `ldmatrix.x4.trans`: each lane takes its row address from the lane that holds it in PTX;
  - `mma.sync` m16n8k16 (f16 and bf16 into FP32, f16 into f16) and m16n8k32 (s8 into s32): the A rows and B columns of a lane's
    fragment are gathered by shuffles and the dot products are done in FP32 / int32 (`v_dot4` through `sudot4` for int8). The
    accumulator layout is PTX's (`g = lane >> 2`, `t = lane & 3`), so what the kernels do with `d[]` afterwards is untouched.
- **64 KB of LDS a workgroup** (NVIDIA lets a block ask for 99 KB). A kernel that asks for more does not fail on HIP: it runs
  and writes zeros or garbage. Three kernels asked for more:
  - the GEMV's shared tile: `99 * 1024` became `NEMO_SMEM_MAX` (64 KB on HIP) in both places it is used;
  - the Q8 GEMM: two stages of `KS = 2` are 72 KB, so `NEMO_Q8_KS` is 1 on HIP;
  - attention: the key tile `AKT` is 64 on HIP (97 KB at 128, 49.6 KB at 64).
- **FP4 tensor cores stay sm_120a only.** `nemo_fp4_ok()` is false on HIP, so the prompt's expert GEMMs take the BF16 path
  (hipBLAS) that the engine already had for other cards. Decode does not use FP4 mma either way.
- **Compat headers** for what the Nemotron files include and the gfx906 mode did not need: `cuda_bf16.h`
  (`__floats2bfloat162_rn`), `cuda_fp8.h`, `cuda_fp4.h`, `cudaMemset2DAsync`, `__cvta_generic_to_shared`,
  `CUBLAS_COMPUTE_32F_FAST_TF32` (plain FP32 on hipBLAS). `strata_dp4a` uses `sudot4` on gfx11/gfx12, which have no `sdot4`.
- **CMake:** `nemo_c4_bench` (cuSPARSE) and `nemo_fp4_parity` (FP4 mma) are CUDA only.

12 files, +299 / −22 for the kernels, and the load change below. Nothing changes for a CUDA build: every change is behind `STRATA_HIP_GFX906` or in the compat headers.

## Parity on gfx1151

The references are the fork's own: `tools/nemotron/ref_layers.py`, `mamba_naive.py`, `attn_naive.py` and `moe_naive.py` in
float64 from NVIDIA's BF16 and NVFP4 checkpoints (the first shards are enough for blocks 0–7), 64 tokens. The reference itself
against transformers: Mamba block 0 2.1e-3 (transformers' chunked FP32 scan), attention block 7 1.9e-6, MoE blocks 1 / 5
3.9e-7 / 3.5e-7, as in the fork's notes. Full output: [`AMD_GFX1151_parity.txt`](AMD_GFX1151_parity.txt).

Relative error ||got − want|| / ||want||, worst token. The 5090 column is from [STAGE2-4_KERNELS.md](STAGE2-4_KERNELS.md).

**Mamba-2, block 0** (`mamba_parity`):

| | gfx1151 | RTX 5090 |
|---|---:|---:|
| y after the gated norm | 5.0e-7 | 5.0e-7 |
| final SSM state / conv state | 8.3e-8 / 0 | 1.1e-7 / 0 |
| whole mixer | 1.8e-7 | 2.3e-7 |
| prefill in two pieces (37 + 27) | 5.0e-7 | 5.0e-7 |
| prefill against decode of the same tokens | bit-identical | bit-identical |
| chunked SSD, 64 tokens in one call | 3.3e-7 | — |
| negative test (conv taps reversed) | 0.81 → caught | 0.81 → caught |

**Attention, block 7** (`attn_parity`), against the reference with BF16 K/V:

| | gfx1151 |
|---|---:|
| q projection | 6.2e-8 |
| decode, 64 chunks: attention / block output against the exact reference | 1.9e-2 / 7.5e-3 |
| prompt, 2 chunks | 2.5e-2 / 8.3e-3 |
| verify windows, 16 chunks | 1.9e-2 / 7.5e-3 |

These are larger than the 5.1e-7 in STAGE2-4_KERNELS.md because the K cache became int8 after that table was written:
`attn_parity` now expects about 2e-2 from int8 K (tolerance 3e-2 decode / 4e-2 prompt, block output under 1e-2). The prompt path
goes through the emulated int8 and f16 mma; decode does not. The two agree, so the emulation adds nothing measurable.

**MoE, blocks 1 and 5** (`moe_parity`, NVFP4 experts, 512 experts top-22):

| | block 1 | block 5 | RTX 5090 (1 / 5) |
|---|---:|---:|---:|
| router logits | 6.5e-8 | 5.7e-8 | 6.5e-8 / 5.8e-8 |
| top-22 sets that differ from the reference | 0 of 64 | 0 of 64 | 0 / 0 of 64 |
| routing weights | 1.9e-7 | 2.2e-7 | 2.2e-7 / 2.3e-7 |
| routed latent sum | 7.2e-7 | 3.3e-7 | 1.9e-6 / 3.1e-7 |
| MoE output | 5.2e-7 | 2.9e-7 | 5.4e-7 / 2.4e-7 |

**GEMM and GEMV** (synthetic, against float64):

| test | result |
|---|---|
| `nemo_q8_parity` | quantization and layouts byte for byte; W8A8 GEMM worst \|err\| 8.5e-5 of max \|y\| 310 |
| `nemo_gemm_parity` | NVFP4, Q8_0, BF16, F32 GEMV 3.1e-7 – 7.0e-7; M = 1..100 bit-identical to M = 1 |
| `nemo_iq_parity` | IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0 3.2e-7 – 5.9e-7; 22 experts at once = alone |

`nemo_q8_parity`'s speed line on this chip: the emulated int8 mma GEMM does 11 TOP/s at 16384 × 4096 × 4096, hipBLAS BF16
5 TFLOP/s. The emulation is a correctness port, not a fast one: gfx11 has WMMA, and a WMMA version of these fragments is the
obvious next step.

## The whole model

The engine keeps every expert in a page-locked RAM arena and copies misses into a VRAM cache. Here Linux has 31 GB of RAM
and the GPU 96 GB, so the arena cannot be allocated, while all 20,480 experts fit in the GPU's memory. With
`STRATA_NEMO_NO_ARENA=1` the arena is skipped; when the expert cache gets every expert of a layer (512 slots), it is filled
once at load and slot e holds expert e, so nothing is fetched afterwards. The fill copies through a page-locked blob: a
`hipMemcpy` straight from the read-only GGUF mapping never returned on ROCm 7.1.1 (the engine sat in the HSA runtime at
100% of one core). Misses, if any, still read the mapping. `STRATA_NEMO_NO_LOAN=1` keeps the chunk scratch from lending slots.

```
strata-nemotron: loaded in 131.5 s: dense 6.63 GiB in VRAM, 512 expert slots per MoE layer (59.1 GiB),
K/V 131072 cells (grows to 131072), experts NOT page-locked
```

System RAM stayed at 22 GB available through the whole run (load and every request). The config is
[`strata-nemotron.example.json`](strata-nemotron.example.json) without `vision`, `--prompt-cache` and
`--conversation-cache-mib` (both are pinned RAM), with `--max-context 131072 --kv-base 131072 --vram-reserve-mib 2048`
and `--engine strata` on `serve/server.py`.

**Speed**, through the server, greedy, MTP on (`--spec 5 --spec-min-p 0.8`):

| prompt, tokens | time to first token | prefill, tok/s | decode, tok/s |
|---:|---:|---:|---:|
| 328 | 6.9 s | 48 | 9.8 |
| 2,428 | 17.6 s | 138 | 10.0 |
| 9,628 | 55.4 s | 174 | 10.6 |
| 19,228 | 110.9 s | 173 | 9.9 |

Expert hit rate 100.0% on every request. MTP accepted 29–30 of 39–42 drafts in 18–19 windows (2.5–2.7 tokens a window). Short chat
answers (300–460 tokens) decoded at 11.5–16.0 tok/s; even a 30-token prompt took 5.1–5.7 s to the first token.

For scale, the RTX 5090 (README): 2,277 tok/s prefill and 118 tok/s decode at 4K. Decode here is about a tenth. The
memory is ~256 GB/s against ~1.8 TB/s, and the GEMV and attention kernels were written for sm_120's shapes. Prefill is
about a fourteenth: the prompt's experts run through hipBLAS BF16 instead of FP4 tensor cores, and the attention and
latent projections through the shuffle-emulated mma. Both are the next work (below).

**Answers.** Five fixed questions, thinking off, temperature 0:

| question | answer |
|---|---|
| Taiwan's National Health Insurance: the year it started, the ministry in charge (Traditional Chinese) | 1995-03-01, the Ministry of Health and Welfare (formerly the Department of Health, reorganized in 2013): right |
| two pipes fill a pool in 6 h and 3 h; both open | 1/6 + 1/3 = 1/2, 2 hours, with the steps: right |
| `is_prime(n)` and the primes below 20 | correct function, `[2, 3, 5, 7, 11, 13, 17, 19]`: right |
| rewrite a casual sentence as a formal Taiwanese government letter | three usable versions and a term table; one version uses 蒞臨 (an honorific for the other side) for the writer's own visit |
| answer with one JSON object, nothing else | `{"city": "Taipei", "country": "Taiwan"}` and nothing else: right |

Coherent, correct Traditional Chinese, no garbled tokens, which is what a broken kernel on this path would show first.

## Build

ROCm 7.14.1 (TheRock) for gfx1151 as in [STRIX_HALO.md](../STRIX_HALO.md), section 1. Then:

```sh
export SDK=$HOME/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
cmake -S . -B build-hip1151 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-hip1151 -j 8 --target strata-nemotron nemo_q8_parity nemo_gemm_parity nemo_iq_parity \
  mamba_parity attn_parity moe_parity
```

On this machine the binaries are run with the system's ROCm, not TheRock's: `LD_LIBRARY_PATH=/opt/rocm/lib`. With ROCm 7.14.1's
runtime, `hipMemcpy` host to device fails here (`PCIe probe failed`, then `invalid argument`) against amdgpu 6.16.6, while
`rocminfo` reports the GPU as fine. Compiled with 7.14.1 and run on 7.1.1, everything above passes.

The parity tests take a dump folder:

```sh
python tools/nemotron/ref_layers.py --ckpt bf16 --layers 8 --tokens 64 --out ref_bf16.npz
python tools/nemotron/mamba_naive.py ref_bf16.npz bf16 0 --dump dump_mamba
python tools/nemotron/attn_naive.py ref_bf16.npz bf16 7 --dump dump_attn
python tools/nemotron/ref_layers.py --ckpt nvfp4 --layers 6 --tokens 64 --out ref_nvfp4.npz --no-decode
python tools/nemotron/moe_naive.py ref_nvfp4.npz nvfp4 1 --dump dump_moe1
python tools/nemotron/moe_naive.py ref_nvfp4.npz nvfp4 5 --dump dump_moe5
LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip1151/mamba_parity dump_mamba   # and attn_parity dump_attn, moe_parity dump_moe1 ...
```

The references were made on a separate x86 machine (torch 2.15 CPU, transformers 5.21, llama.cpp's `gguf-py` on the
`PYTHONPATH`), from shards 1–5 of the BF16 checkpoint and 1–8 of the NVFP4 one, and copied over.

## Not done yet

- **Speed.** A tenth of the 5090 in decode, a fourteenth in prefill (see above); the 5 s to the first token of a short
  prompt is not explained yet.
- **WMMA for the emulated mma.** The int8 / f16 / bf16 fragments map onto gfx11's `v_wmma_*` instructions; the shuffle
  emulation is correct and slow. The prompt's experts could take the same route instead of hipBLAS BF16.
- **Other RDNA parts.** Only gfx1151 was built and tested. The same changes should hold for gfx1100 / gfx1201 (wave32, 64 KB
  LDS), untested.
