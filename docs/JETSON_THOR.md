# Strata on the NVIDIA Jetson AGX Thor (Linux, ARM64)

The Jetson AGX Thor developer kit carries the Jetson T5000 module: a Blackwell GPU (compute capability 11.0, 2,560 CUDA cores) and 14 Arm
CPU cores that share 128 GB of memory (122 GB visible to Linux). The same module is inside the Advantech MIC-743. This page says how
Qwen3.8-Flash-Next IQ3_S was built and run on one Thor, and what it measured. Tested on 2026-11-02 with Jetson Linux R39.2 (JetPack 7.2),
driver 595.78, CUDA 13.2, Strata at upstream `c62d5e3`. Nothing here was tried on another Jetson.

## What works, and what does not

- **Works:** Qwen3.8-Flash-Next IQ3_S (ISTA-DASLab GSQ-RCO, two GGUF shards, 78 GB) at 128K context with an int8 KV cache, the
  OpenAI-compatible server, and GitHub Copilot desktop as a client (custom endpoint, Chat Completions).
- **Not tried:** the NVFP4 pack, DeepSeek-V4.1-Flash, picture reading, the Terminal's engine packages (the engine was compiled
  here), more than one chat slot.

## Build the engine

The feed's `linux-aarch64` engine targets the GB10 (`sm_121`). This Thor ran an engine compiled for its own GPU (`sm_110`):

```sh
python3 -m venv .venv-build
.venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
.venv-build/bin/pip install ninja cmake
PATH="$PWD/.venv-build/bin:/usr/local/cuda/bin:$PATH" PYTHON="$PWD/.venv-build/bin/python" \
  ARCHS=110 BUILD_DSV41=0 BUILD_VISION=0 JOBS=12 sh spark/build_engine.sh
cp -r out-spark/engine ./engine
```

`ARCHS=110` is the only Thor-specific part. `BUILD_DSV41=0 BUILD_VISION=0` skip the DeepSeek and picture engines, which were not used.
cmake and ninja came from pip, inside the venv.

## Set up the model

```sh
./setup.sh --setup --family qwen --model IQ3_S --context 131072 --vision no --yes --no-start \
  --data-dir ~/Strata-data --gguf-dir ~/Strata-data/gguf/qwen38-iq3s \
  --host 0.0.0.0 --port 8081 --api-key "$(cat ~/.strata_api_key)" --parallel 1
```

Setup does not recognise the T5000 as unified memory, so it leaves out the settings it writes for the DGX Spark. Add them to the
`args` of `strata-iq3_s.json` by hand (and change `--prefill auto` to `--prefill auto:32768`):

```json
"--mmap-experts", "--mtp-q4", "all", "--mtp-hnorm", "stream"
```

The reasons are the ones in [DGX_SPARK_LINUX.md](DGX_SPARK_LINUX.md): with `--mmap-experts` the experts are read from the model files
in place instead of a second copy in RAM. Keep the API key in a file only its owner can read (`chmod 600`).

## Measured

| Request | Prompt | Prompt reading | Output |
|---|---|---|---|
| Short question | 55 tokens | about 1 s | 42.8 tok/s, 5.2 s in all |
| Long document | 56,553 tokens | 52.7 s (1,073 tok/s) | 36.6 tok/s |
| GitHub Copilot desktop, first turn of a new chat | 38,526 tokens | 41 s (940 tok/s) | first answer 43 s after sending |
| Copilot writes a 237-line `index.html` | cached prefix | about 1 s | 2,012 tokens at 47.9 tok/s |

Copilot's later turns in the same chat start in about a second, because the conversation prefix stays in the prompt cache. The whole
task (write the page, open it in Copilot's browser panel, test it) took 2 min 28 s; a follow-up edit with its own test took 1 min 32 s.
Memory in use, for the whole machine with the model loaded and serving: 63 GB of 122 GB (`tegrastats`).

The thinking budget is on by default. A client that sends a small `max_tokens` can get a reply that ends while still thinking and has
no answer; give it room or set `reasoning_budget_tokens`.

## GitHub Copilot desktop

Settings → Model providers → Add provider → Custom endpoint. Base URL `http://<thor>:8081/v1`, Wire API `completions`, the API key
from `~/.strata_api_key`. Add the model by ID (the server answers under its own model name whatever ID the client sends), then choose
it in the chat's model menu.
