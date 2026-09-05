# sglang-flashnext-sm120

> ### ⚠️ Experimental fork — not production quality
>
> This is a personal **fork** of
> [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120).
> All of the original work — the patches, the tuning, the benchmarks below — is theirs.
>
> This fork exists only to make the launch scripts runnable outside the original author's
> machine (paths were hardcoded) and to fix a few startup failures found while bringing the
> stack up on a rented GPU container. It is scratch work from a debugging session, offered
> as-is: **not tested at production scale, not maintained, and carrying no guarantee that it
> matches upstream behaviour.** Use the upstream repo unless you specifically need these fixes.
>
> **The performance numbers below were measured by the upstream author on their own hardware.
> They have not been reproduced here.**


**Qwen3.8-Flash-Next (180B MoE, NVFP4) at 231 tok/s single-stream on a single RTX PRO 6000 Blackwell (96 GB, sm120).**

Patches, launch scripts and benchmarks for serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` at TP1
on the official SGLang `qwen4-main-squashed` branch — no Docker, no fork. Builds on the sm120
groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000),
then goes ~35-45% past its published numbers.

## Results

| | jpezzulli | this repo (temp 0.6) | this repo (greedy, lossless) |
|---|---|---|---|
| Decode, 1 stream | 171 tok/s | **231 median / 243 best** | 203 |
| Decode, 4 streams | 428 | **620–657** | 549 |
| Decode, 8 streams (optional 8-way config) | — | 758 | |
| Prefill (2.5K) | 10–12K | ~10.4K tok/s | |
| TTFT | — | ~135 ms | |

Launch profiles, selected as an argument to the unified launcher (`./serve <profile>`):

| profile | context window | concurrency | KV pool | decode C1 |
|---|---|---|---|---|
| `best` — interactive + agents | 262144 (native) | 4-way | ~572K tokens | **231 tok/s** |
| `single` — one huge session | **786432** (YaRN ×3) | 2-way | **~827K tokens** | 185 tok/s |
| `conc` — throughput *(added in this fork)* | 32768 (native) | 16-way | — | **unbenchmarked** |

`best` and `single` carry the upstream-validated settings unchanged. `conc` is new here and its
values are reasoned rather than measured — see [Changes in this fork](#changes-in-this-fork).

The long-context profile trades the fp8 dense-weight copies back for KV head-room and is
validated with needle retrieval at 653K-token depth (start / middle / end all pass).

Validated with greedy/needle/cached-prefix/GSM/code gates and 2.4M tokens of soak testing
(0 errors, flat VRAM/RAM).

## Changes in this fork

Scripts resolve the repo root from their own location, so a checkout works anywhere.
`BASE`, `REPO`, `TARGET_MODEL`, `CACHE_BASE`, `CARGO_BIN` and `MEMMAX` are all env-overridable;
nothing is hardcoded to a particular machine.

Three startup failures fixed:

- `sglang serve: error: unrecognized arguments: c` — `serve_single.sh` ran
  `bash -c 'exec bash serve.sh ...'` with no `$0`, so a literal `c` reached the CLI.
- `bad interpreter: No such file or directory` — a venv created at one path and then moved
  keeps absolute shebangs. The launcher probe now skips those and prints the recreate command.
- `systemd --user` is absent in most containers, so the unit could never start; it now falls
  back to `nohup` + pidfile.

`do_build.sh` locates `uv` rather than assuming a miniconda path, fails early if the venv is
missing, and scales build jobs to `nproc` (24 was sized for a 32-core host; each `cicc` is ~3 GB).

### Host packages

On a fresh container the Triton/FlashInfer JIT needs two things that are easy to miss:

```bash
apt-get install -y python3.10-dev    # else: fatal error: Python.h: No such file or directory
pip install ninja                    # else: FileNotFoundError: 'ninja'
```

Match `python3.X-dev` to the venv interpreter:
`.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['include'])"`

## Contents

```
patches/            six patches against sgl-project/sglang @ qwen4-main-squashed
serve               unified launcher: ./serve best|single|conc|list
Dockerfile          container build (UNTESTED) · docker-compose.yml · .dockerignore
scripts/            serve.sh (knobbed core) · serve_best.sh · serve_single.sh (compat shims)
                    bench_sglang.py · make_hot_tokens.py · do_build.sh
docs/               STATUS.md (ops guide) · PERF_CEILING.md (analysis + dead-ends)
results/            benchmark JSONs, baseline -> final
hot_tokens_64k.pt   FR-Spec draft-vocab map
```

## The optimizations

**Patches** (0001–0003 unblock sm120; 0004–0006 are the speed work, all env-gated):
1. `0001b` — RecoverSSM + WY output-only MTP verify on FlashInfer for sm120.
2. `0002` — FP8-KV tile dequant for the QSA sparse prefill (2× KV capacity).
3. `0003` — fp32 prefill state for the sm120 FlashInfer GDN kernel.
4. `0004` — Triton low-M GEMM: cuBLAS-under-graph-capture runs the decode projections at
   20–75% of DRAM bandwidth on sm120; this kernel reaches ~90%.
5. `0005` — W8A16 fp8 weight-only serving of the dense bf16 stack (85% of per-step traffic,
   untouched by the NVFP4 checkpoint). Runtime-only: the checkpoint is never modified.
6. `0006` — same fp8 treatment for the HyperConnection mix and the lm_head (also halves
   every MTP draft step's logits).

**Config levers** (in `scripts/serve.sh`, each documented inline with its measured ladder):
- Relaxed MTP acceptance `0.3` (C1 179 → 231; exact at temp 0, set `1.0` for lossless sampling).
- FR-Spec: draft head scores a 64K hot-token subset of the 248K vocab (verify stays exact).
- 8-way concurrency: `--max-mamba-cache-size` must be ~6× max-running-requests or the
  speculative CUDA graphs silently cap at bs=4 (8-way used to run *slower* than 4-way).

## Reproduce

```bash
# model checkpoint (~135 GB; the ~50 GB PLE n-gram table is served from host RAM)
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir Qwen3.8-Flash-Next-NVFP4
# note: hf_xet can stall on the largest shards; scripts/serve.sh's docs and
# docs/STATUS.md describe the curl fallback that resumes reliably.

git clone -b qwen4-main-squashed https://github.com/sgl-project/sglang sglang-official
cd sglang-official && bash ../scripts/do_build.sh
git apply ../patches/0002-fp8-qsa-tile-dequant.patch --exclude='test/*'
git apply ../patches/0003-sm120-fp32-prefill-state.patch
git apply ../patches/0001b-recoverssm-wy-sm120-PORTED.patch
git am    ../patches/0004*.patch ../patches/0005*.patch ../patches/0006*.patch
# paths resolve from the repo location - nothing to edit. Then:
./serve best                   # OpenAI API on :8001, ~5 min to ready
./serve list                   # show all profiles and their settings
```

Single-GPU-with-display safety knobs (learned the hard way): keep `--cuda-graph-max-bs`
small, cap JIT compilation with `MAX_JOBS=4`, run under a systemd `MemoryMax` cage.
Details and every explored dead-end: `docs/`.

## Docker (untested)

> **This has never been built or run.** It encodes the manual steps below plus the two host
> packages that are easy to miss (`python3.10-dev`, `ninja`). Expect to fix something on the
> first `docker build`.

Weights are **not** baked into the image (~135 GB) — mount them:

```bash
./scripts/fetch_model.sh /data/Qwen3.8-Flash-Next-NVFP4     # ~135 GB, resumable

docker build -t qwen-flashnext .
docker run --gpus all -p 8001:8001 -e FOREGROUND=1 \
  -v /data/Qwen3.8-Flash-Next-NVFP4:/opt/qwen/models/Qwen3.8-Flash-Next-NVFP4:ro \
  -v qwen-cache:/opt/qwen/cache \
  qwen-flashnext best

# or
MODEL_DIR=/data/Qwen3.8-Flash-Next-NVFP4 PROFILE=conc docker compose up
```

Two things that matter:

- **`-v qwen-cache:/opt/qwen/cache`** — FlashInfer autotune JIT-compiles kernels on first
  start (~20 min). A persistent cache volume means you pay that once, not per container. It
  cannot be baked into the image, since compiling needs a GPU at build time.
- **`FOREGROUND=1`** — makes `./serve` exec the server instead of detaching. Without it a
  backgrounded PID 1 exits and takes the container with it. The compose file sets it already.

Requires the NVIDIA Container Toolkit on the host; the driver comes from the host, not the
image. The base is CUDA 13.0 `devel` (not `runtime`) because `nvcc` is needed for the runtime
kernel JIT, and it matches the `cu130` wheel index `do_build.sh` installs from.

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang), branch `qwen4-main-squashed`
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — the sm120
  RecoverSSM/WY and fp8-QSA work this repo ports and builds on
