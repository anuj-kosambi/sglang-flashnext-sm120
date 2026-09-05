#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 at TP1 on the RTX PRO 6000 (sm120), official sglang qwen4-main-squashed branch.
# SAFE defaults; override via env to push toward jpezzulli's optimized profile.
set -euo pipefail
# Repo root: resolved from this script's own location, so the checkout works anywhere.
# Override with BASE=/path/to/repo for an out-of-tree layout.
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
_SCRIPT_DIR="$(cd -P "$(dirname "$_self")" && pwd)"
BASE="${BASE:-$(cd -P "$_SCRIPT_DIR/.." && pwd)}"
REPO="${REPO:-$BASE/sglang-official}"
# Pick an sglang launcher whose shebang interpreter actually exists. A venv that was created
# elsewhere and then moved keeps absolute paths in its shebangs -> "bad interpreter". Skip those.
_usable_sglang() {
  local f="$1" shb interp
  [[ -n "$f" && -x "$f" ]] || return 1
  shb="$(head -c 256 "$f" 2>/dev/null | head -1)"
  case "$shb" in
    '#!'*) interp="${shb#\#!}"; interp="${interp%% *}"
           [[ -x "$interp" ]] || { echo "WARN: $f -> missing interpreter $interp (moved/stale venv), skipping" >&2; return 1; } ;;
  esac
  return 0
}
SGLANG="${SGLANG:-}"
if [[ -z "$SGLANG" ]]; then
  for c in "$REPO/.venv/bin/sglang" "$BASE/.venv/bin/sglang" "$(command -v sglang || true)"; do
    _usable_sglang "$c" && { SGLANG="$c"; break; }
  done
fi
if [[ -z "$SGLANG" ]]; then
  echo "ERROR: no usable sglang (looked in $REPO/.venv, $BASE/.venv, PATH)." >&2
  echo "  A 'bad interpreter' shebang means the venv was created at another path and moved." >&2
  echo "  Venvs are not relocatable - recreate it:" >&2
  echo "    uv venv $REPO/.venv --python 3.12 && bash $BASE/scripts/do_build.sh" >&2
  exit 1
fi
TARGET_MODEL="${TARGET_MODEL:-$BASE/models/Qwen3.8-Flash-Next-NVFP4}"
CACHE_BASE="${CACHE_BASE:-$BASE/cache}"
PORT="${PORT:-8001}"

# ---- tunable knobs (safe defaults) ----
CTX="${CTX:-32768}"                 # jpezzulli optimized: 524288 (YaRN factor 2)
MEMFRAC="${MEMFRAC:-0.85}"          # optimized: 0.981
MAXREQ="${MAXREQ:-4}"
LINEAR_BACKEND="${LINEAR_BACKEND:-triton}"   # safe: triton;  perf: flashinfer (sm120)
SPEC="${SPEC:-1}"                   # 1 = enable native NEXTN MTP, 0 = disable
HICACHE="${HICACHE:-0}"             # 0 = off (no NIXL);  1 = enable hierarchical cache
CUDAGRAPH_MAXBS="${CUDAGRAPH_MAXBS:-4}"   # only capture graphs up to this bs (must be >= MAXREQ). Capturing 1..256 OOMs the display GPU.
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"     # offload N GB of weights to host RAM for extra VRAM headroom (costs throughput)
AUTOTUNE="${AUTOTUNE:-0}"                  # 0 = disable flashinfer autotune (avoids the parallel-cicc RAM storm); 1 = enable (only with low MAX_JOBS)
KVDTYPE="${KVDTYPE:-auto}"                 # auto (fp16/bf16, triton-safe) or fp8_e4m3 (needs flashinfer backend; crashes triton GDN/QSA)
# FlashInfer GDN on sm120 (unpatched official branch): server_args REQUIRES bf16 SSM state on SM100+, but the
# radix-cache state-checkpoint plan (built only under --mamba-radix-cache-strategy extra_buffer + track-interval)
# demands fp32 on sm120 -> contradiction = jpezzulli's 280825c3e2 patch. Sidestep: bf16 SSM + no_buffer (no
# track mask -> no checkpoint plan). Cost: no GDN prefix-STATE caching across turns (TTFT on multi-turn), not decode speed.
SSM_DTYPE="${SSM_DTYPE:-bfloat16}"
# NOTE: no_buffer is NOT viable for this model (compressed QSA forces page-size 64; no_buffer needs page 1).
# FlashInfer GDN on sm120 is therefore blocked on the unpatched branch (sm120 DSL kernel is fp32-state, sglang demands
# bf16 for flashinfer decode, and extra_buffer tracking needs checkpoints) -> needs jpezzulli's WY-output-only patch.
# Known-good: LINEAR_BACKEND=triton. Keep extra_buffer always.
MAMBA_RADIX="${MAMBA_RADIX:-extra_buffer}"

mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,flashinfer,sglang/jit}
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDACXX="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export CC="${CC:-gcc}" CXX="${CXX:-g++}" CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX:-g++}}" TORCH_CUDA_ARCH_LIST=12.0
# CARGO_BIN: only prepended if it exists (rust toolchain is optional at serve time).
CARGO_BIN="${CARGO_BIN:-$HOME/.cargo/bin}"
[[ -d "$CARGO_BIN" ]] && export PATH="$CARGO_BIN:$PATH"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor" TRITON_CACHE_DIR="$CACHE_BASE/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
# CAP kernel-JIT parallelism: unbounded cicc compilers (nproc=32 x ~3GB each) + 50GB PLE => 120GB RAM + swap thrash => crash.
export MAX_JOBS="${MAX_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-4}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

args=(
  serve --model-path "$TARGET_MODEL" --load-format safetensors
  --served-model-name "${SERVED_NAME:-pennyroyal}" --host 0.0.0.0 --port "$PORT" --tp 1
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype "$KVDTYPE"
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --page-size 64 --max-running-requests "$MAXREQ" --chunked-prefill-size 4096
  --cuda-graph-max-bs "$CUDAGRAPH_MAXBS"
  --mamba-ssm-dtype "$SSM_DTYPE" --max-mamba-cache-size "${MAMBA_CACHE:-24}"
  --mamba-radix-cache-strategy "$MAMBA_RADIX"
  --linear-attn-decode-backend "$LINEAR_BACKEND" --linear-attn-prefill-backend "$LINEAR_BACKEND"
  --ple-offload-embedding --trust-remote-code
  --chat-template "$TARGET_MODEL/chat_template.jinja"
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-request-time-stats-logging --enable-metrics --watchdog-timeout 1800
)
[[ "$CPU_OFFLOAD_GB" -gt 0 ]] && args+=( --cpu-offload-gb "$CPU_OFFLOAD_GB" )
[[ "$AUTOTUNE" == "1" ]] || args+=( --disable-flashinfer-autotune )   # default: autotune OFF (its parallel cicc JIT storm OOMs host RAM)
[[ "$MAMBA_RADIX" == "extra_buffer" ]] && args+=( --mamba-track-interval 64 )   # state tracking only exists for extra_buffer
# RecoverSSM / WY output-only MTP verify (ported jpezzulli 280825c3e2, branch sm120-wy). jpezzulli: "none".
[[ -n "${GDN_MTP_CACHE_MODE:-}" ]] && args+=( --gdn-mtp-cache-mode "$GDN_MTP_CACHE_MODE" )
# MTP depth: jpezzulli = 3 steps / 4 draft tokens - and that is the MAXIMUM on this model/branch:
# SPEC_DRAFT must be <= the QSA compress ratio (4): "Qwen QSA requires speculative_num_draft_tokens <= the QSA compress
# ratio (4): the pending index-key ring holds one group" (SPEC_STEPS=4 -> 5 draft tokens fails at startup, tested 2026-08-30).
SPEC_STEPS="${SPEC_STEPS:-3}"; SPEC_DRAFT="${SPEC_DRAFT:-$((SPEC_STEPS+1))}"
# Relaxed MTP acceptance (2026-08-31): force-accept a draft token the target gives >= this prob
# instead of coin-flipping it. LOSSY at temp>0 (sharpens toward high-prob tokens; exact at temp 0).
# Ladder measured at temp 0.6 (C1 median): 1.0 lossless = 179 - 0.5 = 203 - 0.3 = 231 tok/s.
# 0.3 passes greedy/needles/8x GSM/code/French gates; raise toward 1.0 if outputs feel dull/sharp.
SPEC_ACCEPT_SINGLE="${SPEC_ACCEPT_SINGLE:-0.3}"; SPEC_ACCEPT_ACC="${SPEC_ACCEPT_ACC:-0.3}"
# FR-Spec (2026-08-31): draft lm_head scores only a 64K hot-token subset (vocab is 248K) ->
# draft logits ~4x cheaper; verification stays exact so only draft quality could dip (accept
# length measured unchanged, gates + French pass). C1 200.8->219.4, C4 541->592 (temp 0.6).
# Set SPEC_TOKEN_MAP=none to disable. Map = 32K base BPE + top code-corpus tokens + specials.
SPEC_TOKEN_MAP="${SPEC_TOKEN_MAP:-$BASE/hot_tokens_64k.pt}"
[[ "$SPEC" == "1" ]] && args+=( --speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk 1 --speculative-num-draft-tokens "$SPEC_DRAFT" --speculative-draft-model-quantization unquant
  --speculative-accept-threshold-single "$SPEC_ACCEPT_SINGLE" --speculative-accept-threshold-acc "$SPEC_ACCEPT_ACC" )
[[ "$SPEC" == "1" && "$SPEC_TOKEN_MAP" != "none" && -f "$SPEC_TOKEN_MAP" ]] && args+=( --speculative-token-map "$SPEC_TOKEN_MAP" )
[[ "$HICACHE" == "1" ]] && args+=( --enable-hierarchical-cache --hicache-size 32
  --hicache-host-memory-mode cache --hicache-write-policy write_through --hicache-io-backend kernel )

# Long-context YaRN rope override (factor = CTX/262144 for CTX beyond the native window).
[[ -n "${ROPE_OVERRIDE:-}" ]] && args+=( --json-model-override-args "$ROPE_OVERRIDE" )

# Extra CLI args are appended last; argparse last-wins so they can override anything above.
# Guard: only forward args that look like real flags (or values following one). A stray bare
# word - e.g. the literal "c" leaked by a `systemd-run ... -- bash -c '...'` wrapper - would
# otherwise reach argparse as "unrecognized arguments: c".
extra=(); _prev_was_flag=0
for a in "$@"; do
  case "$a" in
    -*) extra+=( "$a" ); _prev_was_flag=1 ;;
    *)  if [[ "$_prev_was_flag" == "1" ]]; then extra+=( "$a" ); _prev_was_flag=0
        else echo "WARN: dropping stray non-flag argument: '$a'" >&2; fi ;;
  esac
done
echo "sglang ${args[*]} ${extra[*]-}"
exec "$SGLANG" "${args[@]}" ${extra[@]+"${extra[@]}"}
