#!/usr/bin/env bash
# =============================================================================
#  Qwen3.8-Flash-Next NVFP4 · TP1 · RTX PRO 6000 Blackwell (sm120)
#  Single launcher: profile selection + the full knobbed sglang invocation.
#
#     ./serve.sh <profile> [extra sglang args...]
#
#  PROFILES
#    best     4-way,  262144 ctx (native),  fp8 weight copies.  C1 ~231 tok/s  [upstream-validated]
#    single   2-way,  786432 ctx (YaRN x3), max KV pool.        C1 ~185 tok/s  [upstream-validated]
#    conc    16-way,   32768 ctx,           throughput-oriented. UNVERIFIED (see below)
#    list     print the profile table and exit
#
#  EXAMPLES
#    ./serve.sh best                  ./serve.sh conc --port 9000
#    MAXREQ=8 ./serve.sh conc         FOREGROUND=1 ./serve.sh best     # container mode
#
#  Every value is an env override. Paths resolve from this script's location;
#  BASE=/path/to/repo overrides. FOREGROUND=1 execs the server instead of
#  detaching - required in a container, where a backgrounded PID 1 exits.
#
#  WARNING on 'conc': its values are reasoned, NOT benchmarked. Batch-16 cuda-graph
#  capture is the likely first failure; lower MEMFRAC or CUDAGRAPH_MAXBS if it OOMs.
#  Validate with scripts/bench_sglang.py. The best/single numbers are upstream's,
#  measured on their hardware and not reproduced here.
# =============================================================================
set -euo pipefail

_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
_SCRIPT_DIR="$(cd -P "$(dirname "$_self")" && pwd)"
BASE="${BASE:-$_SCRIPT_DIR}"
REPO="${REPO:-$BASE/sglang-official}"   # sglang source tree (venv usually lives here)
UNIT="${UNIT:-qwen-sglang}"
MEMMAX="${MEMMAX:-}"          # e.g. 112G. Unset = no host-RAM cap.

PROFILE="${1:-}"; [[ $# -gt 0 ]] && shift || true

_yarn() {  # YaRN rope override; factor = CTX / 262144 (native window)
  local factor="$1"
  printf '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":%s,"original_max_position_embeddings":262144}}}' "$factor"
}

case "$PROFILE" in
  best)
    ENVS=(
      MEMFRAC="${MEMFRAC:-0.95}" CTX="${CTX:-262144}" MAXREQ="${MAXREQ:-4}"
      CUDAGRAPH_MAXBS="${CUDAGRAPH_MAXBS:-4}" MAMBA_CACHE="${MAMBA_CACHE:-24}"
      SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1
    ) ;;
  single)
    ENVS=(
      MEMFRAC="${MEMFRAC:-0.98}" CTX="${CTX:-786432}" MAXREQ="${MAXREQ:-2}"
      CUDAGRAPH_MAXBS="${CUDAGRAPH_MAXBS:-2}" MAMBA_CACHE="${MAMBA_CACHE:-12}"
      SGLANG_SM120_LOWM_FP8_WEIGHT=0
      ROPE_OVERRIDE="${ROPE_OVERRIDE:-$(_yarn 3.0)}"
    ) ;;
  conc)
    # UNVERIFIED. 32768 is inside the native window, so no YaRN rope override is needed.
    # CUDAGRAPH_MAXBS must be >= MAXREQ. MAMBA_CACHE must be ~6x MAXREQ (96 here): upstream
    # measured that a smaller value silently caps the speculative CUDA graphs at bs=4, making
    # 8-way run SLOWER than 4-way. It fails quietly, so do not lower it without benchmarking.
    # MEMFRAC is below 'best' deliberately: 16 captured cuda graphs need headroom.
    ENVS=(
      MEMFRAC="${MEMFRAC:-0.90}" CTX="${CTX:-32768}" MAXREQ="${MAXREQ:-16}"
      CUDAGRAPH_MAXBS="${CUDAGRAPH_MAXBS:-16}" MAMBA_CACHE="${MAMBA_CACHE:-96}"
      CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
      SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1
    ) ;;
  list)
    printf '%-8s %-8s %-6s %-9s %-8s %s\n' PROFILE CTX MAXREQ CUDAGRAPH MEMFRAC NOTE
    printf '%-8s %-8s %-6s %-9s %-8s %s\n' best   262144 4  4  0.95 'validated upstream, ~231 tok/s C1'
    printf '%-8s %-8s %-6s %-9s %-8s %s\n' single 786432 2  2  0.98 'validated upstream, ~185 tok/s C1'
    printf '%-8s %-8s %-6s %-9s %-8s %s\n' conc    32768 16 16 0.90 'UNVERIFIED - benchmark before use'
    exit 0 ;;
  ""|-h|--help|help)
    sed -n '2,22p' "$_self"; echo; echo "Profiles: best | single | conc | list"; exit 0 ;;
  *)
    echo "ERROR: unknown profile '$PROFILE' (want: best | single | conc | list)" >&2; exit 1 ;;
esac

# Settings shared by every profile (a profile above may override any of them).
ENVS+=(
  BASE="$BASE"
  LINEAR_BACKEND="${LINEAR_BACKEND:-flashinfer}" SSM_DTYPE="${SSM_DTYPE:-bfloat16}"
  MAMBA_RADIX="${MAMBA_RADIX:-extra_buffer}" KVDTYPE="${KVDTYPE:-fp8_e4m3}"
  SPEC="${SPEC:-1}" HICACHE="${HICACHE:-0}" GDN_MTP_CACHE_MODE="${GDN_MTP_CACHE_MODE:-none}"
  CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
  AUTOTUNE="${AUTOTUNE:-1}" MAX_JOBS="${MAX_JOBS:-4}"
  FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-4}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
)

# Apply the profile into this shell so the core defaults below see it. Profile values are
# themselves ${VAR:-default}, so anything already set in the environment still wins.
for _kv in "${ENVS[@]}"; do export "$_kv"; done
EXPORTS=( "${ENVS[@]}" )   # reused verbatim for the systemd --setenv list



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
  --page-size 64 --max-running-requests "$MAXREQ" --chunked-prefill-size "${CHUNKED_PREFILL:-4096}"
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
# Guard: only forward args that look like real flags (or values following one), so a stray
# bare word cannot reach argparse as "unrecognized arguments: c".
extra=(); _prev_was_flag=0
for a in "$@"; do
  case "$a" in
    -*) extra+=( "$a" ); _prev_was_flag=1 ;;
    *)  if [[ "$_prev_was_flag" == "1" ]]; then extra+=( "$a" ); _prev_was_flag=0
        else echo "WARN: dropping stray non-flag argument: '$a'" >&2; fi ;;
  esac
done


# ---------------------------------------------------------------------------
#  Launch
# ---------------------------------------------------------------------------
mkdir -p "$BASE/logs"; rm -f "$BASE/logs/serve.log"
echo "profile: $PROFILE   base: $BASE"
echo "sglang ${args[*]} ${extra[*]-}"

# FOREGROUND=1 wins over everything: in a container we must BE the process, even if a
# session bus happens to exist. Otherwise systemd --user if available, else nohup.
if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "running in foreground (FOREGROUND=1); logging to stdout and $BASE/logs/serve.log"
  exec "$SGLANG" "${args[@]}" ${extra[@]+"${extra[@]}"} 2>&1 | tee "$BASE/logs/serve.log"
elif systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user stop "$UNIT" 2>/dev/null || true
  for i in $(seq 1 40); do
    [ "$(systemctl --user show "$UNIT" -p LoadState --value 2>/dev/null)" = "not-found" ] && break
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true; sleep 2
  done
  setenvs=(); for kv in "${EXPORTS[@]}"; do setenvs+=( --setenv="$kv" ); done
  systemd-run --user --unit="$UNIT" \
    ${MEMMAX:+--property=MemoryMax=$MEMMAX} --property=MemorySwapMax=0 \
    "${setenvs[@]}" --working-directory="$BASE" \
    -- "$SGLANG" "${args[@]}" ${extra[@]+"${extra[@]}"}
  echo "started as user unit '$UNIT' ($(systemctl --user is-active "$UNIT"))"
else
  pkill -f "sglang.*--served-model-name" 2>/dev/null || true
  sleep 2
  nohup "$SGLANG" "${args[@]}" ${extra[@]+"${extra[@]}"} > "$BASE/logs/serve.log" 2>&1 &
  echo $! > "$BASE/logs/serve.pid"
  echo "started pid $(cat "$BASE/logs/serve.pid") (no systemd --user; stop: kill \$(cat $BASE/logs/serve.pid))"
fi
echo "ready when: curl -s http://127.0.0.1:${PORT:-8001}/health -> 200   (first ever start ~20 min autotune)"
echo "log: $BASE/logs/serve.log"
