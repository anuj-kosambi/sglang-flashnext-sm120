#!/usr/bin/env bash
# SINGLE-SESSION LONG-CONTEXT PROFILE - Qwen3.8-Flash-Next NVFP4, TP1, RTX PRO 6000 (sm120).
# Trades the fp8 dense-weight copies (~3.6 GB VRAM, ~20% decode speed) for the biggest
# possible KV pool, and extends rope with YaRN factor 3.0 -> 786,432-token context window.
# Use serve_best.sh instead for the max-throughput / 8-way concurrency profile.
#   ./serve_single.sh            # start (unit qwen-sglang, :8001, model pennyroyal)
#   systemctl --user stop qwen-sglang
set -euo pipefail
# Repo root: resolved from this script's own location, so the checkout works anywhere.
# Override with BASE=/path/to/repo for an out-of-tree layout.
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
_SCRIPT_DIR="$(cd -P "$(dirname "$_self")" && pwd)"
BASE="${BASE:-$_SCRIPT_DIR}"
cd "$BASE"

# systemd --user does not exist in most containers (no user session bus). Fall back to a plain
# background process with the same env when it is unavailable.
if systemctl --user show-environment >/dev/null 2>&1; then USE_SYSTEMD=1; else USE_SYSTEMD=0; fi

ROPE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":3.0,"original_max_position_embeddings":262144}}}'

# MEMMAX: host-RAM cap for the unit (e.g. 112G). Unset = no cap; set it to match your box.
MEMMAX="${MEMMAX:-}"

ENVS=(
  BASE="$BASE"
  MEMFRAC=0.98 CTX="${CTX:-786432}" MAXREQ=2
  LINEAR_BACKEND=flashinfer SSM_DTYPE=bfloat16 MAMBA_RADIX=extra_buffer
  KVDTYPE=fp8_e4m3 SPEC=1 HICACHE=0
  GDN_MTP_CACHE_MODE=none
  SGLANG_SM120_LOWM_FP8_WEIGHT=0
  CUDAGRAPH_MAXBS=2 MAMBA_CACHE=12 CPU_OFFLOAD_GB=0
  ROPE_OVERRIDE="$ROPE"
  AUTOTUNE=1 MAX_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=2
)

mkdir -p logs; rm -f logs/serve.log

if [[ "$USE_SYSTEMD" == "1" ]]; then
  systemctl --user stop qwen-sglang 2>/dev/null || true
  for i in $(seq 1 40); do
    [ "$(systemctl --user show qwen-sglang -p LoadState --value 2>/dev/null)" = "not-found" ] && break
    systemctl --user reset-failed qwen-sglang 2>/dev/null || true; sleep 2
  done
  setenvs=(); for kv in "${ENVS[@]}"; do setenvs+=( --setenv="$kv" ); done
  systemd-run --user --unit=qwen-sglang \
    ${MEMMAX:+--property=MemoryMax=$MEMMAX} --property=MemorySwapMax=0 \
    "${setenvs[@]}" \
    --working-directory="$BASE" \
    -- bash -c 'exec bash "$0" > logs/serve.log 2>&1' "$BASE/scripts/serve.sh"
else
  pkill -f 'sglang.*--served-model-name' 2>/dev/null || true
  sleep 2
  env "${ENVS[@]}" nohup bash "$BASE/scripts/serve.sh" > "$BASE/logs/serve.log" 2>&1 &
  echo $! > "$BASE/logs/serve.pid"
  echo "pid $(cat "$BASE/logs/serve.pid") (no systemd --user; stop with: kill \$(cat $BASE/logs/serve.pid))"
fi

echo "qwen-sglang started (single-session long-context profile). Wait for: curl -s http://127.0.0.1:8001/health -> 200"
echo "log: $BASE/logs/serve.log"
