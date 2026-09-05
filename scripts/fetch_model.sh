#!/usr/bin/env bash
# Download the NVFP4 checkpoint (~135 GB) into MODEL_DIR.
# The README warns hf_xet can stall on the largest shards; HF_HUB_DISABLE_XET=1
# falls back to plain HTTP, which resumes reliably.
#   ./scripts/fetch_model.sh [target-dir]
set -euo pipefail
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
BASE="${BASE:-$(cd -P "$(dirname "$_self")/.." && pwd)}"
DEST="${1:-${MODEL_DIR:-$BASE/models/Qwen3.8-Flash-Next-NVFP4}}"
REPO_ID="${REPO_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
command -v hf >/dev/null || { echo "ERROR: 'hf' not found - pip install huggingface_hub[cli]" >&2; exit 1; }
mkdir -p "$DEST"
echo "downloading $REPO_ID -> $DEST (~135 GB; resumable, safe to re-run)"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" hf download "$REPO_ID" --local-dir "$DEST"
echo "done. sanity check:"
ls -1 "$DEST" | head -5
[[ -f "$DEST/chat_template.jinja" ]] && echo "chat_template.jinja present" || echo "WARN: chat_template.jinja missing - serve.sh passes it with --chat-template"
