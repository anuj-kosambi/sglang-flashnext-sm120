#!/usr/bin/env bash
set -uo pipefail
# Repo root: resolved from this script's own location, so the checkout works anywhere.
# Override with BASE=/path/to/repo for an out-of-tree layout.
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
_SCRIPT_DIR="$(cd -P "$(dirname "$_self")" && pwd)"
BASE="${BASE:-$(cd -P "$_SCRIPT_DIR/.." && pwd)}"
REPO="${REPO:-$BASE/sglang-official}"
cd "$REPO"

# Venv is NOT created here - do_build only activates it. Create it first if missing:
#   uv venv "$REPO/.venv" --python 3.12     (or: python3 -m venv "$REPO/.venv")
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: no venv at $REPO/.venv - create it first (uv venv $REPO/.venv)" >&2
  exit 1
fi
source .venv/bin/activate

# uv: prefer one already on PATH / in the venv; fall back to a conda copy if present.
UV="${UV:-}"
if [[ -z "$UV" ]]; then
  for c in "$REPO/.venv/bin/uv" "$BASE/.venv/bin/uv" "$(command -v uv || true)" \
           "$HOME/.local/bin/uv" "$HOME/miniconda3/bin/uv" /opt/conda/bin/uv; do
    [[ -n "$c" && -x "$c" ]] && { UV="$c"; break; }
  done
fi
if [[ -z "$UV" ]]; then
  echo "ERROR: uv not found. Install it:  pip install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
  exit 1
fi
echo "uv: $UV"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDACXX="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
# gcc-13 was the author's box; most containers only ship plain gcc. Override CC/CXX if you have versioned ones.
export CC="${CC:-gcc}" CXX="${CXX:-g++}" CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX:-g++}}" TORCH_CUDA_ARCH_LIST=12.0
CARGO_BIN="${CARGO_BIN:-$HOME/.cargo/bin}"
[[ -d "$CARGO_BIN" ]] && export PATH="$CARGO_BIN:$PATH"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
# Author used 24 (32-core box). Each cicc is ~3GB RAM; scale to this machine and cap so the build cannot OOM.
_NPROC="$(nproc 2>/dev/null || echo 8)"
_J="${BUILD_JOBS:-$(( _NPROC < 16 ? _NPROC : 16 ))}"
export MAX_JOBS="$_J" CMAKE_BUILD_PARALLEL_LEVEL="$_J" CARGO_BUILD_JOBS="$_J"
echo "build jobs: $_J (nproc=$_NPROC)"

echo "=== install start $(date) ==="
"$UV" pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  -e python
echo "=== install exit=$? $(date) ==="
python -c "import sglang; print('sglang', sglang.__version__)" 2>&1 | tail -2
