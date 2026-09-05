#!/usr/bin/env bash
# Compatibility shim: profiles now live in the unified launcher at the repo root.
#   ./serve best [extra sglang args...]
set -euo pipefail
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
_DIR="$(cd -P "$(dirname "$_self")" && pwd)"
exec "${BASE:-$(cd -P "$_DIR/.." && pwd)}/serve" best "$@"
