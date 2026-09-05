#!/usr/bin/env bash
# Compatibility shim -> ./serve single
set -euo pipefail
_self="${BASH_SOURCE[0]}"
while [[ -L "$_self" ]]; do _d="$(cd -P "$(dirname "$_self")" && pwd)"; _self="$(readlink "$_self")"; [[ "$_self" != /* ]] && _self="$_d/$_self"; done
exec "$(cd -P "$(dirname "$_self")" && pwd)/serve" single "$@"
