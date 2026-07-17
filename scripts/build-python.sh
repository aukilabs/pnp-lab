#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${MATURIN_BIN:-}" ]]; then
  MATURIN=("$MATURIN_BIN")
elif command -v maturin >/dev/null 2>&1; then
  MATURIN=(maturin)
elif command -v uvx >/dev/null 2>&1; then
  MATURIN=(uvx --from 'maturin==1.14.1' maturin)
else
  echo "error: install maturin or uv/uvx to build the Python wheel" >&2
  exit 1
fi

OUT_DIR="$ROOT/bindings/python/dist"
mkdir -p "$OUT_DIR"
"${MATURIN[@]}" build \
  --manifest-path bindings/python/Cargo.toml \
  --release \
  --out "$OUT_DIR" \
  "$@"
