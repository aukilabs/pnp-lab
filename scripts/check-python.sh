#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required for the isolated Python test" >&2
  exit 1
fi

if [[ -n "${MATURIN_BIN:-}" ]]; then
  MATURIN=("$MATURIN_BIN")
elif command -v maturin >/dev/null 2>&1; then
  MATURIN=(maturin)
elif command -v uvx >/dev/null 2>&1; then
  MATURIN=(uvx --from 'maturin==1.14.1' maturin)
else
  echo "error: install maturin or uv/uvx to test the Python wheel" >&2
  exit 1
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/pnplab-python.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

uv venv --quiet --system-site-packages --python "${PYTHON:-python3}" "$TMP/venv"
PYTHON_BIN="$TMP/venv/bin/python"
WHEEL_DIR="$TMP/wheels"
mkdir -p "$WHEEL_DIR"

"${MATURIN[@]}" build \
  --manifest-path bindings/python/Cargo.toml \
  --release \
  --interpreter "$PYTHON_BIN" \
  --out "$WHEEL_DIR"

uv pip install --quiet --no-deps --python "$PYTHON_BIN" "$WHEEL_DIR"/*.whl

if ! "$PYTHON_BIN" -c 'import numpy, pytest' >/dev/null 2>&1; then
  uv pip install --quiet --python "$PYTHON_BIN" 'numpy>=1.24' 'pytest>=8'
fi

"$PYTHON_BIN" -m pytest -q bindings/python/tests
