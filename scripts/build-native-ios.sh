#!/usr/bin/env bash
# Build libpeyote_pnp_ffi.a for device + simulator and package
# PnpRust.xcframework into bindings/expo-pnp/ios/.
#
# Requires: Xcode (xcodebuild), cbindgen, Rust targets:
#   aarch64-apple-ios
#   aarch64-apple-ios-sim
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${PNP_MOBILE_PROFILE:-release}"
OUT_XCFW="$ROOT/bindings/expo-pnp/ios/PnpRust.xcframework"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/pnp-ios.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

generate_header() {
  if ! command -v cbindgen >/dev/null 2>&1; then
    echo "error: cbindgen not found. Install with: cargo install cbindgen" >&2
    exit 1
  fi

  (
    cd crates/pnp-ffi
    mkdir -p include
    cbindgen --crate pnp-ffi -c cbindgen.toml -o include/pnp.h
  )

  if [ ! -s crates/pnp-ffi/include/pnp.h ]; then
    echo "error: failed to generate crates/pnp-ffi/include/pnp.h" >&2
    exit 1
  fi
}

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "error: xcodebuild not found (need Xcode)." >&2
  exit 1
fi

DEVICE_TARGET=aarch64-apple-ios
SIM_TARGET=aarch64-apple-ios-sim

for t in "$DEVICE_TARGET" "$SIM_TARGET"; do
  if ! rustup target list --installed | grep -qx "$t"; then
    echo "==> Installing Rust target $t"
    rustup target add "$t"
  fi
done

echo "==> Building pnp-ffi staticlib (device, profile $PROFILE)"
cargo build --locked --profile "$PROFILE" -p pnp-ffi --target "$DEVICE_TARGET"

echo "==> Building pnp-ffi staticlib (simulator)"
cargo build --locked --profile "$PROFILE" -p pnp-ffi --target "$SIM_TARGET"
generate_header

HEADER="$ROOT/crates/pnp-ffi/include/pnp.h"
DEVICE_LIB="$ROOT/target/$DEVICE_TARGET/$PROFILE/libpeyote_pnp_ffi.a"
SIM_LIB="$ROOT/target/$SIM_TARGET/$PROFILE/libpeyote_pnp_ffi.a"

for f in "$HEADER" "$DEVICE_LIB" "$SIM_LIB"; do
  if [[ ! -f "$f" ]]; then
    echo "error: missing $f" >&2
    exit 1
  fi
done

mk_headers() {
  local dir="$1"
  mkdir -p "$dir"
  cp "$HEADER" "$dir/pnp.h"
  cat >"$dir/module.modulemap" <<'EOF'
module PnpRust {
  header "pnp.h"
  export *
}
EOF
}

DEVICE_HEADERS="$STAGE/device/Headers"
SIM_HEADERS="$STAGE/sim/Headers"
mk_headers "$DEVICE_HEADERS"
mk_headers "$SIM_HEADERS"

DEVICE_STAGED="$STAGE/device/libpeyote_pnp_ffi.a"
SIM_STAGED="$STAGE/sim/libpeyote_pnp_ffi.a"
cp "$DEVICE_LIB" "$DEVICE_STAGED"
cp "$SIM_LIB" "$SIM_STAGED"

rm -rf "$OUT_XCFW"
xcodebuild -create-xcframework \
  -library "$DEVICE_STAGED" -headers "$DEVICE_HEADERS" \
  -library "$SIM_STAGED" -headers "$SIM_HEADERS" \
  -output "$OUT_XCFW"

echo "OK: iOS XCFramework ready at bindings/expo-pnp/ios/PnpRust.xcframework"
