#!/usr/bin/env bash
# Build libpnp_ffi.so for Android ABIs and install into
# bindings/expo-pnp jniLibs (+ C header for the JNI wrapper).
#
# Requires: cargo, Android NDK (ANDROID_NDK_HOME or SDK ndk/), cbindgen.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_JNI="$ROOT/bindings/expo-pnp/android/src/main/jniLibs"
OUT_INCLUDE="$ROOT/bindings/expo-pnp/android/src/main/cpp/include"
PROFILE="${PNP_MOBILE_PROFILE:-release}"

version_gt() {
  local IFS=.
  local -a left=()
  local -a right=()
  local i left_part right_part

  read -r -a left <<< "$1"
  read -r -a right <<< "$2"

  for i in 0 1 2 3; do
    left_part="${left[$i]:-0}"
    right_part="${right[$i]:-0}"
    if [ "$left_part" -gt "$right_part" ] 2>/dev/null; then
      return 0
    fi
    if [ "$left_part" -lt "$right_part" ] 2>/dev/null; then
      return 1
    fi
  done

  return 1
}

latest_ndk_root() {
  local sdk_root="$1"
  local ndk_parent="$sdk_root/ndk"
  local candidate candidate_name latest="" latest_name=""

  [ -d "$ndk_parent" ] || return 0

  for candidate in "$ndk_parent"/*; do
    [ -d "$candidate" ] || continue
    candidate_name="$(basename "$candidate")"
    if [ -z "$latest_name" ] || version_gt "$candidate_name" "$latest_name"; then
      latest="$candidate"
      latest_name="$candidate_name"
    fi
  done

  printf '%s' "$latest"
}

resolve_ndk_root() {
  local sdk_root ndk_root

  for ndk_root in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
    if [ -n "$ndk_root" ] && [ -d "$ndk_root" ]; then
      printf '%s' "$ndk_root"
      return 0
    fi
  done

  for sdk_root in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk"; do
    if [ -n "$sdk_root" ] && [ -d "$sdk_root" ]; then
      ndk_root="$(latest_ndk_root "$sdk_root")"
      if [ -n "$ndk_root" ]; then
        printf '%s' "$ndk_root"
        return 0
      fi
    fi
  done
}

preferred_host_tags() {
  case "$(uname -s):$(uname -m)" in
    Darwin:arm64)
      printf '%s\n' darwin-arm64 darwin-x86_64
      ;;
    Darwin:*)
      printf '%s\n' darwin-x86_64 darwin-arm64
      ;;
    Linux:*)
      printf '%s\n' linux-x86_64
      ;;
    *)
      printf '%s\n' darwin-x86_64 linux-x86_64 darwin-arm64
      ;;
  esac
}

resolve_ndk_bin() {
  local ndk_root="$1"
  local host_tag candidate

  while IFS= read -r host_tag; do
    candidate="$ndk_root/toolchains/llvm/prebuilt/$host_tag/bin"
    if [ -d "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done < <(preferred_host_tags)
}

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

NDK_ROOT="$(resolve_ndk_root)"

if [ -z "$NDK_ROOT" ] || [ ! -d "$NDK_ROOT" ]; then
  echo "error: Android NDK not found. Set ANDROID_NDK_HOME." >&2
  exit 1
fi

NDK_BIN="$(resolve_ndk_bin "$NDK_ROOT")"
if [ ! -d "$NDK_BIN" ]; then
  echo "error: Android NDK toolchain bin not found under $NDK_ROOT." >&2
  exit 1
fi

export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_BIN/aarch64-linux-android24-clang"
export CC_armv7_linux_androideabi="$NDK_BIN/armv7a-linux-androideabi24-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$NDK_BIN/armv7a-linux-androideabi24-clang"
export CC_i686_linux_android="$NDK_BIN/i686-linux-android24-clang"
export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$NDK_BIN/i686-linux-android24-clang"
export CC_x86_64_linux_android="$NDK_BIN/x86_64-linux-android24-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$NDK_BIN/x86_64-linux-android24-clang"

echo "==> Building pnp-ffi for Android (profile $PROFILE)"
echo "    NDK: $NDK_ROOT"

cargo build --locked --profile "$PROFILE" -p pnp-ffi --target aarch64-linux-android
cargo build --locked --profile "$PROFILE" -p pnp-ffi --target armv7-linux-androideabi
cargo build --locked --profile "$PROFILE" -p pnp-ffi --target i686-linux-android
cargo build --locked --profile "$PROFILE" -p pnp-ffi --target x86_64-linux-android
generate_header

mkdir -p \
  "$OUT_JNI/arm64-v8a" \
  "$OUT_JNI/armeabi-v7a" \
  "$OUT_JNI/x86" \
  "$OUT_JNI/x86_64" \
  "$OUT_INCLUDE"

cp "target/aarch64-linux-android/$PROFILE/libpnp_ffi.so" "$OUT_JNI/arm64-v8a/libpnp_ffi.so"
cp "target/armv7-linux-androideabi/$PROFILE/libpnp_ffi.so" "$OUT_JNI/armeabi-v7a/libpnp_ffi.so"
cp "target/i686-linux-android/$PROFILE/libpnp_ffi.so" "$OUT_JNI/x86/libpnp_ffi.so"
cp "target/x86_64-linux-android/$PROFILE/libpnp_ffi.so" "$OUT_JNI/x86_64/libpnp_ffi.so"
cp crates/pnp-ffi/include/pnp.h "$OUT_INCLUDE/pnp.h"

for abi in arm64-v8a armeabi-v7a x86 x86_64; do
  echo "    installed $OUT_JNI/$abi/libpnp_ffi.so ($(wc -c < "$OUT_JNI/$abi/libpnp_ffi.so") bytes)"
done

echo "OK: Android natives ready under bindings/expo-pnp/android/src/main/jniLibs/"
