# PnPKit — common development commands
#
# Common:
#   just test          # Rust workspace tests
#   just expo-native   # build Android + iOS artifacts into bindings/expo-pnp

set shell := ["bash", "-euo", "pipefail", "-c"]

# Default: list available recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Check that all required (and optional) tools are installed
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    PASS="\033[0;32m✔\033[0m"
    FAIL="\033[0;31m✘\033[0m"
    WARN="\033[0;33m⚠\033[0m"
    errors=0
    warnings=0

    echo "Checking required tools..."
    echo ""

    # --- Rust toolchain ---
    if command -v rustup &>/dev/null; then
        ver=$(rustc --version 2>/dev/null | awk '{print $2}')
        echo -e "  $PASS  rustc $ver"
    else
        echo -e "  $FAIL  rustc not found"
        echo "         Install Rust: https://rustup.rs/"
        errors=$((errors + 1))
    fi

    if command -v cargo &>/dev/null; then
        echo -e "  $PASS  cargo"
    else
        echo -e "  $FAIL  cargo not found (installed with rustup)"
        errors=$((errors + 1))
    fi

    # --- just (you're already here, but check anyway) ---
    if command -v just &>/dev/null; then
        echo -e "  $PASS  just"
    else
        echo -e "  $FAIL  just not found"
        echo "         Install: cargo install just"
        errors=$((errors + 1))
    fi

    # --- cbindgen (required for FFI header generation) ---
    if command -v cbindgen &>/dev/null; then
        ver=$(cbindgen --version 2>/dev/null | awk '{print $2}')
        echo -e "  $PASS  cbindgen $ver"
    else
        echo -e "  $FAIL  cbindgen not found (required for C FFI builds)"
        echo "         Install: cargo install cbindgen"
        errors=$((errors + 1))
    fi

    echo ""
    echo "Checking Rust targets..."
    echo ""

    # --- Native (always available) ---
    host=$(rustc -vV | grep host | awk '{print $2}')
    echo -e "  $PASS  $host (host)"

    # --- iOS ---
    for target in aarch64-apple-ios aarch64-apple-ios-sim; do
        if rustup target list --installed | grep -q "^${target}$"; then
            echo -e "  $PASS  $target"
        else
            echo -e "  $WARN  $target not installed (needed for iOS builds)"
            echo "         Install: rustup target add $target"
            warnings=$((warnings + 1))
        fi
    done

    # --- Android ---
    for target in aarch64-linux-android x86_64-linux-android; do
        if rustup target list --installed | grep -q "^${target}$"; then
            echo -e "  $PASS  $target"
        else
            echo -e "  $WARN  $target not installed (needed for Android builds)"
            echo "         Install: rustup target add $target"
            warnings=$((warnings + 1))
        fi
    done

    # --- WASM ---
    for target in wasm32-wasip2; do
        if rustup target list --installed | grep -q "^${target}$"; then
            echo -e "  $PASS  $target"
        else
            echo -e "  $WARN  $target not installed (needed for WASM builds)"
            echo "         Install: rustup target add $target"
            warnings=$((warnings + 1))
        fi
    done

    echo ""
    echo "Checking Android NDK..."
    echo ""

    ndk_found=false
    for ndk_root in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
        if [ -n "$ndk_root" ] && [ -d "$ndk_root" ]; then
            ndk_ver=$(basename "$ndk_root")
            echo -e "  $PASS  Android NDK $ndk_ver ($ndk_root)"
            ndk_found=true
            break
        fi
    done
    if [ "$ndk_found" = false ]; then
        for sdk_root in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk"; do
            latest_ndk=""
            latest_name=""
            if [ -n "$sdk_root" ] && [ -d "$sdk_root/ndk" ]; then
                for candidate in "$sdk_root"/ndk/*; do
                    [ -d "$candidate" ] || continue
                    candidate_name=$(basename "$candidate")
                    if [ -z "$latest_name" ] || [ "$candidate_name" \> "$latest_name" ]; then
                        latest_ndk="$candidate"
                        latest_name="$candidate_name"
                    fi
                done
            fi
            if [ -n "$latest_ndk" ]; then
                echo -e "  $PASS  Android NDK $latest_name ($latest_ndk)"
                ndk_found=true
                break
            fi
        done
    fi

    if [ "$ndk_found" = false ]; then
        echo -e "  $WARN  Android NDK not found (needed for Android builds)"
        echo "         Install via Android Studio > SDK Manager > SDK Tools > NDK"
        echo "         Or set ANDROID_NDK_HOME to point to your NDK installation"
        warnings=$((warnings + 1))
    fi

    echo ""
    echo "Checking optional tools..."
    echo ""

    # --- cargo-component (WASM) ---
    if command -v cargo-component &>/dev/null; then
        ver=$(cargo component --version 2>/dev/null | awk '{print $2}')
        echo -e "  $PASS  cargo-component $ver"
    else
        echo -e "  $WARN  cargo-component not found (needed for WASM builds)"
        echo "         Install: cargo install cargo-component"
        warnings=$((warnings + 1))
    fi

    # --- jco (WASM transpile to JS) ---
    if command -v jco &>/dev/null; then
        ver=$(jco --version 2>/dev/null)
        echo -e "  $PASS  jco $ver"
    else
        echo -e "  $WARN  jco not found (needed for WASM-to-JS transpile)"
        echo "         Install: npm install -g @bytecodealliance/jco"
        warnings=$((warnings + 1))
    fi

    # --- Python + opencv (reference data) ---
    if command -v python3 &>/dev/null; then
        pyver=$(python3 --version 2>/dev/null | awk '{print $2}')
        if python3 -c "import cv2" 2>/dev/null; then
            echo -e "  $PASS  python3 $pyver + opencv-python"
        else
            echo -e "  $WARN  python3 $pyver found, but opencv-python not installed"
            echo "         Install: pip install opencv-python numpy"
            warnings=$((warnings + 1))
        fi
    else
        echo -e "  $WARN  python3 not found (needed to regenerate reference test data)"
        warnings=$((warnings + 1))
    fi

    # --- Summary ---
    echo ""
    if [ $errors -gt 0 ]; then
        echo -e "$FAIL $errors required tool(s) missing. Fix the errors above before building."
        exit 1
    elif [ $warnings -gt 0 ]; then
        echo -e "$PASS All required tools present. $warnings optional tool(s) missing (see above)."
        echo "  Core builds and tests will work. Install optional tools for cross-compilation targets."
    else
        echo -e "$PASS Everything installed. You're ready to go!"
    fi

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run all tests across the workspace
test:
    cargo test --workspace --locked

# Run only pnp-core unit tests
test-core:
    cargo test -p pnp-core --lib --locked

# Run integration tests (requires reference data)
test-integration:
    cargo test -p pnp-core --test integration --locked

# Run FFI tests
test-ffi:
    cargo test -p pnp-ffi --locked

# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------

# Build all crates in debug mode
build:
    cargo build --workspace --locked

# Build all crates in release mode
build-release:
    cargo build --workspace --release --locked

# Build the WASM component (debug)
build-wasm:
    cargo component build -p pnp-wasm --target wasm32-wasip2 --locked

# Build the WASM component (release)
build-wasm-release:
    cargo component build --release -p pnp-wasm --target wasm32-wasip2 --locked

# Build the C FFI library (release, macOS)
build-ffi:
    cargo build --release -p pnp-ffi --locked

# Build the C FFI library for iOS (aarch64)
build-ios:
    cargo build --release -p pnp-ffi --target aarch64-apple-ios --locked

# Build the C FFI library for iOS Simulator (aarch64)
build-ios-sim:
    cargo build --release -p pnp-ffi --target aarch64-apple-ios-sim --locked

# Build the C FFI library for all Android targets into the Expo package
build-android:
    ./scripts/build-native-android.sh

# ---------------------------------------------------------------------------
# Expo module (bindings/expo-pnp) native binaries
# Prebuilt artifacts land inside the package so app consumers need no Rust.
# ---------------------------------------------------------------------------

# Build Android libpeyote_pnp_ffi.so → bindings/expo-pnp/android/src/main/jniLibs/
expo-android:
    ./scripts/build-native-android.sh

# Build iOS PnpRust.xcframework → bindings/expo-pnp/ios/
expo-ios:
    ./scripts/build-native-ios.sh

# Both platforms
expo-native: expo-android expo-ios

# Host unit tests for the FFI crate (no NDK/Xcode required)
ffi-test:
    cargo test -p pnp-ffi --locked

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

# Verify no_std compilation
check-nostd:
    cargo check -p pnp-core --no-default-features --locked

# Run all checks (test + no_std + clippy)
check-all: test check-nostd
    cargo clippy --workspace --locked -- -D warnings

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

# Transpile WASM component to browser-ready JS via jco
transpile: build-wasm-release
    jco transpile target/wasm32-wasip2/release/pnp_wasm.wasm -o dist/wasm

# Regenerate the C header (requires cbindgen CLI)
generate-header:
    cd crates/pnp-ffi && cbindgen --crate pnp-ffi -c cbindgen.toml -o include/pnp.h

# Generate reference test data (requires Python + opencv-python)
generate-reference:
    python3 tests/generate_reference.py

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Print artifact sizes
sizes: build-release
    @echo "=== Release artifact sizes ==="
    @ls -lh target/release/libpeyote_pnp_ffi.a 2>/dev/null || true
    @ls -lh target/release/libpeyote_pnp_ffi.dylib 2>/dev/null || true
    @ls -lh bindings/expo-pnp/android/src/main/jniLibs/arm64-v8a/libpeyote_pnp_ffi.so 2>/dev/null || true

# Clean all build artifacts
clean:
    cargo clean
