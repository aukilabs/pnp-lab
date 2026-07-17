# Contributing

Thank you for considering contributing to the PnP pose estimator.

## Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:aukilabs/pnpkit.git
cd pnpkit
```

### 2. Install required tools

These are needed for core development (building, testing, generating the C header):

| Tool | Install | Used for |
|------|---------|----------|
| [Rust](https://rustup.rs/) 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | Compiler toolchain |
| [just](https://github.com/casey/just) | `cargo install just` | Task runner |
| [cbindgen](https://github.com/mozilla/cbindgen) | `cargo install cbindgen` | C header generation for pnp-ffi |

### 3. Install cross-compilation targets

Only install the targets you need. None are required for core development and tests.

**iOS:**

```bash
rustup target add aarch64-apple-ios        # devices
rustup target add aarch64-apple-ios-sim    # simulator
```

**Android** (requires the Android NDK, see below):

```bash
rustup target add aarch64-linux-android    # arm64-v8a (devices)
rustup target add x86_64-linux-android     # x86_64 (emulator)
```

**WASM:**

```bash
rustup target add wasm32-wasip2
cargo install cargo-component              # WASM Component Model builder
```

### 4. Android NDK setup

Android builds require the [Android NDK](https://developer.android.com/ndk) for the cross-linker. Install it through one of:

- **Android Studio**: SDK Manager > SDK Tools > check "NDK (Side by side)" > Apply
- **Command line**: `sdkmanager --install "ndk;27.1.12297006"` (or latest)

The build recipes auto-detect the NDK from `ANDROID_NDK_HOME` or the default Android Studio path (`~/Library/Android/sdk/ndk/<version>`). To use a non-standard location, export `ANDROID_NDK_HOME`:

```bash
export ANDROID_NDK_HOME=/path/to/your/ndk
```

Android builds target API level 24 (Android 7.0) by default.

### 5. Optional tools

| Tool | Install | Used for |
|------|---------|----------|
| [jco](https://github.com/nicknisi/jco) | `npm install -g @bytecodealliance/jco` | Transpile WASM component to browser JS |
| Python 3.9+ | System or `brew install python` | Regenerating reference data; Python bindings |
| [uv](https://github.com/astral-sh/uv) or Maturin | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Build/test `bindings/python` |
| [opencv-python](https://pypi.org/project/opencv-python/) | `pip install opencv-python numpy` | Used by the reference data generator |

### 6. Verify your setup

```bash
just setup
```

This runs an idempotent check of all required and optional tools, Rust targets, and the Android NDK. It will tell you exactly what's missing and how to install it.

### 7. Run the test suite

```bash
just test
```

## Development Workflow

### Running Tests

```bash
just test              # All tests across the workspace
just test-core         # pnp-core unit tests only
just test-integration  # Integration tests against OpenCV reference data
just test-ffi          # C FFI binding tests
just python-test       # Isolated Python wheel + pytest suite
just check-nostd       # Verify no_std compatibility
just check-all         # Tests + no_std + clippy
```

### Building

```bash
just build              # Debug build (all crates)
just build-release      # Release build (all crates)
```

**Platform-specific FFI builds:**

```bash
just build-ffi            # macOS (host)
just build-ios            # iOS device (aarch64)
just build-ios-sim        # iOS Simulator (aarch64)
just build-android        # Android arm64 + x86_64
just build-android-arm64  # Android arm64 only
just build-android-x86_64 # Android x86_64 only (emulator)
```

**WASM builds:**

```bash
just build-wasm           # Debug
just build-wasm-release   # Release
just transpile            # Release + transpile to browser JS
```

### Reference Data

The test suite validates against OpenCV `cv::solvePnP` reference output stored in `tests/reference_vectors/`. To regenerate:

```bash
pip install opencv-python numpy
just generate-reference
```

## Project Structure

```
.cargo/
  config.toml           Android cross-compilation notes
crates/
  pnp-core/             Core solver library (no_std + alloc)
    src/
      types.rs           Data types (Vector2/3, Quaternion, Matrix3x3, Pose, etc.)
      rodrigues.rs       Rodrigues rotation vector <-> matrix conversion
      pose_tools.rs      OpenCV/OpenGL coordinate conversion, pose inversion
      solve.rs           Top-level API and method dispatch
      epnp.rs            EPnP solver (coplanar/non-coplanar)
      iterative.rs       Levenberg-Marquardt refinement
      sqpnp.rs           SQPnP solver
    tests/
      integration.rs     Integration tests against OpenCV reference
  pnp-wasm/              WASM Component Model guest
    wit/pnp.wit          WIT interface definition
    src/lib.rs           Type conversion + solver delegation
  pnp-ffi/               C FFI layer
    src/lib.rs           #[repr(C)] types + extern "C" functions
    include/pnp.h        Auto-generated C header (cbindgen)
    cbindgen.toml        cbindgen configuration
bindings/
  python/                aukilabs-pnpkit Maturin/PyO3 package
  expo-pnp/              Expo module + prebuilt Android/iOS natives
tests/
  generate_reference.py  OpenCV reference data generator
  reference_vectors/     JSON reference output
```

## Guidelines

### Code Style

- Follow standard `rustfmt` formatting (`cargo fmt`).
- The core crate (`pnp-core`) must remain `no_std` compatible. Use `alloc` for heap allocations, `libm` for math functions.
- All public APIs need doc comments.

### Testing

- Every new feature or bug fix must include tests.
- Unit tests go in the same file as the code (`#[cfg(test)] mod tests`).
- Integration tests that require reference data go in `crates/pnp-core/tests/`.
- Use the existing reference test vectors where applicable. If a new test case is needed, add it to `tests/generate_reference.py` and regenerate.

### Tolerances

When comparing against reference data:

| Metric | Tolerance |
|--------|-----------|
| Position | < 1e-3 (1 mm) |
| Rotation | < 1e-4 rad (~0.006 deg) |
| Reprojection error | < 1.0 pixel |

### Adding a New Solver

1. Create `crates/pnp-core/src/<solver>.rs`
2. Add `pub mod <solver>;` to `lib.rs`
3. Add a variant to `SolvePnpMethod` in `types.rs`
4. Wire the dispatch in `solve.rs`
5. Add tests that validate against known poses and reference data
6. Update the WIT interface if the method enum changes
7. Update FFI types and regenerate the C header

### Commits

- Write clear, descriptive commit messages.
- Keep commits focused on a single change.
- Run `just test` before pushing.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
