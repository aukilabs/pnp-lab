# Contributing to PnPLab

Thank you for helping improve PnPLab. This guide covers repository setup,
development workflow, and pull-request expectations.

## Before you start

- Search the [issue tracker](https://github.com/aukilabs/pnplab/issues) before
  filing a duplicate bug or feature request.
- For large features, public API changes, new dependencies, or solver redesigns,
  open an issue first so the approach can be agreed before substantial
  implementation work.
- Keep changes focused. Unrelated refactors make numerical regressions harder
  to review.

## Development setup

### Required for the core workspace

| Tool | Notes |
|------|--------|
| Git | — |
| [Rust](https://rustup.rs/) (stable) | See `rust-toolchain.toml` |
| [`just`](https://just.systems/) | Recommended for multi-target recipes |
| [cbindgen](https://github.com/mozilla/cbindgen) | C header generation for `pnp-ffi` |

Clone and verify:

```bash
git clone https://github.com/aukilabs/pnplab.git
cd pnplab
just setup          # or: cargo test --workspace --locked
just test
```

HTTPS is preferred for public contributors; SSH works if you already use it
with GitHub.

### Optional tooling by area

Install only what you need for the code you touch:

| Area | Tools |
|------|--------|
| Python bindings | Python 3.9+, [uv](https://github.com/astral-sh/uv) or Maturin |
| Android natives | Android NDK, Rust targets `aarch64-linux-android`, `x86_64-linux-android` |
| iOS natives | macOS, Xcode, `aarch64-apple-ios`, `aarch64-apple-ios-sim` |
| WASM | `wasm32-wasip2`, `cargo-component` (and `jco` for JS transpile) |
| Reference data | `opencv-python`, `numpy` |

Android NDK is auto-detected from `ANDROID_NDK_HOME` / `ANDROID_NDK_ROOT` or
the default Android Studio NDK path. Builds target API 24 by default.

```bash
# iOS
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Android
rustup target add aarch64-linux-android x86_64-linux-android

# WASM
rustup target add wasm32-wasip2
cargo install cargo-component
```

## Making changes

### Module boundaries

- **`pnp-core`**: pure solvers and types. Must stay `no_std` + `alloc` (use
  `libm` for math, not `std`). Multi-view is the general joint solver:
  `multiview` (`MultiViewRig`, `CameraView`, `MultiViewObservation`),
  `multiview_solve` (`solve_pnp_multiview*`). Stereo is N=2:
  `stereo` (`StereoRig`), `stereo_solve` (thin wrappers over multiview),
  `triangulate`, `absolute_orientation`, plus stereo square helpers in
  `square_pose`. Internals use OpenCV frames; public multiview/stereo
  poses return OpenGL (primary / left), matching mono.
- **`pnp-ffi`**: `#[repr(C)]` types and `extern "C"` entry points only; no
  heavy logic beyond conversion (includes stereo rig / solve / triangulate;
  no multiview C API yet).
- **`pnp-wasm`**: WIT world + thin conversion to core (mono only today; no
  stereo/multiview WIT exports yet).
- **`bindings/python`**: PyO3 facade; prefer validating shapes in Python and
  keeping solvers in Rust (stereo: `solve_pnp_stereo`, `triangulate`; no
  multiview Python API yet).
- **`bindings/expo-pnp`**: TypeScript API + platform glue; natives are built
  with `just expo-native` (no dual-camera stereo product wiring yet).

### Style and documentation

- Format with `cargo fmt --all` (or `just fmt`).
- **Git hooks:** once per clone, run `just install-hooks` (sets
  `core.hooksPath` to `.githooks/`). The `pre-commit` hook runs `cargo fmt
  --all` and re-stages fixed `.rs` files so CI’s `cargo fmt --check` does
  not fail on formatting. Bypass only when necessary: `git commit --no-verify`.
- Document every **public** Rust item (`///` rustdoc). Module-level `//!`
  comments should describe purpose, conventions, and references.
- Prefer clear names and short functions over cleverness.
- Do not commit generated build output (`target/`, wheels, `node_modules/`,
  etc.). `.gitignore` covers the common cases.

### Tests

- Every bug fix or feature should include tests.
- Unit tests live next to the code (`#[cfg(test)] mod tests`).
- OpenCV cross-checks live in `crates/pnp-core/tests/integration.rs` and
  `tests/reference_vectors/`.
- When changing numerics, keep tolerances consistent with the table below.

| Metric | Typical tolerance |
|--------|-------------------|
| Position | &lt; 1e-3 |
| Rotation | &lt; 1e-3 rad (stricter where possible) |
| Reprojection | &lt; 1.0 px |

Regenerate OpenCV reference vectors only when intentionally expanding the
suite:

```bash
pip install opencv-python numpy
just generate-reference
```

## Checks to run

Minimum for core Rust changes (matches CI):

```bash
just install-hooks        # once per clone: versioned pre-commit (auto cargo fmt)
just fmt-check            # or: cargo fmt --all -- --check
cargo test --workspace --locked --exclude pnplab-python
cargo check -p pnp-core --no-default-features --locked
```

GitHub Actions (`.github/workflows/ci.yml`) runs those checks plus clippy and
the Python suite on every push and pull request.

Broader recipes:

```bash
just install-hooks        # enable .githooks (pre-commit fmt)
just fmt                  # cargo fmt --all
just fmt-check            # cargo fmt --check (CI)
just check-all            # tests + no_std + clippy
just test-core            # pnp-core unit tests only
just test-integration     # OpenCV reference integration tests
just test-ffi             # C FFI tests
just python-test          # isolated Python wheel + pytest
just build-wasm-release   # WASM component (if you touch pnp-wasm / WIT)
```

Platform-specific:

```bash
just build-ffi            # host C library
just expo-android         # Android .so → bindings/expo-pnp
just expo-ios             # XCFramework → bindings/expo-pnp
just expo-native          # both
```

After changing the C API, rebuild the header:

```bash
just generate-header
# or: cargo build -p pnp-ffi
```

## Pull requests

1. Branch from `develop` (or the repo’s default development branch).
2. Keep the PR focused; split unrelated work.
3. Update [CHANGELOG.md](CHANGELOG.md) under **Unreleased** for user-visible
   changes (API, solvers, bindings, breaking changes).
4. Ensure the checks above pass for the areas you touched.
5. Write a clear PR description: problem, approach, how to test.

### Commit messages

- Prefer imperative mood: “Add Camera distortion support”.
- Explain *why* when the *what* is not obvious from the diff.
- One logical change per commit when practical.

## Adding a new solver method

1. Implement `crates/pnp-core/src/<solver>.rs`.
2. Export the module from `lib.rs`.
3. Add a `SolvePnpMethod` variant in `types.rs`.
4. Dispatch from `solve.rs`.
5. Add unit tests and, if possible, OpenCV reference coverage.
6. Update FFI enums, regenerate `pnp.h`, and update WIT / language bindings.

## Coordinate and camera conventions

Document any deviation in the PR. Defaults:

- Image pixels: OpenCV (origin top-left, +Y down).
- `solve_pnp` returns **OpenGL** object pose.
- Distortion coeffs: OpenCV order; empty `dist` means ideal pinhole.
- Square corners: TL → TR → BR → BL.

## Security and safety

- C FFI functions must document `# Safety` requirements (null pointers, buffer
  lengths, string lifetimes).
- Do not introduce `unsafe` in `pnp-core` without strong justification and
  review.
- Never commit secrets, credentials, or large private datasets.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
