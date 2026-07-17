# PnPKit

PnPKit is a pure-Rust [Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point)
pose estimator. It recovers the 6-DoF pose of a calibrated camera from known 3D
landmarks and their 2D image observations, with no OpenCV dependency.

The core is written in Rust and is exposed through Rust, Python/NumPy, C,
WebAssembly, and an Expo module for React Native.

## Features

- Three solver methods: EPnP, iterative (Levenberg-Marquardt), and SQPnP
- Square-marker pose from camera rays (AR / portal calibration workflows)
- `no_std` compatible core (with `alloc`)
- C FFI with auto-generated header for iOS / Android / desktop embedding
- Python/NumPy bindings (`aukilabs-pnpkit` / `auki_pnpkit`)
- WASM Component Model interface via WIT (`auki:pnp@0.1.0`)
- Numerically validated against OpenCV `cv::solvePnP` reference output

## Workspace structure

```text
crates/
  pnp-core/    Pure Rust solver library (no_std + alloc)
  pnp-ffi/     C FFI layer with cbindgen-generated header
  pnp-wasm/    WASM Component Model guest (wit-bindgen)

bindings/
  python/      PyPI/Maturin project with NumPy-friendly API
  expo-pnp/    Expo module + prebuilt Android/iOS natives
```

## Quick start

```bash
git clone git@github.com:aukilabs/pnpkit.git
cd pnpkit
cargo test --workspace --locked
```

[`just`](https://just.systems/) is optional but provides the shortest commands:

```bash
just setup          # check tools, targets, NDK
just test           # Rust workspace tests
just expo-native    # Android + iOS artifacts → bindings/expo-pnp
```

### Rust

```toml
[dependencies]
pnp-core = { path = "crates/pnp-core" }
```

```rust
use pnp_core::types::*;
```

### Python

```bash
just python-build         # wheel → bindings/python/dist
just python-test          # isolated wheel + pytest
```

```python
import numpy as np
import auki_pnpkit

pose = auki_pnpkit.solve_pnp(
    object_points,   # (N, 3)
    image_points,    # (N, 2)
    camera_matrix,   # (3, 3) OpenCV K
    method="iterative",
)
```

See [bindings/python/README.md](bindings/python/README.md).

### Expo

Autolink `bindings/expo-pnp` (for example via a git submodule under
`modules/pnpkit` and Expo `autolinking.searchPaths`):

```json
{
  "expo": {
    "autolinking": {
      "searchPaths": ["./node_modules", "./modules/pnpkit/bindings"]
    }
  }
}
```

```ts
import { estimateSquarePoseFromRays } from "expo-pnp";
```

## Development

| Command | Purpose |
|---|---|
| `just test` | Workspace tests (release-friendly via `--locked`) |
| `just check-nostd` | Verify `pnp-core` builds without `std` |
| `just expo-android` | Install Android `.so` files into the Expo package |
| `just expo-ios` | Package `PnpRust.xcframework` into the Expo package |
| `just expo-native` | Both mobile targets |
| `just build-wasm-release` | WASM component |
| `just python-build` | Build Python wheel into `bindings/python/dist` |
| `just python-test` | Isolated Python/NumPy integration tests |

Additional platform requirements:

- Android builds need the Android NDK and `cbindgen`
- iOS builds need macOS, Xcode, and the iOS Rust targets
- WASM builds need `cargo-component` (and `jco` for JS transpile)
- Python bindings need Python 3.9+ and Maturin or `uv`/`uvx`

## License

MIT — see [LICENSE](LICENSE).
