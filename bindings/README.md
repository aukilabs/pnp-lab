# PnPLab bindings

Language and framework integrations built on the Rust crates. Python is not on
PyPI yet; the Expo module is consumed from this repository (not npm).

| Package | Path | Notes |
|---------|------|--------|
| Python / NumPy | [`python/`](python/) | Planned PyPI name `aukilabs-pnplab`, import `auki_pnplab` |
| Expo / React Native | [`expo-pnp/`](expo-pnp/) | TypeScript API + prebuilt Android/iOS natives |

Low-level C and WASM APIs live under `crates/pnp-ffi` and `crates/pnp-wasm`
rather than this directory.

## Build artifacts

| Command | Output |
|---------|--------|
| `just python-build` | `bindings/python/dist/` |
| `just expo-android` | `bindings/expo-pnp/android/src/main/jniLibs/` |
| `just expo-ios` | `bindings/expo-pnp/ios/PnpRust.xcframework/` |
| `just expo-native` | Both mobile targets |

Workspace checks that do not launch an app: `just test`, `just python-test`,
`just ffi-test`.

See the [root README](../README.md) for getting started and
[CONTRIBUTING.md](../CONTRIBUTING.md) for development workflow.
