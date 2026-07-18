# PnPKit bindings

Publishable language and framework integrations built on the Rust crates:

| Package | Path | Notes |
|---------|------|--------|
| Python / NumPy | [`python/`](python/) | PyPI name `aukilabs-pnpkit`, import `auki_pnpkit` |
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
