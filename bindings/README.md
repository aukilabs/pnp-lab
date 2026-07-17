# PnPKit bindings

This directory contains the publishable language and framework integrations
built on the PnPKit Rust crates:

- [`python/`](python/) — Maturin/PyO3 package `aukilabs-pnpkit` (imports as
  `auki_pnpkit`) with NumPy-friendly PnP and square-pose APIs.
- [`expo-pnp/`](expo-pnp/) — the Expo module, prebuilt Android libraries, iOS
  XCFramework, and TypeScript API.

Build artifacts are written into their corresponding binding package:

| Command | Output |
|---|---|
| `just python-build` | `bindings/python/dist/` |
| `just expo-android` | `bindings/expo-pnp/android/src/main/jniLibs/` |
| `just expo-ios` | `bindings/expo-pnp/ios/PnpRust.xcframework/` |

Run `just test`, `just python-test`, and `just ffi-test` for workspace checks
that do not launch an example application.
