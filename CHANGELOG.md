# Changelog

## Unreleased — pnpkit extraction

- Extracted the PnP workspace from the peyote app into a standalone monorepo.
- Grouped the Expo module under `bindings/expo-pnp`, separate from the core
  Rust library crates.
- Prebuilt Android `.so` libraries and the iOS XCFramework install into the
  Expo package (same consumer flow as QRKit).
