# expo-pnp

Expo module for square-marker and landmark PnP pose solving, powered by the
[PnPKit](https://github.com/aukilabs/pnpkit) Rust workspace.

## Native build artifacts

Prebuilt Android libraries and the iOS XCFramework live inside this package.
Rebuild them from the repository root:

```sh
just expo-android   # → android/src/main/jniLibs/**
just expo-ios       # → ios/PnpRust.xcframework
just expo-native    # both
```

Android needs an NDK (`ANDROID_NDK_HOME` or the SDK `ndk/` tree) and
`cbindgen`. iOS needs Xcode and the `aarch64-apple-ios` /
`aarch64-apple-ios-sim` Rust targets.

## JavaScript API

```ts
import {
  estimateSquarePoseFromRays,
  solvePnpCameraPose,
  cameraPixelsToSquareRays,
} from "expo-pnp";
```

Native calibration code can also call the platform facades (`PnpNative` on
Android/Kotlin and iOS/Swift) without going through the JS bridge.
