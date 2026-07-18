# expo-pnp

Expo module for landmark PnP and square-marker pose solving, powered by the
[PnPLab](https://github.com/aukilabs/pnplab) Rust workspace.

## Install / autolink

This package is intended to live under your app’s Expo modules search path
(for example a git submodule at `modules/pnplab/bindings/expo-pnp`):

```json
{
  "expo": {
    "autolinking": {
      "searchPaths": ["./node_modules", "./modules/pnplab/bindings"]
    }
  }
}
```

Requires a **dev client** (or bare) build — not Expo Go — because of the native
Rust libraries.

## Native artifacts

Prebuilt Android `.so` libraries and the iOS XCFramework ship inside this
package. Rebuild them from the **repository root**:

```sh
just expo-android   # → android/src/main/jniLibs/**
just expo-ios       # → ios/PnpRust.xcframework
just expo-native    # both
```

| Platform | Requirements |
|----------|----------------|
| Android | NDK (`ANDROID_NDK_HOME` or SDK `ndk/`), `cbindgen`, arm64 + x86_64 targets |
| iOS | macOS, Xcode, `aarch64-apple-ios` + `aarch64-apple-ios-sim` |

## JavaScript API

```ts
import {
  solvePnpCameraPose,
  estimateSquarePoseFromRays,
  estimateSquarePoseFromPixels,
  cameraPixelsToSquareRays,
  type PnpCamera,
} from "expo-pnp";

const camera: PnpCamera = {
  fx: 815.85,
  fy: 815.85,
  cx: 960,
  cy: 540,
  dist: [], // optional OpenCV coeffs
};

const pose = await solvePnpCameraPose(
  landmarks,
  observations,
  camera,
  "iterative",
);

// Square marker from four image corners (TL, TR, BR, BL):
const estimate = await estimateSquarePoseFromPixels(
  cornerPixels,
  physicalSizeMeters,
  camera,
);

// Or unproject yourself, then call the ray API:
const rays = cameraPixelsToSquareRays({ camera, pixels: cornerPixels });
const fromRays = await estimateSquarePoseFromRays(rays, physicalSizeMeters);
```

Native calibration code can also call the platform facades (`PnpNative` on
Android/Kotlin and iOS/Swift) without the JS bridge.

## Coordinate notes

- Image pixels: OpenCV-style (top-left origin).
- `solvePnpCameraPose` returns the **camera** pose in the same OpenGL-style
  convention as the Rust core.
- Corner order for squares: top-left → top-right → bottom-right → bottom-left.
- `dist` uses OpenCV Brown–Conrady order; omit or pass `[]` for pinhole.

## License

MIT — see the repository [LICENSE](../../LICENSE).
