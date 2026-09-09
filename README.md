# PnPLab

[![CI](https://github.com/aukilabs/pnp-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aukilabs/pnp-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**PnPLab** is a pure-Rust
[Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point) pose
estimator. Given a calibrated camera, known 3D landmarks, and their 2D image
observations, it recovers a 6-DoF pose with **no OpenCV runtime dependency**.

The same solvers are available from Rust, Python/NumPy, C, WebAssembly, and an
Expo module for React Native.

> **Pre-1.0 software.** APIs may still change. Pin a git revision, and read
> [CHANGELOG.md](CHANGELOG.md) before upgrading. The crates are **not on
> crates.io** yet — depend on this repository.

## Why PnPLab

| | |
|---|---|
| **No OpenCV at runtime** | Solvers are pure Rust (`nalgebra` + `libm`). OpenCV is used only to generate optional reference test vectors. |
| **Camera-first** | First-class `Camera` with pinhole, Brown–Conrady, and OpenCV fisheye models |
| **Multi-view** | Joint PnP for N ≥ 1 calibrated views; stereo is the N=2 helper |
| **Portable** | `no_std` + `alloc` core; C, Python, WASM, and Expo bindings from the same crate |
| **Checked** | Numerically compared against OpenCV `cv::solvePnP` reference vectors |

## Features

- Three solver methods: **EPnP**, **iterative** (Levenberg–Marquardt), and **SQPnP**
- First-class monocular `Camera` model with optional Brown–Conrady distortion
  (OpenCV coefficient order) and OpenCV fisheye
- **Multi-view joint PnP**: `MultiViewRig` / `solve_pnp_multiview` for N ≥ 1
  calibrated views with fixed extrinsics (primary = `views[0]`)
- **Calibrated stereo** as N=2 helper: `StereoRig`, joint left+right landmark
  PnP (delegates to multiview), midpoint triangulation, and stereo
  square-marker pose
- **Multi-view monocular calibration**: Zhang-style init + joint BA
  (`calibrate_camera`, `calibrate_from_square_views`) for intrinsics and
  optional Brown–Conrady distortion from many views of a known planar target
- Square-marker pose from **camera rays** or **image pixels** (AR / portal
  calibration workflows)
- `no_std` + `alloc` core for embedded and mobile targets
- C FFI with auto-generated header (`cbindgen`)
- Python/NumPy package (`aukilabs-pnplab` / `auki_pnplab`)
- WASM Component Model interface (`auki:pnp@0.2.0`)

### Language support

| Binding | Solve | Stereo | Multi-view | Calibration |
|---------|:-----:|:------:|:----------:|:-----------:|
| Rust (`pnp-core`) | yes | yes | yes | yes |
| C FFI | yes | yes | — | yes |
| Python | yes | yes | — | yes |
| WASM | yes | — | — | — |
| Expo / React Native | yes | — | — | — |

## Install

### Rust

Not on crates.io yet. Depend on git and pin a revision for production:

```toml
[dependencies]
pnp-core = { git = "https://github.com/aukilabs/pnp-lab", package = "pnp-core" }
# pin with: rev = "<commit sha>"
```

In this workspace:

```toml
pnp-core = { path = "crates/pnp-core" }
```

### Python

Not on PyPI yet. Build a local wheel from the repository root (requires
[Maturin](https://www.maturin.rs/) or [uv](https://github.com/astral-sh/uv)):

```bash
just python-build         # → bindings/python/dist/
just python-test          # isolated wheel + pytest
```

```python
import auki_pnplab
```

Full API: [bindings/python/README.md](bindings/python/README.md).

### C / native

```bash
cargo build --release -p pnp-ffi --locked
# header: crates/pnp-ffi/include/pnp.h
# library: target/release/libpnp_ffi.{a,so,dylib}
```

Link against the static or dynamic library and include `pnp.h`. Solve entry
points take a `pnp_camera_t` (`fx`, `fy`, `cx`, `cy`, optional `dist`).

### Expo / React Native

Prebuilt Android `.so` files and an iOS XCFramework live under
`bindings/expo-pnp` after:

```bash
just expo-native
```

Requires a **dev client** (or bare) build — not Expo Go. Autolink via Expo
`autolinking.searchPaths` (for example a git submodule):

```json
{
  "expo": {
    "autolinking": {
      "searchPaths": ["./node_modules", "./modules/pnp-lab/bindings"]
    }
  }
}
```

```ts
import {
  estimateSquarePoseFromRays,
  estimateSquarePoseFromPixels,
  solvePnpCameraPose,
} from "expo-pnp";
```

See [bindings/README.md](bindings/README.md) and
[bindings/expo-pnp/README.md](bindings/expo-pnp/README.md).

## Getting started from source

### Requirements

| Tool | Required for |
|------|----------------|
| [Rust](https://rustup.rs/) (stable; see `rust-toolchain.toml`) | Building and testing the workspace |
| [just](https://just.systems/) (optional) | Short recipes for multi-target workflows |
| Python 3.9+ + [uv](https://github.com/astral-sh/uv) or Maturin | Python bindings |
| Android NDK + `cbindgen` | Android native artifacts |
| macOS + Xcode | iOS native artifacts |
| `cargo-component` (+ `jco` for JS transpile) | WASM |

```bash
git clone https://github.com/aukilabs/pnp-lab.git
cd pnp-lab
cargo test --workspace --locked
```

SSH: `git clone git@github.com:aukilabs/pnp-lab.git`

With `just`:

```bash
just setup    # check toolchain, targets, optional deps
just test     # full Rust workspace tests
```

## Quick start (Rust)

```rust
use pnp_core::{
    solve_pnp, Camera, Landmark, LandmarkObservation, SolvePnpMethod, Vector2, Vector3,
};

fn estimate_object_pose() -> Result<pnp_core::Pose, pnp_core::PnpError> {
    let camera = Camera::pinhole(
        /* fx */ 815.8511,
        /* fy */ 815.8511,
        /* cx */ 960.0,
        /* cy */ 540.0,
    )?;
    // Optional distortion (OpenCV order: k1, k2, p1, p2 [, k3 ...]):
    // let camera = Camera::new(815.8511, 815.8511, 960.0, 540.0, &[0.1, -0.05, 0.0, 0.0, 0.0])?;
    // OpenCV fisheye calibration uses a distinct equidistant model:
    // let camera = Camera::opencv_fisheye(815.8511, 815.8511, 960.0, 540.0, &[0.01, -0.003, 0.0, 0.0])?;

    let landmarks = vec![
        Landmark {
            id: "0".into(),
            position: Vector3::new(-0.15, -0.15, 0.0),
        },
        Landmark {
            id: "1".into(),
            position: Vector3::new(0.15, -0.15, 0.0),
        },
        Landmark {
            id: "2".into(),
            position: Vector3::new(0.15, 0.15, 0.0),
        },
        Landmark {
            id: "3".into(),
            position: Vector3::new(-0.15, 0.15, 0.0),
        },
    ];
    let observations = vec![
        LandmarkObservation {
            id: "0".into(),
            position: Vector2::new(849.3577, 461.7641),
        },
        LandmarkObservation {
            id: "1".into(),
            position: Vector2::new(1070.642, 461.7641),
        },
        LandmarkObservation {
            id: "2".into(),
            position: Vector2::new(1096.898, 636.8014),
        },
        LandmarkObservation {
            id: "3".into(),
            position: Vector2::new(823.1021, 636.8014),
        },
    ];

    // Returns object pose in OpenGL coordinates (Y-up, Z-backward).
    // Image points are distorted pixels; Camera undistorts when dist is set.
    solve_pnp(
        &landmarks,
        &observations,
        &camera,
        SolvePnpMethod::Iterative,
    )
}
```

Useful entry points:

| API | Purpose |
|-----|---------|
| `solve_pnp` | Object pose in OpenGL coordinates |
| `solve_pnp_camera_pose` | Camera pose (inverse of object pose) |
| `solve_pnp_multiview` | Object pose from N-view observations (OpenGL, primary = `views[0]`) |
| `solve_pnp_multiview_camera_pose` | Multi-view camera pose (inverse of object pose) |
| `solve_pnp_stereo` | Object pose from stereo observations (OpenGL, left primary; N=2 multiview) |
| `solve_pnp_stereo_camera_pose` | Stereo camera pose (inverse of object pose) |
| `triangulate_midpoint` | 3D point in left OpenCV frame from a stereo pair |
| `estimate_square_pose_from_rays` | Square marker from four 3D rays |
| `estimate_square_pose_from_pixels` | Square marker from four image corners + `Camera` |
| `estimate_square_pose_from_stereo_pixels` | Square marker from dual-eye corner pixels |
| `calibrate_from_square_views` | Monocular intrinsics from multi-view square / QR corners |
| `calibrate_camera` | Monocular intrinsics from shared 3D object points + multi-view pixels |
| `Camera::project` / `undistort_pixel` / `unproject_opengl_ray` | Projection helpers |
| `MultiViewRig` / `MultiViewObservation` / `CameraView` | N-view calibrated rig and sparse per-view pixels |
| `StereoRig` / `StereoLandmarkObservation` | Calibrated stereo pair and partial observations |

### Python

```python
import numpy as np
import auki_pnplab

object_points = np.array(
    [[-0.15, -0.15, 0.0], [0.15, -0.15, 0.0], [0.15, 0.15, 0.0], [-0.15, 0.15, 0.0]],
    dtype=np.float64,
)
image_points = np.array(
    [[849.3577, 461.7641], [1070.642, 461.7641], [1096.898, 636.8014], [823.1021, 636.8014]],
    dtype=np.float64,
)
camera = {
    "fx": 815.8511,
    "fy": 815.8511,
    "cx": 960.0,
    "cy": 540.0,
    "dist": [],  # or OpenCV [k1, k2, p1, p2, k3, ...]
}

pose = auki_pnplab.solve_pnp(object_points, image_points, camera, method="iterative")
print(pose["position"], pose["rotation"])
```

A pinhole `(3, 3)` OpenCV camera matrix is also accepted in place of the
`camera` dict.

## Multi-view

**Multi-view is the general joint-PnP API.** A `MultiViewRig` holds N ≥ 1
calibrated views; `views[0]` is **primary**. The returned object pose is
expressed in the primary frame (OpenGL, same meaning as `solve_pnp`). Each
non-primary view has a fixed extrinsic `from_primary` in **OpenCV** convention
(`+Z` forward):

```text
X_view = R * X_primary + t
```

For the primary view, `from_primary` is forced to identity on construction.
Observations are sparse: each `MultiViewObservation` carries a `pixels` vector
aligned with `rig.views` (`Some(uv)` or `None` for missing/occluded).

Stereo is the N=2 special case (`MultiViewRig::from_stereo` / `StereoRig::to_multiview`);
`solve_pnp_stereo` delegates to `solve_pnp_multiview` with no intentional
behavior change. If product calibration is OpenGL, convert extrinsics with
`pose_tools::from_opengl_to_opencv` before building the rig.

```rust
use pnp_core::{
    solve_pnp_multiview, Camera, CameraView, Landmark, MultiViewObservation,
    MultiViewRig, Pose, Quaternion, SolvePnpMethod, Vector2, Vector3,
};

fn multiview_example() -> Result<pnp_core::Pose, pnp_core::PnpError> {
    let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0)?;
    // Three coplanar cameras on a 12 cm grid; view 0 is primary.
    let rig = MultiViewRig::new(vec![
        CameraView {
            camera: cam.clone(),
            from_primary: Pose::identity(),
        },
        CameraView {
            camera: cam.clone(),
            from_primary: Pose::new(Vector3::new(0.12, 0.0, 0.0), Quaternion::identity()),
        },
        CameraView {
            camera: cam,
            from_primary: Pose::new(Vector3::new(0.0, 0.12, 0.0), Quaternion::identity()),
        },
    ])?;

    // Or from a stereo pair: MultiViewRig::from_stereo(&stereo_rig)

    let landmarks = vec![
        Landmark {
            id: "0".into(),
            position: Vector3::new(-0.1, -0.1, 0.0),
        },
        Landmark {
            id: "1".into(),
            position: Vector3::new(0.1, -0.1, 0.0),
        },
        Landmark {
            id: "2".into(),
            position: Vector3::new(0.1, 0.1, 0.0),
        },
        Landmark {
            id: "3".into(),
            position: Vector3::new(-0.1, 0.1, 0.0),
        },
    ];
    // pixels.len() == rig.num_views(); None = not observed in that view.
    let observations = vec![
        MultiViewObservation {
            id: "0".into(),
            pixels: vec![
                Some(Vector2::new(220.0, 140.0)),
                Some(Vector2::new(200.0, 140.0)),
                None, // occluded in view 2
            ],
        },
        MultiViewObservation {
            id: "1".into(),
            pixels: vec![
                Some(Vector2::new(420.0, 140.0)),
                Some(Vector2::new(400.0, 140.0)),
                Some(Vector2::new(420.0, 100.0)),
            ],
        },
        MultiViewObservation {
            id: "2".into(),
            pixels: vec![
                Some(Vector2::new(420.0, 340.0)),
                Some(Vector2::new(400.0, 340.0)),
                Some(Vector2::new(420.0, 300.0)),
            ],
        },
        MultiViewObservation {
            id: "3".into(),
            pixels: vec![
                Some(Vector2::new(220.0, 340.0)),
                Some(Vector2::new(200.0, 340.0)),
                Some(Vector2::new(220.0, 300.0)),
            ],
        },
    ];

    // Object pose in OpenGL, primary = views[0] (same meaning as solve_pnp).
    solve_pnp_multiview(
        &landmarks,
        &observations,
        &rig,
        SolvePnpMethod::Iterative,
    )
}
```

Pipeline notes:

| Topic | Behavior |
|-------|----------|
| Seed | Monocular PnP preferring **primary**; if primary is sparse, seed from the richest view with enough points and transport into primary |
| Refine | Joint LM over all-view reprojection residuals (OpenCV internals) |
| Partial observations | `None` pixels are skipped in the residual |
| Public pose | **OpenGL** object pose, primary = `views[0]` (same meaning as `solve_pnp`) |
| Extrinsics | Per-view `from_primary` is **OpenCV** (`X_view = R * X_primary + t`) |

## Stereo

Calibrated stereo is a thin N=2 wrapper over multi-view: left is primary,
`right_from_left` becomes the right view's `from_primary`. The left camera is
the frame of the returned object pose (and the preferred monocular seed). If
left is sparse, the seed may use the right view and transport into left.

```text
X_right = R_rl * X_left + t_rl
```

```rust
use pnp_core::{
    solve_pnp_stereo, triangulate_midpoint, Camera, Landmark, Pose, Quaternion,
    SolvePnpMethod, StereoLandmarkObservation, StereoRig, Vector2, Vector3,
};

fn stereo_example() -> Result<pnp_core::Pose, pnp_core::PnpError> {
    let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0)?;
    let right = left.clone();
    // 12 cm horizontal baseline; right_from_left in OpenCV frame.
    let right_from_left = Pose::new(
        Vector3::new(0.12, 0.0, 0.0),
        Quaternion::identity(),
    );
    let rig = StereoRig::new(left, right, right_from_left)?;

    let landmarks = vec![
        Landmark {
            id: "0".into(),
            position: Vector3::new(-0.1, -0.1, 0.0),
        },
        Landmark {
            id: "1".into(),
            position: Vector3::new(0.1, -0.1, 0.0),
        },
        Landmark {
            id: "2".into(),
            position: Vector3::new(0.1, 0.1, 0.0),
        },
        Landmark {
            id: "3".into(),
            position: Vector3::new(-0.1, 0.1, 0.0),
        },
    ];
    // Matched by id. Either eye may be missing (partial observations).
    let observations = vec![
        StereoLandmarkObservation {
            id: "0".into(),
            left: Some(Vector2::new(220.0, 140.0)),
            right: Some(Vector2::new(200.0, 140.0)),
        },
        StereoLandmarkObservation {
            id: "1".into(),
            left: Some(Vector2::new(420.0, 140.0)),
            right: None, // right eye occluded for this landmark
        },
        StereoLandmarkObservation {
            id: "2".into(),
            left: Some(Vector2::new(420.0, 340.0)),
            right: Some(Vector2::new(400.0, 340.0)),
        },
        StereoLandmarkObservation {
            id: "3".into(),
            left: Some(Vector2::new(220.0, 340.0)),
            right: Some(Vector2::new(200.0, 340.0)),
        },
    ];

    // Object pose in OpenGL, primary view = left (same as mono solve_pnp).
    let pose = solve_pnp_stereo(
        &landmarks,
        &observations,
        &rig,
        SolvePnpMethod::Iterative,
    )?;

    // Single correspondence → 3D in the left OpenCV camera frame.
    let _point = triangulate_midpoint(
        &rig,
        Vector2::new(320.0, 240.0),
        Vector2::new(295.0, 240.0),
    )?;

    Ok(pose)
}
```

Pipeline notes:

| Topic | Behavior |
|-------|----------|
| Implementation | Thin wrapper over `solve_pnp_multiview` (N=2) |
| Seed | Monocular PnP preferring **left**; if left is sparse, seed from right when it has enough points and transport into left |
| Refine | Joint LM over left + right reprojection residuals (OpenCV internals) |
| Partial observations | Missing left or right pixels are skipped in the residual |
| Public pose | **OpenGL** object pose, left primary (same meaning as `solve_pnp`) |
| Triangulation | Midpoint of skew rays → left **OpenCV** 3D point |
| Square stereo | `estimate_square_pose_from_stereo_pixels` triangulates corners then fits |

C and Python expose the same stereo surface (`pnp_solve_stereo` /
`pnp_triangulate`, and `auki_pnplab.solve_pnp_stereo` /
`triangulate`). Multi-view is Rust-core today (bindings follow-on). See
[bindings/python/README.md](bindings/python/README.md).

## Camera calibration (multi-view)

Recover monocular intrinsics and optional Brown–Conrady distortion from
**many views** of a known planar target—no OpenCV runtime. The pipeline is
Zhang-style planar initialization (homographies → initial `K`, pose seeds via
`solve_pnp`) followed by joint Levenberg–Marquardt over free intrinsics,
distortion, and per-view poses.

**Views need diversity.** Purely frontal, parallel-to-plane captures are
ill-conditioned; include tilts and varied angles. With only 4 points per view
(e.g. QR outer corners), phones typically need **~8–15 diverse views** in
practice. Single-view calibration is not supported (`min_views` default 3).

**`fix_aspect_ratio = true` (default) is recommended for phones** and most
square-pixel sensors so the solver keeps `fx == fy`. Set it false only when
you know the pixel aspect is anisotropic.

Prefer [`calibrate_from_square_views`](crates/pnp-core/src/calibrate.rs) for
QR / planar quads: pass image corners ordered **TL → TR → BR → BL** and the
marker `physical_size` (meters if your object model is meters). For general
planar or non-square targets (≥4 shared 3D points, same order every view),
use `calibrate_camera`.

Square object model (centered, Z = 0, half-side `h = physical_size / 2`):

| Corner | Object point |
|--------|----------------|
| TL | `(-h, +h, 0)` |
| TR | `(+h, +h, 0)` |
| BR | `(+h, -h, 0)` |
| BL | `(-h, -h, 0)` |

```rust
use pnp_core::{
    calibrate_from_square_views, CalibrateOptions, Vector2,
};

fn calibrate_qr_corners(
    // One [TL, TR, BR, BL] per view — e.g. from a QR detector.
    corners_per_view: &[[Vector2; 4]],
) -> Result<pnp_core::CalibrationResult, pnp_core::PnpError> {
    // Prefer many tilted views; 4 pts/view needs angle diversity.
    // Phones: keep fix_aspect_ratio (default true). dist_len: 0|2|4|5|8.
    let options = CalibrateOptions {
        fix_aspect_ratio: true,
        dist_len: 5,
        ..CalibrateOptions::default()
    };

    let result = calibrate_from_square_views(
        corners_per_view,
        /* physical_size */ 0.05, // marker side length (meters if object is meters)
        /* image_width */ 1920,
        /* image_height */ 1080,
        &options,
    )?;

    // result.camera: fx, fy, cx, cy, dist (OpenCV order)
    // result.rms_reprojection_error / per_view_rms
    // result.object_poses: OpenGL object poses (same as solve_pnp)
    Ok(result)
}
```

Pipeline notes:

| Topic | Behavior |
|-------|----------|
| Init | Planar homographies → Zhang `K` (≥3 views); pose seeds via monocular PnP |
| Refine | Joint LM over free intrinsics, distortion, and per-view rvec/tvec |
| Distortion | OpenCV Brown–Conrady; `dist_len` ∈ `{0, 2, 4, 5, 8}` (2 packs as `[k1,k2,0,0,0]`) |
| Public poses | **OpenGL** object poses per used view (same as `solve_pnp`) |
| Image size | Required (`image_width` / `image_height`) for principal-point init |
| Defaults | `min_views=3`, `fix_aspect_ratio=true`, `dist_len=5` |

**Bindings:** Python (`auki_pnplab.calibrate_from_square_views` /
`calibrate_camera`) and C FFI (`pnp_calibrate_from_square_views`,
`pnp_calibrate_options_t`) are available. **WASM and Expo calibration
bindings are deferred** for a follow-on. See
[bindings/python/README.md](bindings/python/README.md) and
`crates/pnp-ffi/include/pnp.h`.

## Coordinate conventions

| Topic | Convention |
|-------|------------|
| Image pixels | OpenCV-style: origin top-left, +X right, +Y down |
| `solve_pnp` / `solve_pnp_multiview` / `solve_pnp_stereo` result | **OpenGL** object pose (Y-up, Z-backward) |
| Multi-view primary frame | **`views[0]`** (returned pose; preferred seed) |
| `CameraView.from_primary` | Pose of view in primary frame, **OpenCV** convention |
| Stereo primary frame | **Left** camera (returned pose, triangulation origin; preferred seed) |
| `StereoRig.right_from_left` | Pose of right in left frame, **OpenCV** convention (= multiview `from_primary`) |
| Algebraic solvers (internal) | OpenCV camera frame (+Z forward) then converted |
| Distortion | OpenCV Brown–Conrady / rational: `k1,k2,p1,p2[,k3[,k4,k5,k6]]` |
| Square corners | Top-left → top-right → bottom-right → bottom-left |
| OpenGL rays from pixels | `x=(u-cx)/fx`, `y=-(v-cy)/fy`, `z=-1` after optional undistort |
| OpenCV rays (stereo / triangulate) | `x=(u-cx)/fx`, `y=(v-cy)/fy`, `z=1` after optional undistort |
| `triangulate_midpoint` output | Left camera **OpenCV** frame (+Z forward) |

## Repository layout

```text
crates/
  pnp-core/     Pure Rust solvers (no_std + alloc)
  pnp-ffi/      C ABI + cbindgen header
  pnp-wasm/     WASM Component Model guest (WIT)

bindings/
  python/       Maturin / PyO3 package (aukilabs-pnplab)
  expo-pnp/     Expo module + prebuilt Android/iOS natives

scripts/        Cross-target build and check helpers
tests/          OpenCV reference vectors and generator
```

## Development commands

| Command | Purpose |
|---------|---------|
| `just test` | Workspace tests (`--locked`) |
| `just check-nostd` | Verify `pnp-core` without `std` |
| `just check-all` | Tests + no_std + clippy |
| `just python-build` / `just python-test` | Python wheel and integration tests |
| `just expo-android` / `just expo-ios` / `just expo-native` | Mobile artifacts → Expo package |
| `just build-wasm-release` | WASM component |
| `just generate-reference` | Regenerate OpenCV reference JSON |

CI (GitHub Actions) runs on every push and pull request: `cargo fmt`, workspace
tests, `no_std` check, clippy, and the Python integration suite
(`.github/workflows/ci.yml`).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
coding guidelines, testing, and pull-request expectations.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2025–2026 Auki Labs.

## Security

Please report vulnerabilities privately. See [SECURITY.md](SECURITY.md).
