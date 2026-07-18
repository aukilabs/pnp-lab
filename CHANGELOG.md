# Changelog

## Unreleased — pnpkit extraction

- Extracted the PnP workspace from the peyote app into a standalone monorepo.
- Grouped the Expo module under `bindings/expo-pnp`, separate from the core
  Rust library crates.
- Prebuilt Android `.so` libraries and the iOS XCFramework install into the
  Expo package (same consumer flow as QRKit).
- Added the `aukilabs-pnpkit` Python/NumPy package under `bindings/python`
  with `solve_pnp`, `solve_pnp_camera_pose`,
  `camera_pose_from_solve_pnp_pose`, and `estimate_square_pose_from_rays`.
- **Breaking:** public solve APIs now take a first-class `Camera`
  (`fx/fy/cx/cy` + optional OpenCV `dist`) instead of a bare `Matrix3x3`.
  Distorted pixels are undistorted before algebraic solvers. Added
  `estimate_square_pose_from_pixels`, `Camera::project` /
  `undistort_pixel` / `unproject_opengl_ray`. WIT package bumped to
  `auki:pnp@0.2.0`.
- **Stereo (calibrated left+right):**
  - Core: `StereoRig` / `StereoLandmarkObservation`, `solve_pnp_stereo` /
    `solve_pnp_stereo_camera_pose` (mono left seed + joint LM), midpoint
    `triangulate_midpoint`, `estimate_square_pose_from_stereo_pixels`, and
    Umeyama `absolute_orientation` helper. Left is primary; public stereo
    poses are **OpenGL** (same as mono); `right_from_left` is **OpenCV**
    extrinsics; triangulation returns left-OpenCV 3D. Partial observations
    (missing left or right pixel) are supported in the joint residual.
  - C FFI: `pnp_stereo_rig_t`, `pnp_stereo_observation_t`,
    `peyote_pnp_solve_stereo`, `peyote_pnp_triangulate` (header via cbindgen).
  - Python: `solve_pnp_stereo`, `solve_pnp_stereo_camera_pose`, `triangulate`.
  - Out of scope for this release: stereo rectification, disparity matching,
    WIT stereo exports, Expo dual-camera product wiring.
