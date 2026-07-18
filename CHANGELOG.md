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
- **Multi-view joint PnP (N ≥ 1 calibrated views):**
  - Core: `MultiViewRig` / `CameraView` / `MultiViewObservation`,
    `solve_pnp_multiview` / `solve_pnp_multiview_camera_pose` (mono seed +
    joint LM over all-view residuals). Primary is `views[0]`; per-view
    `from_primary` is **OpenCV** extrinsics (`X_view = R * X_primary + t`);
    public poses are **OpenGL** (same as mono). Sparse observations use an
    aligned `pixels` vec with `None` holes. Seed prefers primary; otherwise
    the richest view with enough points, transported into primary.
  - Stereo is N=2: `MultiViewRig::from_stereo` / `StereoRig::to_multiview`.
  - Out of scope for this release: multiview C/Python/WIT/Expo bindings,
    free-extrinsic BA, N-view square pose.
- **Stereo (calibrated left+right):**
  - Core: `StereoRig` / `StereoLandmarkObservation`, `solve_pnp_stereo` /
    `solve_pnp_stereo_camera_pose` (thin wrappers over multiview; no
    intentional behavior change), midpoint `triangulate_midpoint`,
    `estimate_square_pose_from_stereo_pixels`, and Umeyama
    `absolute_orientation` helper. Left is primary; public stereo poses are
    **OpenGL** (same as mono); `right_from_left` is **OpenCV** extrinsics;
    triangulation returns left-OpenCV 3D. Partial observations (missing left
    or right pixel) are supported in the joint residual. Seed prefers left;
    right-only seeds when right has enough points for the method.
  - C FFI: `pnp_stereo_rig_t`, `pnp_stereo_observation_t`,
    `peyote_pnp_solve_stereo`, `peyote_pnp_triangulate` (header via cbindgen).
  - Python: `solve_pnp_stereo`, `solve_pnp_stereo_camera_pose`, `triangulate`.
  - Out of scope for this release: stereo rectification, disparity matching,
    WIT stereo exports, Expo dual-camera product wiring.
- **Multi-view monocular camera calibration:**
  - Core: `CalibrateOptions` / `CalibrationView` / `CalibrationResult`,
    `calibrate_camera` (shared object points + multi-view pixels),
    `calibrate_from_square_views` (QR / planar quad convenience; corners
    **TL→TR→BR→BL** + `physical_size`), Zhang planar init + joint LM
    over free intrinsics, Brown–Conrady distortion (`dist_len` 0/2/4/5/8),
    and per-view poses. Defaults: `min_views=3`, `fix_aspect_ratio=true`
    (recommended for phones), `dist_len=5`. Public `object_poses` are
    **OpenGL** (same as `solve_pnp`). Requires diverse views (tilts); pure
    frontal parallel planes are ill-conditioned. Single-view is not supported.
  - Python: `calibrate_from_square_views`, `calibrate_camera`.
  - C FFI: `pnp_calibrate_options_t`, `peyote_pnp_calibrate_from_square_views`.
  - Out of scope for this release: WASM WIT calibration, Expo/TS calibration
    wrappers, single-view calibration, multi-camera extrinsics, OpenCV runtime.
