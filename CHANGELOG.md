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
