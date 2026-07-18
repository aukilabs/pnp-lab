//! Multi-view monocular camera calibration: public types, validation, square
//! planar object points, and planar homography DLT.
//!
//! Zhang-style initialization and joint bundle adjustment land in later tasks.
//! This module exposes configuration / result types, input checks, and a
//! private plane-to-image homography estimator used by the calibration init.

use crate::camera::Camera;
use crate::types::{PnpError, Pose, Vector2, Vector3};
use alloc::vec::Vec;
use nalgebra::{DMatrix, Matrix3};

/// Options controlling multi-view intrinsic + distortion calibration.
///
/// Defaults match the locked design for phone/QR calibration:
/// `min_views = 3`, `fix_aspect_ratio = true`, `dist_len = 5`.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateOptions {
    /// Minimum number of views required (default 3). Single-view is not supported.
    pub min_views: usize,
    /// When true, force `fx == fy` during estimation (default true).
    pub fix_aspect_ratio: bool,
    /// When true, hold principal point fixed (default false).
    pub fix_principal_point: bool,
    /// Number of free distortion coefficients: `0 | 2 | 4 | 5 | 8`.
    ///
    /// Length 2 is packed into a length-5 `Camera.dist` as `[k1, k2, 0, 0, 0]`.
    pub dist_len: usize,
    /// Maximum Levenberg–Marquardt iterations (default 100).
    pub max_iterations: usize,
    /// LM function-tolerance stopping criterion (default `1e-10`).
    pub function_tolerance: f64,
    /// If `Some(t)`, fail when overall RMS reprojection exceeds `t`.
    /// `None` means no hard RMS failure (default).
    pub rms_success_threshold: Option<f64>,
}

impl Default for CalibrateOptions {
    fn default() -> Self {
        Self {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 5,
            max_iterations: 100,
            function_tolerance: 1e-10,
            rms_success_threshold: None,
        }
    }
}

/// One calibration view: image observations of the shared object points.
///
/// `image_points` must be parallel to the shared object-point list (same order
/// every view). Pixel coordinates use the OpenCV convention (origin top-left,
/// +X right, +Y down).
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationView {
    /// Distorted image points matching the shared object-point order.
    pub image_points: Vec<Vector2>,
}

/// Result of multi-view monocular calibration.
///
/// `object_poses` are **OpenGL** object poses (same convention as
/// [`crate::solve_pnp`]), one per view that contributed to the solution.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResult {
    /// Estimated camera intrinsics and distortion.
    pub camera: Camera,
    /// Overall RMS reprojection error in pixels.
    pub rms_reprojection_error: f64,
    /// Per-view RMS reprojection error (same order as used views).
    pub per_view_rms: Vec<f64>,
    /// OpenGL object pose per used view (same convention as [`crate::solve_pnp`]).
    pub object_poses: Vec<Pose>,
    /// Number of views used in the solution.
    pub views_used: usize,
}

/// Shared planar square object points for side length `physical_size`.
///
/// The square is centered on the origin in the Z = 0 plane. Half-side
/// `h = physical_size / 2`. Corner order is **TL → TR → BR → BL**:
///
/// | Corner | Object point   |
/// |--------|----------------|
/// | TL     | `(-h, +h, 0)`  |
/// | TR     | `(+h, +h, 0)`  |
/// | BR     | `(+h, -h, 0)`  |
/// | BL     | `(-h, -h, 0)`  |
///
/// Local +X runs TL→TR; local +Y points toward the top of the marker.
/// Units match `physical_size` (typically meters).
///
/// # Errors
///
/// Returns [`PnpError::SolverFailed`] if `physical_size` is non-finite or ≤ 0.
pub fn square_object_points(physical_size: f64) -> Result<[Vector3; 4], PnpError> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PnpError::SolverFailed);
    }
    let h = physical_size / 2.0;
    Ok([
        Vector3::new(-h, h, 0.0),  // TL
        Vector3::new(h, h, 0.0),   // TR
        Vector3::new(h, -h, 0.0),  // BR
        Vector3::new(-h, -h, 0.0), // BL
    ])
}

/// Validate shared object points, views, image size, and options for calibration.
///
/// # Errors
///
/// | Condition | Error |
/// |-----------|--------|
/// | `image_width < 2` or `image_height < 2` | [`PnpError::SolverFailed`] |
/// | `views.len() < options.min_views` | [`PnpError::InsufficientPoints`] |
/// | any view `image_points.len() != object_points.len()` | [`PnpError::MismatchedCounts`] |
/// | `object_points.len() < 4` | [`PnpError::InsufficientPoints`] |
/// | non-finite object or image coordinates | [`PnpError::SolverFailed`] |
/// | `dist_len` not in `{0, 2, 4, 5, 8}` | [`PnpError::SolverFailed`] |
// Used by the public calibrate entry points (Task 3+); keep reachable from tests now.
#[allow(dead_code)]
pub(crate) fn validate_calibrate_inputs(
    object_points: &[Vector3],
    views: &[CalibrationView],
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
) -> Result<(), PnpError> {
    if image_width < 2 || image_height < 2 {
        return Err(PnpError::SolverFailed);
    }
    if views.len() < options.min_views {
        return Err(PnpError::InsufficientPoints);
    }
    if object_points.len() < 4 {
        return Err(PnpError::InsufficientPoints);
    }
    if !matches!(options.dist_len, 0 | 2 | 4 | 5 | 8) {
        return Err(PnpError::SolverFailed);
    }

    for p in object_points {
        if !is_finite_vec3(p) {
            return Err(PnpError::SolverFailed);
        }
    }

    let n = object_points.len();
    for view in views {
        if view.image_points.len() != n {
            return Err(PnpError::MismatchedCounts);
        }
        for ip in &view.image_points {
            if !ip.x.is_finite() || !ip.y.is_finite() {
                return Err(PnpError::SolverFailed);
            }
        }
    }

    Ok(())
}

#[allow(dead_code)] // only referenced via validate_calibrate_inputs for now
fn is_finite_vec3(v: &Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

// ---------------------------------------------------------------------------
// Homography (planar Z=0) — DLT
// ---------------------------------------------------------------------------

/// Minimum correspondences for a planar homography.
const HOMOGRAPHY_MIN_POINTS: usize = 4;
/// Relative area threshold for rejecting collinear 2D point sets.
const HOMOGRAPHY_COLLINEAR_EPS: f64 = 1e-12;
/// Floor for `|H[2,2]|` when normalizing so `H[(2, 2)] = 1`.
const HOMOGRAPHY_H22_EPS: f64 = 1e-12;

/// Estimate a plane-to-image homography via Direct Linear Transform (DLT).
///
/// Maps planar object coordinates `(X, Y)` on the Z = 0 plane to image pixels
/// `(u, v)`:
///
/// ```text
/// [u v 1]^T  ~  H [X Y 1]^T
/// ```
///
/// Returns a [`nalgebra::Matrix3`] (column-major storage; access element
/// `(r, c)` as `H[(r, c)]`). When `|H[(2, 2)]|` is not near zero the matrix is
/// scaled so that `H[(2, 2)] = 1`.
///
/// Internally applies Hartley isotropic normalization of both point sets,
/// builds the classic 2n×9 design matrix, and takes the null-space of `A`
/// (eigenvector of `AᵀA` for the smallest eigenvalue) as `H`.
///
/// # Errors
///
/// | Condition | Error |
/// |-----------|--------|
/// | length mismatch | [`PnpError::MismatchedCounts`] |
/// | fewer than 4 pairs | [`PnpError::InsufficientPoints`] |
/// | non-finite coordinates | [`PnpError::SolverFailed`] |
/// | collinear object or image points, or degenerate `H` | [`PnpError::SolverFailed`] |
// Used by Zhang init (Task 3+); keep reachable from unit tests now.
#[allow(dead_code)]
fn estimate_homography_dlt(
    object_xy: &[(f64, f64)],
    image_uv: &[Vector2],
) -> Result<Matrix3<f64>, PnpError> {
    if object_xy.len() != image_uv.len() {
        return Err(PnpError::MismatchedCounts);
    }
    let n = object_xy.len();
    if n < HOMOGRAPHY_MIN_POINTS {
        return Err(PnpError::InsufficientPoints);
    }

    for &(x, y) in object_xy {
        if !x.is_finite() || !y.is_finite() {
            return Err(PnpError::SolverFailed);
        }
    }
    for p in image_uv {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(PnpError::SolverFailed);
        }
    }

    if points_2d_collinear(object_xy) {
        return Err(PnpError::SolverFailed);
    }
    let image_xy: Vec<(f64, f64)> = image_uv.iter().map(|p| (p.x, p.y)).collect();
    if points_2d_collinear(&image_xy) {
        return Err(PnpError::SolverFailed);
    }

    // Hartley isotropic normalization for numerical stability.
    let (t_obj, obj_n) = normalize_points_2d(object_xy)?;
    let (t_img, img_n) = normalize_points_2d(&image_xy)?;

    // Design matrix A (2n × 9): A h = 0, h row-major of H_norm.
    // Null-space via eigen of AᵀA (full 9×9): thin SVD of A is only
    // min(2n,9)-rank and misses the kernel when n = 4 (8 < 9).
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let (x, y) = obj_n[i];
        let (u, v) = img_n[i];
        let r0 = 2 * i;
        let r1 = r0 + 1;
        // [ X Y 1  0 0 0  -uX -uY -u ]
        a[(r0, 0)] = x;
        a[(r0, 1)] = y;
        a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -u * x;
        a[(r0, 7)] = -u * y;
        a[(r0, 8)] = -u;
        // [ 0 0 0  X Y 1  -vX -vY -v ]
        a[(r1, 3)] = x;
        a[(r1, 4)] = y;
        a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -v * x;
        a[(r1, 7)] = -v * y;
        a[(r1, 8)] = -v;
    }

    let ata = a.transpose() * &a;
    let eigen = ata.symmetric_eigen();
    let eigenvalues = &eigen.eigenvalues;
    let mut min_idx = 0usize;
    let mut min_val = eigenvalues[0];
    for i in 1..9 {
        if eigenvalues[i] < min_val {
            min_val = eigenvalues[i];
            min_idx = i;
        }
    }
    // Reject if the "null" eigenvalue is not small relative to the largest
    // (rank-deficient correspondence set / pure noise).
    let mut max_val = eigenvalues[0].abs();
    for i in 1..9 {
        max_val = max_val.max(eigenvalues[i].abs());
    }
    if max_val < HOMOGRAPHY_H22_EPS {
        return Err(PnpError::SolverFailed);
    }

    let h_vec = eigen.eigenvectors.column(min_idx);
    let h_norm = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7], h_vec[8],
    );

    if !h_norm.iter().all(|e| e.is_finite()) || h_norm.norm() < HOMOGRAPHY_H22_EPS {
        return Err(PnpError::SolverFailed);
    }

    // Denormalize: H = T_img^{-1} H_norm T_obj
    let t_img_inv = t_img.try_inverse().ok_or(PnpError::SolverFailed)?;
    let mut h = t_img_inv * h_norm * t_obj;

    if !h.iter().all(|e| e.is_finite()) {
        return Err(PnpError::SolverFailed);
    }

    // Prefer H[2,2] = 1 when possible; else unit Frobenius norm.
    let h22 = h[(2, 2)];
    if h22.abs() > HOMOGRAPHY_H22_EPS {
        h /= h22;
    } else {
        let nrm = h.norm();
        if nrm < HOMOGRAPHY_H22_EPS {
            return Err(PnpError::SolverFailed);
        }
        h /= nrm;
    }

    if !h.iter().all(|e| e.is_finite()) {
        return Err(PnpError::SolverFailed);
    }

    Ok(h)
}

/// True if all 2D points are (nearly) collinear.
fn points_2d_collinear(pts: &[(f64, f64)]) -> bool {
    if pts.len() < 3 {
        return true;
    }
    // Use the first two distinct points as a baseline; if none, all coincide.
    let (x0, y0) = pts[0];
    let mut base: Option<(f64, f64)> = None;
    for &(x, y) in &pts[1..] {
        let dx = x - x0;
        let dy = y - y0;
        if dx * dx + dy * dy > HOMOGRAPHY_COLLINEAR_EPS {
            base = Some((dx, dy));
            break;
        }
    }
    let Some((bx, by)) = base else {
        return true; // all points identical
    };
    let base_len2 = bx * bx + by * by;
    for &(x, y) in pts {
        let dx = x - x0;
        let dy = y - y0;
        // |cross| / |base| as relative out-of-line measure vs span.
        let cross = bx * dy - by * dx;
        if cross.abs() > HOMOGRAPHY_COLLINEAR_EPS * base_len2.max(1.0) {
            return false;
        }
    }
    true
}

/// Hartley isotropic normalization: centroid to origin, mean distance √2.
///
/// Returns `(T, normalized_points)` where `T` maps original → normalized:
/// `p_n ~ T p`.
fn normalize_points_2d(pts: &[(f64, f64)]) -> Result<(Matrix3<f64>, Vec<(f64, f64)>), PnpError> {
    let n = pts.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for &(x, y) in pts {
        cx += x;
        cy += y;
    }
    cx /= n;
    cy /= n;

    let mut mean_dist = 0.0;
    for &(x, y) in pts {
        let dx = x - cx;
        let dy = y - cy;
        mean_dist += libm::sqrt(dx * dx + dy * dy);
    }
    mean_dist /= n;
    if mean_dist < HOMOGRAPHY_H22_EPS {
        return Err(PnpError::SolverFailed);
    }

    let s = libm::sqrt(2.0) / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);

    let mut out = Vec::with_capacity(pts.len());
    for &(x, y) in pts {
        out.push((s * (x - cx), s * (y - cy)));
    }
    Ok((t, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn square_object_points_order_and_size() {
        let pts = square_object_points(0.2).unwrap();
        let h = 0.1;
        assert!((pts[0].x + h).abs() < 1e-12 && (pts[0].y - h).abs() < 1e-12); // TL
        assert!((pts[1].x - h).abs() < 1e-12 && (pts[1].y - h).abs() < 1e-12); // TR
        assert!((pts[2].x - h).abs() < 1e-12 && (pts[2].y + h).abs() < 1e-12); // BR
        assert!((pts[3].x + h).abs() < 1e-12 && (pts[3].y + h).abs() < 1e-12); // BL
                                                                               // side length
        let d = ((pts[1].x - pts[0].x).powi(2) + (pts[1].y - pts[0].y).powi(2)).sqrt();
        assert!((d - 0.2).abs() < 1e-12);
        // Z = 0 plane
        for p in &pts {
            assert!(p.z.abs() < 1e-12);
        }
    }

    #[test]
    fn square_object_points_rejects_bad_size() {
        assert_eq!(square_object_points(0.0), Err(PnpError::SolverFailed));
        assert_eq!(square_object_points(-0.1), Err(PnpError::SolverFailed));
        assert_eq!(square_object_points(f64::NAN), Err(PnpError::SolverFailed));
        assert_eq!(
            square_object_points(f64::INFINITY),
            Err(PnpError::SolverFailed)
        );
    }

    #[test]
    fn validate_rejects_too_few_views_and_bad_dist_len() {
        let opts = CalibrateOptions {
            min_views: 3,
            ..Default::default()
        };
        let obj = square_object_points(0.1).unwrap();
        let views = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 4],
            };
            2
        ];
        assert_eq!(
            validate_calibrate_inputs(&obj, &views, 640, 480, &opts),
            Err(PnpError::InsufficientPoints)
        );
        let mut bad = CalibrateOptions::default();
        bad.dist_len = 3;
        let views3 = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 4],
            };
            3
        ];
        assert!(validate_calibrate_inputs(&obj, &views3, 640, 480, &bad).is_err());
    }

    #[test]
    fn validate_accepts_valid_inputs() {
        let opts = CalibrateOptions::default();
        let obj = square_object_points(0.1).unwrap();
        let views = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 4],
            };
            3
        ];
        assert!(validate_calibrate_inputs(&obj, &views, 640, 480, &opts).is_ok());
    }

    #[test]
    fn validate_rejects_small_image_mismatched_and_nonfinite() {
        let opts = CalibrateOptions::default();
        let obj = square_object_points(0.1).unwrap();
        let views = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 4],
            };
            3
        ];
        assert_eq!(
            validate_calibrate_inputs(&obj, &views, 1, 480, &opts),
            Err(PnpError::SolverFailed)
        );
        assert_eq!(
            validate_calibrate_inputs(&obj, &views, 640, 0, &opts),
            Err(PnpError::SolverFailed)
        );

        let short = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 3],
            };
            3
        ];
        assert_eq!(
            validate_calibrate_inputs(&obj, &short, 640, 480, &opts),
            Err(PnpError::MismatchedCounts)
        );

        let few_obj = [Vector3::new(0.0, 0.0, 0.0); 3];
        let views_3pt = vec![
            CalibrationView {
                image_points: vec![Vector2::new(0.0, 0.0); 3],
            };
            3
        ];
        assert_eq!(
            validate_calibrate_inputs(&few_obj, &views_3pt, 640, 480, &opts),
            Err(PnpError::InsufficientPoints)
        );

        let mut bad_obj = obj;
        bad_obj[0].x = f64::NAN;
        assert_eq!(
            validate_calibrate_inputs(&bad_obj, &views, 640, 480, &opts),
            Err(PnpError::SolverFailed)
        );

        let mut bad_views = views.clone();
        bad_views[1].image_points[2].y = f64::INFINITY;
        assert_eq!(
            validate_calibrate_inputs(&obj, &bad_views, 640, 480, &opts),
            Err(PnpError::SolverFailed)
        );
    }

    #[test]
    fn calibrate_options_defaults() {
        let d = CalibrateOptions::default();
        assert_eq!(d.min_views, 3);
        assert!(d.fix_aspect_ratio);
        assert!(!d.fix_principal_point);
        assert_eq!(d.dist_len, 5);
        assert_eq!(d.max_iterations, 100);
        assert!((d.function_tolerance - 1e-10).abs() < 1e-30);
        assert_eq!(d.rms_success_threshold, None);
    }

    /// Apply H to planar (X,Y): returns inhomogeneous image (u,v).
    fn apply_h(h: &Matrix3<f64>, x: f64, y: f64) -> Vector2 {
        let p = h * nalgebra::Vector3::new(x, y, 1.0);
        Vector2::new(p.x / p.z, p.y / p.z)
    }

    fn homographies_close(a: &Matrix3<f64>, b: &Matrix3<f64>, tol: f64) -> bool {
        // Both expected with H[2,2] = 1 (or comparable scale).
        let scale_a = if a[(2, 2)].abs() > 1e-12 {
            a[(2, 2)]
        } else {
            a.norm()
        };
        let scale_b = if b[(2, 2)].abs() > 1e-12 {
            b[(2, 2)]
        } else {
            b.norm()
        };
        let an = a / scale_a;
        let bn = b / scale_b;
        (an - bn).norm() < tol
    }

    #[test]
    fn homography_recovers_similarity() {
        // Object square corners (Z=0 plane), physical side 0.2 m.
        let h_side = 0.1;
        let object_xy = [
            (-h_side, h_side),  // TL
            (h_side, h_side),   // TR
            (h_side, -h_side),  // BR
            (-h_side, -h_side), // BL
        ];

        // Synthetic H = K * [r1 r2 t] for a mild pose (OpenCV camera frame).
        let fx = 800.0;
        let fy = 800.0;
        let cx = 320.0;
        let cy = 240.0;
        let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        // R ≈ small yaw/pitch; columns r1, r2, r3.
        let yaw = 0.15_f64;
        let pitch = -0.10_f64;
        let cy_ = libm::cos(yaw);
        let sy = libm::sin(yaw);
        let cp = libm::cos(pitch);
        let sp = libm::sin(pitch);
        // R = Ry(yaw) * Rx(pitch)
        let r = Matrix3::new(cy_, sy * sp, sy * cp, 0.0, cp, -sp, -sy, cy_ * sp, cy_ * cp);
        let t = nalgebra::Vector3::new(0.02, -0.01, 0.55);
        let mut h_true = Matrix3::zeros();
        h_true.set_column(0, &r.column(0));
        h_true.set_column(1, &r.column(1));
        h_true.set_column(2, &t);
        h_true = k * h_true;
        h_true /= h_true[(2, 2)];

        let image_uv: Vec<Vector2> = object_xy
            .iter()
            .map(|&(x, y)| apply_h(&h_true, x, y))
            .collect();

        let h_est = estimate_homography_dlt(&object_xy, &image_uv).unwrap();
        assert!(
            homographies_close(&h_est, &h_true, 1e-8),
            "H_est =\n{h_est}\nH_true =\n{h_true}"
        );

        // Reprojection of object corners should match image points.
        for (i, &(x, y)) in object_xy.iter().enumerate() {
            let p = apply_h(&h_est, x, y);
            assert!((p.x - image_uv[i].x).abs() < 1e-8);
            assert!((p.y - image_uv[i].y).abs() < 1e-8);
        }
    }

    #[test]
    fn homography_rejects_too_few_mismatched_collinear() {
        let obj4 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let img3 = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        ];
        assert_eq!(
            estimate_homography_dlt(&obj4, &img3),
            Err(PnpError::MismatchedCounts)
        );

        let obj3 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let img3b = [
            Vector2::new(10.0, 10.0),
            Vector2::new(20.0, 10.0),
            Vector2::new(10.0, 20.0),
        ];
        assert_eq!(
            estimate_homography_dlt(&obj3, &img3b),
            Err(PnpError::InsufficientPoints)
        );

        // Collinear object points (4 points on a line).
        let collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let img4 = [
            Vector2::new(0.0, 0.0),
            Vector2::new(10.0, 5.0),
            Vector2::new(20.0, 15.0),
            Vector2::new(30.0, 40.0),
        ];
        assert_eq!(
            estimate_homography_dlt(&collinear, &img4),
            Err(PnpError::SolverFailed)
        );

        let mut bad = obj4;
        bad[0].0 = f64::NAN;
        let img4_ok = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        assert_eq!(
            estimate_homography_dlt(&bad, &img4_ok),
            Err(PnpError::SolverFailed)
        );
    }
}
