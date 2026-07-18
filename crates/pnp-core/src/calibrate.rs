//! Multi-view monocular camera calibration: Zhang-style init + joint BA.
//!
//! Public entry points: [`calibrate_camera`], [`calibrate_from_square_views`].
//! Internals cover validation, square planar object points, planar homography
//! DLT, Zhang initial intrinsics, BA parameter packing / RMS, and joint
//! Levenberg–Marquardt refinement.
//!
//! # Pipeline ([`calibrate_camera`])
//!
//! 1. Validate inputs  
//! 2. If object points are planar (`Z ≈ 0`), estimate a homography per view  
//! 3. Zhang closed-form `K` from ≥3 homographies, else focal/image-center fallback  
//! 4. Apply option flags (`fix_aspect_ratio`, `fix_principal_point`)  
//! 5. Seed zero distortion of packed length  
//! 6. Per-view [`crate::solve_pnp`] pose seeds (OpenCV rvec/tvec); drop failures  
//! 7. Joint LM refine of free intrinsics, distortion, and poses  
//! 8. Optional hard fail on RMS; return [`CalibrationResult`] (OpenGL poses)
//!
//! # Joint LM (v1)
//!
//! [`refine_calibration_lm`] optimizes free calibration parameters with
//! Levenberg–Marquardt damping matching `multiview_solve` / `iterative`
//! (`lambda` on `diag(JᵀJ)`, accept/reject steps, cost-delta tolerance).
//!
//! The Jacobian is **finite differences** on free parameters (central
//! difference, step `1e-6`). Analytic Jacobians can replace this later
//! without changing the state layout.

use crate::camera::Camera;
use crate::pose_tools;
use crate::rodrigues;
use crate::solve::solve_pnp;
use crate::types::{
    rotation_matrix_to_quaternion, Landmark, LandmarkObservation, Matrix3x3, PnpError, Pose,
    SolvePnpMethod, Vector2, Vector3,
};
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3 as NaVector3};

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

fn is_finite_vec3(v: &Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Absolute threshold for treating object points as planar on Z = 0.
const PLANAR_Z_EPS: f64 = 1e-9;

/// True when every object point has `|Z| < PLANAR_Z_EPS` (Zhang path eligible).
fn object_points_planar_z0(object_points: &[Vector3]) -> bool {
    object_points.iter().all(|p| p.z.abs() < PLANAR_Z_EPS)
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

// ---------------------------------------------------------------------------
// Zhang initial intrinsics (zero skew)
// ---------------------------------------------------------------------------

/// Minimum number of plane-to-image homographies for Zhang's closed form.
const ZHANG_MIN_HOMOGRAPHIES: usize = 3;
/// Floor for |B11|, |B22|, lambda, and recovered focals.
const ZHANG_B_EPS: f64 = 1e-12;

/// Fallback pinhole guess used when Zhang's system is ill-conditioned.
///
/// ```text
/// fx = fy = max(image_width, image_height)
/// cx = image_width / 2
/// cy = image_height / 2
/// ```
///
/// The public calibrate entry point may also use this when the homography
/// stage fails to produce ≥3 valid views.
fn zhang_fallback_intrinsics(image_width: u32, image_height: u32) -> (f64, f64, f64, f64) {
    let w = image_width as f64;
    let h = image_height as f64;
    let f = w.max(h);
    (f, f, w * 0.5, h * 0.5)
}

/// Zhang closed-form initial intrinsics from ≥3 plane-to-image homographies.
///
/// Recovers `(fx, fy, cx, cy)` with **zero skew** from the image of the absolute
/// conic constraints on each `Hᵢ = K [r1 r2 t]` (object plane Z = 0):
///
/// ```text
/// h1ᵀ ω h2 = 0
/// h1ᵀ ω h1 = h2ᵀ ω h2
/// ```
///
/// where `ω = (K Kᵀ)⁻¹` and `B12 = 0` (skew freezes the off-diagonal of the
/// upper 2×2 of the symmetric matrix `B ~ ω`).
///
/// # Ill-conditioned fallback
///
/// If the stacked constraint system or the closed-form recovery of `K` from
/// `B` is numerically invalid (non-finite `H`, rank-deficient `VᵀV`, negative
/// scale/focal squares, non-finite results), returns the documented fallback:
///
/// ```text
/// fx = fy = max(w, h);  cx = w/2;  cy = h/2
/// ```
/// as `Ok(...)` so the joint BA can still start from a coarse pinhole seed.
///
/// # Errors
///
/// | Condition | Error |
/// |-----------|--------|
/// | fewer than 3 homographies | [`PnpError::InsufficientPoints`] |
fn zhang_initial_intrinsics(
    homographies: &[Matrix3<f64>],
    image_width: u32,
    image_height: u32,
) -> Result<(f64, f64, f64, f64), PnpError> {
    if homographies.len() < ZHANG_MIN_HOMOGRAPHIES {
        return Err(PnpError::InsufficientPoints);
    }

    for h in homographies {
        if !h.iter().all(|e| e.is_finite()) {
            return Ok(zhang_fallback_intrinsics(image_width, image_height));
        }
    }

    // Zero-skew B = [[B11, 0, B13], [0, B22, B23], [B13, B23, B33]].
    // Each H contributes two rows of V b = 0 with b = [B11, B22, B13, B23, B33].
    let n = homographies.len();
    let mut v = DMatrix::<f64>::zeros(2 * n, 5);
    for (k, h) in homographies.iter().enumerate() {
        let v12 = v_ij_zeroskew(h, 0, 1);
        let v11 = v_ij_zeroskew(h, 0, 0);
        let v22 = v_ij_zeroskew(h, 1, 1);
        let r0 = 2 * k;
        let r1 = r0 + 1;
        for c in 0..5 {
            v[(r0, c)] = v12[c];
            v[(r1, c)] = v11[c] - v22[c];
        }
    }

    let vtv = v.transpose() * &v;
    let eigen = vtv.symmetric_eigen();
    let eigenvalues = &eigen.eigenvalues;
    let mut min_idx = 0usize;
    let mut min_val = eigenvalues[0];
    let mut max_abs = eigenvalues[0].abs();
    for i in 1..5 {
        if eigenvalues[i] < min_val {
            min_val = eigenvalues[i];
            min_idx = i;
        }
        max_abs = max_abs.max(eigenvalues[i].abs());
    }
    if max_abs < ZHANG_B_EPS {
        return Ok(zhang_fallback_intrinsics(image_width, image_height));
    }

    let b = eigen.eigenvectors.column(min_idx);
    if !b.iter().all(|e| e.is_finite()) {
        return Ok(zhang_fallback_intrinsics(image_width, image_height));
    }

    match recover_k_from_b_zeroskew(b[0], b[1], b[2], b[3], b[4]) {
        Some(k) => Ok(k),
        None => Ok(zhang_fallback_intrinsics(image_width, image_height)),
    }
}

/// Coefficient row for `h_iᵀ B h_j` with zero-skew `B` (no B12 term).
///
/// Columns of `H` are Zhang's `h0, h1, h2`. Returns
/// `[B11, B22, B13, B23, B33]` coefficients.
fn v_ij_zeroskew(h: &Matrix3<f64>, i: usize, j: usize) -> [f64; 5] {
    let hi0 = h[(0, i)];
    let hi1 = h[(1, i)];
    let hi2 = h[(2, i)];
    let hj0 = h[(0, j)];
    let hj1 = h[(1, j)];
    let hj2 = h[(2, j)];
    [
        hi0 * hj0,             // B11
        hi1 * hj1,             // B22
        hi2 * hj0 + hi0 * hj2, // B13
        hi2 * hj1 + hi1 * hj2, // B23
        hi2 * hj2,             // B33
    ]
}

/// Closed-form `(fx, fy, cx, cy)` from zero-skew absolute-conic coefficients.
///
/// Tries both signs of `b` (null-space orientation is arbitrary).
fn recover_k_from_b_zeroskew(
    b11: f64,
    b22: f64,
    b13: f64,
    b23: f64,
    b33: f64,
) -> Option<(f64, f64, f64, f64)> {
    for sign in [1.0_f64, -1.0_f64] {
        let b11 = sign * b11;
        let b22 = sign * b22;
        let b13 = sign * b13;
        let b23 = sign * b23;
        let b33 = sign * b33;

        if b11.abs() < ZHANG_B_EPS || b22.abs() < ZHANG_B_EPS {
            continue;
        }

        // OpenCV/Zhang recovery with B12 = 0:
        //   cy = -B23/B22
        //   λ  = B33 - B13²/B11 - B23²/B22
        //   fx = √(λ/B11), fy = √(λ/B22)
        //   cx = -B13/B11
        let cy = -b23 / b22;
        let lambda = b33 - (b13 * b13) / b11 - (b23 * b23) / b22;
        if lambda / b11 <= ZHANG_B_EPS || lambda / b22 <= ZHANG_B_EPS {
            continue;
        }

        let fx = libm::sqrt(lambda / b11);
        let fy = libm::sqrt(lambda / b22);
        let cx = -b13 / b11;

        if !(fx.is_finite() && fy.is_finite() && cx.is_finite() && cy.is_finite()) {
            continue;
        }
        if fx < ZHANG_B_EPS || fy < ZHANG_B_EPS {
            continue;
        }
        return Some((fx, fy, cx, cy));
    }
    None
}

// ---------------------------------------------------------------------------
// Parameter packing + RMS (joint BA state)
// ---------------------------------------------------------------------------

/// OpenCV object-in-camera pose as Rodrigues `rvec` + translation `tvec`.
type CvRvecTvec = ([f64; 3], NaVector3<f64>);

/// Free-parameter layout for joint calibration BA.
///
/// Flat state vector order:
///
/// ```text
/// [fx, (fy)?, (cx, cy)?, dist_free[0..dist_len],
///  rvec0[3], tvec0[3], …, rvec_{N-1}[3], tvec_{N-1}[3]]
/// ```
///
/// - `fy` is omitted when [`CalibrateOptions::fix_aspect_ratio`] (then `fy = fx`).
/// - `cx, cy` are omitted when [`CalibrateOptions::fix_principal_point`]
///   (locked to [`Self::fixed_cx`] / [`Self::fixed_cy`]).
/// - `dist_free` has length [`CalibrateOptions::dist_len`] (`0 | 2 | 4 | 5 | 8`);
///   packing into [`Camera::dist`] follows the table in [`camera_from_params`].
/// - Per-view poses are OpenCV object-in-camera: Rodrigues `rvec` + `tvec`.
#[derive(Debug, Clone, PartialEq)]
struct CalibLayout {
    fix_aspect_ratio: bool,
    fix_principal_point: bool,
    /// Free distortion coefficient count: `0 | 2 | 4 | 5 | 8`.
    dist_len: usize,
    n_views: usize,
    /// Principal point used when `fix_principal_point` is true.
    fixed_cx: f64,
    fixed_cy: f64,
}

impl CalibLayout {
    /// Build layout from options, view count, and fixed principal point.
    ///
    /// `fixed_cx` / `fixed_cy` are only applied when
    /// `options.fix_principal_point` (typically image center or seed `K`).
    fn new(options: &CalibrateOptions, n_views: usize, fixed_cx: f64, fixed_cy: f64) -> Self {
        Self {
            fix_aspect_ratio: options.fix_aspect_ratio,
            fix_principal_point: options.fix_principal_point,
            dist_len: options.dist_len,
            n_views,
            fixed_cx,
            fixed_cy,
        }
    }

    /// Number of free intrinsic scalars (excluding distortion).
    fn n_intrinsic_params(&self) -> usize {
        let mut n = 1; // fx always free
        if !self.fix_aspect_ratio {
            n += 1; // fy
        }
        if !self.fix_principal_point {
            n += 2; // cx, cy
        }
        n
    }

    /// Total free parameters in the packed state vector.
    fn n_params(&self) -> usize {
        self.n_intrinsic_params() + self.dist_len + self.n_views * 6
    }
}

/// Pack free distortion coefficients from a [`Camera`] for a given `dist_len`.
fn free_dist_from_camera(camera: &Camera, dist_len: usize) -> Vec<f64> {
    let mut d = vec![0.0; dist_len];
    let n = dist_len.min(camera.dist.len());
    d[..n].copy_from_slice(&camera.dist[..n]);
    d
}

/// Build a [`Camera`] from free intrinsics and free distortion coefficients.
///
/// Distortion packing into stored [`Camera::dist`]:
///
/// | `dist_len` | Stored `Camera.dist` |
/// |------------|----------------------|
/// | 0 | `[]` |
/// | 2 | `[k1, k2, 0, 0, 0]` (length 5) |
/// | 4 | `[k1, k2, p1, p2]` |
/// | 5 | `[k1, k2, p1, p2, k3]` |
/// | 8 | full rational 8 |
///
/// # Errors
///
/// Returns [`PnpError::SolverFailed`] if `dist_len` is unsupported, `dist_free`
/// length mismatches `dist_len`, or [`Camera::new`] rejects the values.
fn camera_from_params(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    dist_free: &[f64],
    dist_len: usize,
) -> Result<Camera, PnpError> {
    if !matches!(dist_len, 0 | 2 | 4 | 5 | 8) {
        return Err(PnpError::SolverFailed);
    }
    if dist_free.len() != dist_len {
        return Err(PnpError::SolverFailed);
    }
    for c in dist_free {
        if !c.is_finite() {
            return Err(PnpError::SolverFailed);
        }
    }

    let dist: Vec<f64> = match dist_len {
        0 => Vec::new(),
        2 => vec![dist_free[0], dist_free[1], 0.0, 0.0, 0.0],
        4 | 5 | 8 => dist_free.to_vec(),
        _ => return Err(PnpError::SolverFailed),
    };
    Camera::new(fx, fy, cx, cy, &dist)
}

/// Pack free calibration parameters into a flat state vector.
///
/// `poses_cv` are OpenCV object-in-camera `(rvec, tvec)` pairs, one per view.
/// Length must equal `layout.n_views`. When aspect is fixed, only `camera.fx`
/// is packed (`fy` is forced to `fx` on unpack). When the principal point is
/// fixed, `cx`/`cy` are not packed (see [`CalibLayout::fixed_cx`]).
fn pack_state(layout: &CalibLayout, camera: &Camera, poses_cv: &[CvRvecTvec]) -> Vec<f64> {
    debug_assert_eq!(poses_cv.len(), layout.n_views);
    let mut params = Vec::with_capacity(layout.n_params());

    params.push(camera.fx);
    if !layout.fix_aspect_ratio {
        params.push(camera.fy);
    }
    if !layout.fix_principal_point {
        params.push(camera.cx);
        params.push(camera.cy);
    }

    let free_dist = free_dist_from_camera(camera, layout.dist_len);
    params.extend_from_slice(&free_dist);

    for (rvec, tvec) in poses_cv.iter().take(layout.n_views) {
        params.push(rvec[0]);
        params.push(rvec[1]);
        params.push(rvec[2]);
        params.push(tvec.x);
        params.push(tvec.y);
        params.push(tvec.z);
    }

    params
}

/// Unpack a flat state vector into a [`Camera`] and per-view OpenCV poses.
///
/// # Errors
///
/// Returns [`PnpError::SolverFailed`] if `params.len()` does not match the
/// layout, or if [`camera_from_params`] rejects the values.
fn unpack_state(
    params: &[f64],
    layout: &CalibLayout,
) -> Result<(Camera, Vec<CvRvecTvec>), PnpError> {
    if params.len() != layout.n_params() {
        return Err(PnpError::SolverFailed);
    }
    if !params.iter().all(|p| p.is_finite()) {
        return Err(PnpError::SolverFailed);
    }

    let mut i = 0usize;
    let fx = params[i];
    i += 1;
    let fy = if layout.fix_aspect_ratio {
        fx
    } else {
        let v = params[i];
        i += 1;
        v
    };
    let (cx, cy) = if layout.fix_principal_point {
        (layout.fixed_cx, layout.fixed_cy)
    } else {
        let cx = params[i];
        let cy = params[i + 1];
        i += 2;
        (cx, cy)
    };

    let dist_free = &params[i..i + layout.dist_len];
    i += layout.dist_len;

    let camera = camera_from_params(fx, fy, cx, cy, dist_free, layout.dist_len)?;

    let mut poses = Vec::with_capacity(layout.n_views);
    for _ in 0..layout.n_views {
        let rvec = [params[i], params[i + 1], params[i + 2]];
        let tvec = NaVector3::new(params[i + 3], params[i + 4], params[i + 5]);
        i += 6;
        poses.push((rvec, tvec));
    }

    Ok((camera, poses))
}

/// Convert an OpenCV [`Pose`] to Rodrigues `rvec` + `tvec`.
fn pose_cv_to_rvec_tvec(pose: &Pose) -> CvRvecTvec {
    let r_mat = pose.rotation.normalize().to_na_unit().to_rotation_matrix();
    let m = Matrix3x3::from_na(r_mat.matrix());
    let rvec = rodrigues::rotation_matrix_to_rvec(&m);
    let tvec = NaVector3::new(pose.position.x, pose.position.y, pose.position.z);
    (rvec, tvec)
}

/// Convert Rodrigues `rvec` + `tvec` to an OpenCV [`Pose`].
fn rvec_tvec_to_pose_cv(rvec: &[f64; 3], tvec: &NaVector3<f64>) -> Pose {
    let rot_m = rodrigues::rvec_to_rotation_matrix(rvec);
    let q = rotation_matrix_to_quaternion(&rot_m);
    Pose::new(Vector3::new(tvec.x, tvec.y, tvec.z), q)
}

/// Overall and per-view RMS reprojection error using [`Camera::project`].
///
/// For each point `j` in view `i` (OpenCV object-in-camera pose):
///
/// ```text
/// X_cam = R_i * X_j + t_i
/// u_hat = camera.project(X_cam)   // already in distorted pixel space
/// e     = u_hat - image_points[i][j]
/// ```
///
/// Returns `(overall_rms, per_view_rms)` where RMS is
/// `sqrt(sum ||e||² / n_points)` over the respective point set (OpenCV-style).
/// Points that fail to project (behind the camera) contribute a large residual.
///
/// `poses_cv` and `views` must have the same length; each view's image list
/// should match `object_points` length (callers validate inputs earlier).
fn rms_reprojection(
    object_points: &[Vector3],
    views: &[CalibrationView],
    camera: &Camera,
    poses_cv: &[CvRvecTvec],
) -> (f64, Vec<f64>) {
    let n_views = views.len().min(poses_cv.len());
    let mut sum_sq_all = 0.0;
    let mut count_all = 0usize;
    let mut per_view = Vec::with_capacity(n_views);

    for v in 0..n_views {
        let (rvec, tvec) = &poses_cv[v];
        let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
        let img = &views[v].image_points;
        let n = object_points.len().min(img.len());
        let mut sum_sq = 0.0;
        let mut count = 0usize;

        for j in 0..n {
            let pw = object_points[j].to_na();
            let pc = r * pw + tvec;
            let x_cam = Vector3::from_na(&pc);
            match camera.project(x_cam) {
                Some(proj) => {
                    let dx = proj.x - img[j].x;
                    let dy = proj.y - img[j].y;
                    sum_sq += dx * dx + dy * dy;
                }
                None => {
                    sum_sq += 1e12;
                }
            }
            count += 1;
        }

        let view_rms = if count > 0 {
            libm::sqrt(sum_sq / count as f64)
        } else {
            0.0
        };
        per_view.push(view_rms);
        sum_sq_all += sum_sq;
        count_all += count;
    }

    let overall = if count_all > 0 {
        libm::sqrt(sum_sq_all / count_all as f64)
    } else {
        0.0
    };
    (overall, per_view)
}

// ---------------------------------------------------------------------------
// Joint Levenberg–Marquardt refinement
// ---------------------------------------------------------------------------

const LM_LAMBDA0: f64 = 1e-3;
const LM_LAMBDA_FACTOR: f64 = 10.0;
const LM_PARAM_CONV: f64 = 1e-8;
const LM_FD_EPS: f64 = 1e-6;
/// Residual magnitude when a point fails to project (behind camera).
const LM_BAD_RESIDUAL: f64 = 1e6;

/// Stacked reprojection residuals: length `2 * n_views * n_points`.
///
/// Order is view-major, then point-major, then `(dx, dy)` per point — same
/// residual definition as [`rms_reprojection`].
fn calib_residuals(
    object_points: &[Vector3],
    views: &[CalibrationView],
    camera: &Camera,
    poses_cv: &[CvRvecTvec],
) -> DVector<f64> {
    let n_views = views.len().min(poses_cv.len());
    let n_pts = object_points.len();
    let mut residuals = DVector::zeros(2 * n_views * n_pts);
    let mut row = 0usize;

    for v in 0..n_views {
        let (rvec, tvec) = &poses_cv[v];
        let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
        let img = &views[v].image_points;
        let n = n_pts.min(img.len());

        for j in 0..n_pts {
            if j < n {
                let pw = object_points[j].to_na();
                let pc = r * pw + tvec;
                let x_cam = Vector3::from_na(&pc);
                match camera.project(x_cam) {
                    Some(proj) => {
                        residuals[row] = proj.x - img[j].x;
                        residuals[row + 1] = proj.y - img[j].y;
                    }
                    None => {
                        residuals[row] = LM_BAD_RESIDUAL;
                        residuals[row + 1] = LM_BAD_RESIDUAL;
                    }
                }
            } else {
                residuals[row] = LM_BAD_RESIDUAL;
                residuals[row + 1] = LM_BAD_RESIDUAL;
            }
            row += 2;
        }
    }

    residuals
}

fn residual_cost(residuals: &DVector<f64>) -> f64 {
    residuals.iter().map(|r| r * r).sum()
}

/// Residuals for a packed state vector (large residual vector if unpack fails).
fn residuals_from_params(
    params: &[f64],
    layout: &CalibLayout,
    object_points: &[Vector3],
    views: &[CalibrationView],
) -> DVector<f64> {
    let n = 2 * layout.n_views * object_points.len();
    match unpack_state(params, layout) {
        Ok((camera, poses)) => calib_residuals(object_points, views, &camera, &poses),
        Err(_) => DVector::from_element(n, LM_BAD_RESIDUAL),
    }
}

/// Central finite-difference Jacobian of residuals w.r.t. free packed params.
fn calib_jacobian_fd(
    params: &[f64],
    layout: &CalibLayout,
    object_points: &[Vector3],
    views: &[CalibrationView],
) -> DMatrix<f64> {
    let n_res = 2 * layout.n_views * object_points.len();
    let n_par = params.len();
    let mut jac = DMatrix::zeros(n_res, n_par);

    for k in 0..n_par {
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[k] += LM_FD_EPS;
        minus[k] -= LM_FD_EPS;

        let res_p = residuals_from_params(&plus, layout, object_points, views);
        let res_m = residuals_from_params(&minus, layout, object_points, views);

        let denom = 2.0 * LM_FD_EPS;
        for i in 0..n_res {
            jac[(i, k)] = (res_p[i] - res_m[i]) / denom;
        }
    }

    jac
}

/// Joint LM refine of intrinsics (+ free distortion) and per-view OpenCV poses.
///
/// State layout is [`CalibLayout`] / [`pack_state`]. Jacobian uses finite
/// differences (see module docs). Pose seed `poses_cv0` is OpenCV
/// object-in-camera `(rvec, tvec)`. Returned poses are **OpenGL** (same
/// convention as [`crate::solve_pnp`]).
///
/// When `options.fix_principal_point`, `cx`/`cy` are locked to image center
/// (`image_width/2`, `image_height/2`).
///
/// # Errors
///
/// - View / pose count mismatch, empty views, or non-finite seed →
///   [`PnpError::SolverFailed`]
/// - Non-finite refined parameters → [`PnpError::SolverFailed`]
/// - `options.rms_success_threshold` exceeded after refine →
///   [`PnpError::SolverFailed`]
fn refine_calibration_lm(
    object_points: &[Vector3],
    views: &[CalibrationView],
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
    camera0: Camera,
    poses_cv0: Vec<CvRvecTvec>,
) -> Result<(Camera, Vec<Pose>, f64, Vec<f64>), PnpError> {
    if views.is_empty() || views.len() != poses_cv0.len() {
        return Err(PnpError::SolverFailed);
    }
    if object_points.is_empty() {
        return Err(PnpError::SolverFailed);
    }
    if !matches!(options.dist_len, 0 | 2 | 4 | 5 | 8) {
        return Err(PnpError::SolverFailed);
    }

    let fixed_cx = image_width as f64 * 0.5;
    let fixed_cy = image_height as f64 * 0.5;
    let layout = CalibLayout::new(options, views.len(), fixed_cx, fixed_cy);

    // Apply fixed flags on the seed camera so pack/unpack stay consistent.
    let mut seed_cam = camera0;
    if options.fix_aspect_ratio {
        seed_cam.fy = seed_cam.fx;
    }
    if options.fix_principal_point {
        seed_cam.cx = fixed_cx;
        seed_cam.cy = fixed_cy;
    }

    let mut params = pack_state(&layout, &seed_cam, &poses_cv0);
    if !params.iter().all(|p| p.is_finite()) {
        return Err(PnpError::SolverFailed);
    }

    let mut lambda = LM_LAMBDA0;
    let mut prev_cost = residual_cost(&residuals_from_params(
        &params,
        &layout,
        object_points,
        views,
    ));
    if !prev_cost.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    let n_par = params.len();
    let max_iters = options.max_iterations.max(1);
    let ftol = options.function_tolerance;

    for _ in 0..max_iters {
        let residuals = residuals_from_params(&params, &layout, object_points, views);
        let jacobian = calib_jacobian_fd(&params, &layout, object_points, views);

        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        let mut a = jtj.clone();
        for i in 0..n_par {
            a[(i, i)] += lambda * jtj[(i, i)].max(1e-10);
        }

        let neg_jtr = -&jtr;
        let delta = match a.lu().solve(&neg_jtr) {
            Some(d) => d,
            None => break,
        };

        if delta.norm() < LM_PARAM_CONV {
            break;
        }

        let mut trial = params.clone();
        for i in 0..n_par {
            trial[i] += delta[i];
        }

        let new_cost = residual_cost(&residuals_from_params(
            &trial,
            &layout,
            object_points,
            views,
        ));

        if new_cost.is_finite() && new_cost < prev_cost {
            params = trial;
            lambda = (lambda / LM_LAMBDA_FACTOR).max(1e-10);
            let rel = (prev_cost - new_cost) / prev_cost.max(1e-15);
            if rel < ftol {
                break;
            }
            prev_cost = new_cost;
        } else {
            lambda *= LM_LAMBDA_FACTOR;
            if lambda > 1e16 {
                break;
            }
        }
    }

    let (camera, poses_cv) = unpack_state(&params, &layout)?;
    if !camera.fx.is_finite()
        || !camera.fy.is_finite()
        || !camera.cx.is_finite()
        || !camera.cy.is_finite()
        || camera.dist.iter().any(|d| !d.is_finite())
    {
        return Err(PnpError::SolverFailed);
    }
    for (rvec, tvec) in &poses_cv {
        if !rvec.iter().all(|v| v.is_finite())
            || !tvec.x.is_finite()
            || !tvec.y.is_finite()
            || !tvec.z.is_finite()
        {
            return Err(PnpError::SolverFailed);
        }
    }

    let (overall_rms, per_view_rms) = rms_reprojection(object_points, views, &camera, &poses_cv);
    if !overall_rms.is_finite() {
        return Err(PnpError::SolverFailed);
    }
    if let Some(t) = options.rms_success_threshold {
        if overall_rms > t {
            return Err(PnpError::SolverFailed);
        }
    }

    let object_poses: Vec<Pose> = poses_cv
        .iter()
        .map(|(rvec, tvec)| {
            let cv = rvec_tvec_to_pose_cv(rvec, tvec);
            pose_tools::from_opencv_to_opengl(&cv)
        })
        .collect();

    Ok((camera, object_poses, overall_rms, per_view_rms))
}

// ---------------------------------------------------------------------------
// Public pipeline
// ---------------------------------------------------------------------------

/// Multi-view monocular calibration: Zhang (or fallback) init + joint LM.
///
/// Estimates a single shared [`Camera`] (intrinsics + optional distortion) and
/// per-view **OpenGL** object poses from many observations of the same known
/// 3D object points.
///
/// # Inputs
///
/// - `object_points`: shared 3D model points (same order every view). Planar
///   targets with `Z ≈ 0` enable Zhang homography init.
/// - `views`: one [`CalibrationView`] per image; `image_points` parallel to
///   `object_points`. Pixels are OpenCV (top-left origin, +Y down), distorted.
/// - `image_width` / `image_height`: sensor size in pixels (principal-point
///   fallback / fixed-pp lock use the image center).
/// - `options`: free-parameter flags, `dist_len`, LM controls, optional RMS
///   hard fail.
///
/// # Pipeline
///
/// 1. [`validate_calibrate_inputs`]  
/// 2. If planar `Z≈0`, DLT homography per view (failed views skip Zhang only)  
/// 3. ≥3 valid `H` → Zhang `K`; else `fx=fy=max(w,h)`, `cx=w/2`, `cy=h/2`  
/// 4. Apply `fix_aspect_ratio` / `fix_principal_point`  
/// 5. Seed `Camera` with zero free distortion of length `dist_len`  
/// 6. [`solve_pnp`] (Iterative) per view → OpenCV rvec/tvec; drop PnP failures  
/// 7. If remaining views `< min_views` → [`PnpError::InsufficientPoints`]  
/// 8. [`refine_calibration_lm`] (OpenGL poses out)  
/// 9. Optional `rms_success_threshold` → [`PnpError::SolverFailed`]
///
/// # Errors
///
/// Propagates validation, init, PnP-insufficient-views, LM, and RMS failures
/// as [`PnpError`].
pub fn calibrate_camera(
    object_points: &[Vector3],
    views: &[CalibrationView],
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
) -> Result<CalibrationResult, PnpError> {
    validate_calibrate_inputs(object_points, views, image_width, image_height, options)?;

    // Zhang path only when the model is planar on Z = 0.
    let mut homographies = Vec::new();
    if object_points_planar_z0(object_points) {
        let object_xy: Vec<(f64, f64)> = object_points.iter().map(|p| (p.x, p.y)).collect();
        for view in views {
            if let Ok(h) = estimate_homography_dlt(&object_xy, &view.image_points) {
                homographies.push(h);
            }
        }
    }

    let (fx, mut fy, mut cx, mut cy) = if homographies.len() >= ZHANG_MIN_HOMOGRAPHIES {
        zhang_initial_intrinsics(&homographies, image_width, image_height)?
    } else {
        zhang_fallback_intrinsics(image_width, image_height)
    };

    if options.fix_aspect_ratio {
        fy = fx;
    }
    if options.fix_principal_point {
        cx = image_width as f64 * 0.5;
        cy = image_height as f64 * 0.5;
    }

    // Zero free-distortion seed; packing maps dist_len into Camera.dist.
    let dist_free = vec![0.0; options.dist_len];
    let camera0 = camera_from_params(fx, fy, cx, cy, &dist_free, options.dist_len)?;

    let landmarks: Vec<Landmark> = object_points
        .iter()
        .enumerate()
        .map(|(i, p)| Landmark {
            id: i.to_string(),
            position: *p,
        })
        .collect();

    let mut used_views: Vec<CalibrationView> = Vec::with_capacity(views.len());
    let mut poses_cv0: Vec<CvRvecTvec> = Vec::with_capacity(views.len());

    for view in views {
        let observations: Vec<LandmarkObservation> = view
            .image_points
            .iter()
            .enumerate()
            .map(|(i, uv)| LandmarkObservation {
                id: i.to_string(),
                position: *uv,
            })
            .collect();

        match solve_pnp(
            &landmarks,
            &observations,
            &camera0,
            SolvePnpMethod::Iterative,
        ) {
            Ok(pose_gl) => {
                let pose_cv = pose_tools::from_opengl_to_opencv(&pose_gl);
                poses_cv0.push(pose_cv_to_rvec_tvec(&pose_cv));
                used_views.push(CalibrationView {
                    image_points: view.image_points.clone(),
                });
            }
            Err(_) => {
                // Drop views that fail monocular PnP with the seed camera.
            }
        }
    }

    if used_views.len() < options.min_views {
        return Err(PnpError::InsufficientPoints);
    }

    let (camera, object_poses, rms, per_view_rms) = refine_calibration_lm(
        object_points,
        &used_views,
        image_width,
        image_height,
        options,
        camera0,
        poses_cv0,
    )?;

    Ok(CalibrationResult {
        camera,
        rms_reprojection_error: rms,
        per_view_rms,
        object_poses,
        views_used: used_views.len(),
    })
}

/// Convenience: calibrate from multiple views of a square marker (QR / planar quad).
///
/// Builds shared object points via [`square_object_points`] (`physical_size` side
/// length, TL→TR→BR→BL) and runs [`calibrate_camera`]. Each entry of
/// `corners_per_view` is four image corners in the same order.
///
/// # Errors
///
/// - [`PnpError::SolverFailed`] if `physical_size` is non-finite or ≤ 0  
/// - Propagates all [`calibrate_camera`] errors otherwise
pub fn calibrate_from_square_views(
    corners_per_view: &[[Vector2; 4]],
    physical_size: f64,
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
) -> Result<CalibrationResult, PnpError> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PnpError::SolverFailed);
    }
    let object = square_object_points(physical_size)?;
    let views: Vec<CalibrationView> = corners_per_view
        .iter()
        .map(|c| CalibrationView {
            image_points: c.to_vec(),
        })
        .collect();
    calibrate_camera(&object, &views, image_width, image_height, options)
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

    /// Build OpenCV R = Ry(yaw) * Rx(pitch) * Rz(roll).
    fn rot_ypr(yaw: f64, pitch: f64, roll: f64) -> Matrix3<f64> {
        let (cy, sy) = (libm::cos(yaw), libm::sin(yaw));
        let (cp, sp) = (libm::cos(pitch), libm::sin(pitch));
        let (cr, sr) = (libm::cos(roll), libm::sin(roll));
        let ry = Matrix3::new(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
        let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
        let rz = Matrix3::new(cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0);
        ry * rx * rz
    }

    /// Plane-to-image H = K [r1 r2 t] (object Z = 0), scaled so H[2,2] = 1.
    fn homography_from_pose(
        k: &Matrix3<f64>,
        r: &Matrix3<f64>,
        t: &nalgebra::Vector3<f64>,
    ) -> Matrix3<f64> {
        let mut h = Matrix3::zeros();
        h.set_column(0, &r.column(0));
        h.set_column(1, &r.column(1));
        h.set_column(2, t);
        h = k * h;
        let h22 = h[(2, 2)];
        if h22.abs() > 1e-12 {
            h /= h22;
        }
        h
    }

    #[test]
    fn zhang_recovers_intrinsics_five_poses() {
        // True pinhole (zero skew, mild aspect). Non-centered principal point.
        let fx_t = 920.0;
        let fy_t = 905.0;
        let cx_t = 640.0;
        let cy_t = 360.0;
        let w = 1280u32;
        let h = 720u32;
        let k = Matrix3::new(fx_t, 0.0, cx_t, 0.0, fy_t, cy_t, 0.0, 0.0, 1.0);

        // Planar square + interior samples (Z=0), side 0.2 m.
        let half = 0.1;
        let object_xy: [(f64, f64); 9] = [
            (-half, half),
            (0.0, half),
            (half, half),
            (-half, 0.0),
            (0.0, 0.0),
            (half, 0.0),
            (-half, -half),
            (0.0, -half),
            (half, -half),
        ];

        // ≥5 diverse non-frontal poses (Zhang needs tilt / yaw diversity).
        let poses: [(f64, f64, f64, f64, f64, f64); 6] = [
            (0.25, -0.18, 0.05, 0.01, -0.02, 0.55),
            (-0.30, 0.22, -0.08, -0.03, 0.01, 0.62),
            (0.15, 0.28, 0.12, 0.02, 0.03, 0.48),
            (-0.20, -0.25, 0.0, -0.01, -0.02, 0.70),
            (0.35, 0.10, -0.15, 0.0, 0.02, 0.58),
            (-0.12, 0.32, 0.18, 0.03, -0.01, 0.52),
        ];

        let mut homographies = Vec::with_capacity(poses.len());
        for &(yaw, pitch, roll, tx, ty, tz) in &poses {
            let r = rot_ypr(yaw, pitch, roll);
            let t = nalgebra::Vector3::new(tx, ty, tz);
            let h_true = homography_from_pose(&k, &r, &t);

            // Project via true H, recover H with DLT (pipeline path).
            let image_uv: Vec<Vector2> = object_xy
                .iter()
                .map(|&(x, y)| apply_h(&h_true, x, y))
                .collect();
            let h_est = estimate_homography_dlt(&object_xy, &image_uv).unwrap();
            assert!(
                homographies_close(&h_est, &h_true, 1e-6),
                "DLT H should match synthetic projection H"
            );
            homographies.push(h_est);
        }

        let (fx, fy, cx, cy) =
            zhang_initial_intrinsics(&homographies, w, h).expect("Zhang should succeed");

        // ~1% on focals, few pixels on principal point (noise-free synthetic).
        assert!((fx - fx_t).abs() / fx_t < 0.01, "fx={fx} true={fx_t}");
        assert!((fy - fy_t).abs() / fy_t < 0.01, "fy={fy} true={fy_t}");
        assert!((cx - cx_t).abs() < 3.0, "cx={cx} true={cx_t}");
        assert!((cy - cy_t).abs() < 3.0, "cy={cy} true={cy_t}");
    }

    #[test]
    fn zhang_rejects_too_few_and_falls_back_on_bad_h() {
        let w = 640u32;
        let h = 480u32;
        let (ffx, ffy, fcx, fcy) = zhang_fallback_intrinsics(w, h);
        assert!((ffx - 640.0).abs() < 1e-12);
        assert!((ffy - 640.0).abs() < 1e-12);
        assert!((fcx - 320.0).abs() < 1e-12);
        assert!((fcy - 240.0).abs() < 1e-12);

        // Fewer than 3 homographies → InsufficientPoints (not silent fallback).
        let two = [Matrix3::identity(), Matrix3::identity()];
        assert_eq!(
            zhang_initial_intrinsics(&two, w, h),
            Err(PnpError::InsufficientPoints)
        );

        // Non-finite H → documented fallback Ok(...).
        let mut bad = Matrix3::identity();
        bad[(0, 0)] = f64::NAN;
        let three_bad = [bad, Matrix3::identity(), Matrix3::identity()];
        let (fx, fy, cx, cy) = zhang_initial_intrinsics(&three_bad, w, h).unwrap();
        assert!((fx - ffx).abs() < 1e-12);
        assert!((fy - ffy).abs() < 1e-12);
        assert!((cx - fcx).abs() < 1e-12);
        assert!((cy - fcy).abs() < 1e-12);
    }

    fn cameras_close(a: &Camera, b: &Camera, tol: f64) -> bool {
        (a.fx - b.fx).abs() < tol
            && (a.fy - b.fy).abs() < tol
            && (a.cx - b.cx).abs() < tol
            && (a.cy - b.cy).abs() < tol
            && a.dist.len() == b.dist.len()
            && a.dist
                .iter()
                .zip(b.dist.iter())
                .all(|(x, y)| (x - y).abs() < tol)
    }

    fn poses_rvec_tvec_close(a: &[CvRvecTvec], b: &[CvRvecTvec], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for ((ra, ta), (rb, tb)) in a.iter().zip(b.iter()) {
            for k in 0..3 {
                if (ra[k] - rb[k]).abs() > tol {
                    return false;
                }
            }
            if (ta - tb).norm() > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn pack_unpack_roundtrip_default_options() {
        let opts = CalibrateOptions {
            fix_aspect_ratio: false,
            fix_principal_point: false,
            dist_len: 5,
            ..Default::default()
        };
        let cam = Camera::new(
            900.0,
            910.0,
            640.0,
            360.0,
            &[0.01, -0.02, 0.001, -0.001, 0.005],
        )
        .unwrap();
        let poses = vec![
            ([0.1, -0.2, 0.05], NaVector3::new(0.01, -0.02, 0.5)),
            ([-0.15, 0.1, 0.0], NaVector3::new(-0.03, 0.01, 0.6)),
            ([0.0, 0.25, -0.1], NaVector3::new(0.0, 0.0, 0.55)),
        ];
        let layout = CalibLayout::new(&opts, poses.len(), 640.0, 360.0);
        // free aspect + free pp: fx,fy,cx,cy + 5 dist + 3*6 = 4+5+18 = 27
        assert_eq!(layout.n_params(), 27);

        let packed = pack_state(&layout, &cam, &poses);
        assert_eq!(packed.len(), layout.n_params());
        let (cam2, poses2) = unpack_state(&packed, &layout).unwrap();
        assert!(cameras_close(&cam, &cam2, 1e-12));
        assert!(poses_rvec_tvec_close(&poses, &poses2, 1e-12));
    }

    #[test]
    fn pack_unpack_fix_aspect_and_principal() {
        let opts = CalibrateOptions {
            fix_aspect_ratio: true,
            fix_principal_point: true,
            dist_len: 0,
            ..Default::default()
        };
        // Seed has fy != fx and off-center pp; packing drops them.
        let cam = Camera::pinhole(800.0, 850.0, 310.0, 250.0).unwrap();
        let poses = vec![
            ([0.0, 0.1, 0.0], NaVector3::new(0.0, 0.0, 0.5)),
            ([0.2, -0.1, 0.05], NaVector3::new(0.02, -0.01, 0.6)),
        ];
        let fixed_cx = 320.0;
        let fixed_cy = 240.0;
        let layout = CalibLayout::new(&opts, poses.len(), fixed_cx, fixed_cy);
        // fx only + 0 dist + 2*6 = 13
        assert_eq!(layout.n_params(), 13);

        let packed = pack_state(&layout, &cam, &poses);
        let (cam2, poses2) = unpack_state(&packed, &layout).unwrap();
        assert!((cam2.fx - 800.0).abs() < 1e-12);
        assert!((cam2.fy - 800.0).abs() < 1e-12); // fy forced = fx
        assert!((cam2.cx - fixed_cx).abs() < 1e-12);
        assert!((cam2.cy - fixed_cy).abs() < 1e-12);
        assert!(cam2.dist.is_empty());
        assert!(poses_rvec_tvec_close(&poses, &poses2, 1e-12));
    }

    #[test]
    fn camera_from_params_dist_packing_table() {
        // dist_len 0 → []
        let c0 = camera_from_params(800.0, 800.0, 320.0, 240.0, &[], 0).unwrap();
        assert!(c0.dist.is_empty());

        // dist_len 2 → [k1,k2,0,0,0]
        let c2 = camera_from_params(800.0, 800.0, 320.0, 240.0, &[0.1, -0.05], 2).unwrap();
        assert_eq!(c2.dist, vec![0.1, -0.05, 0.0, 0.0, 0.0]);

        // dist_len 4
        let d4 = [0.1, -0.05, 0.001, -0.002];
        let c4 = camera_from_params(800.0, 800.0, 320.0, 240.0, &d4, 4).unwrap();
        assert_eq!(c4.dist, d4.to_vec());

        // dist_len 5
        let d5 = [0.1, -0.05, 0.001, -0.002, 0.01];
        let c5 = camera_from_params(800.0, 800.0, 320.0, 240.0, &d5, 5).unwrap();
        assert_eq!(c5.dist, d5.to_vec());

        // dist_len 8
        let d8 = [0.1, -0.05, 0.001, -0.002, 0.01, 0.0, 0.0, 0.002];
        let c8 = camera_from_params(800.0, 800.0, 320.0, 240.0, &d8, 8).unwrap();
        assert_eq!(c8.dist, d8.to_vec());

        // reject bad dist_len / length mismatch
        assert!(camera_from_params(800.0, 800.0, 320.0, 240.0, &[0.1], 2).is_err());
        assert!(camera_from_params(800.0, 800.0, 320.0, 240.0, &[], 3).is_err());
    }

    #[test]
    fn pack_unpack_dist_len_2() {
        let opts = CalibrateOptions {
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 2,
            ..Default::default()
        };
        let cam = camera_from_params(700.0, 700.0, 400.0, 300.0, &[0.12, -0.08], 2).unwrap();
        let poses = vec![([0.05, -0.1, 0.02], NaVector3::new(0.0, 0.0, 0.7))];
        let layout = CalibLayout::new(&opts, 1, 400.0, 300.0);
        // fx + cx,cy + 2 dist + 6 = 11
        assert_eq!(layout.n_params(), 11);
        let packed = pack_state(&layout, &cam, &poses);
        let (cam2, poses2) = unpack_state(&packed, &layout).unwrap();
        assert!(cameras_close(&cam, &cam2, 1e-12));
        assert!(poses_rvec_tvec_close(&poses, &poses2, 1e-12));
    }

    #[test]
    fn rms_near_zero_perfect_synthetic() {
        let fx = 800.0;
        let fy = 800.0;
        let cx = 320.0;
        let cy = 240.0;
        // Mild radial distortion so project path exercises dist.
        let cam = Camera::new(fx, fy, cx, cy, &[0.05, -0.01, 0.0, 0.0, 0.0]).unwrap();
        let object = square_object_points(0.2).unwrap();

        let pose_specs: [CvRvecTvec; 3] = [
            ([0.15, -0.10, 0.05], NaVector3::new(0.02, -0.01, 0.55)),
            ([-0.20, 0.18, -0.08], NaVector3::new(-0.03, 0.02, 0.62)),
            ([0.10, 0.25, 0.12], NaVector3::new(0.01, 0.0, 0.48)),
        ];

        let mut views = Vec::with_capacity(pose_specs.len());
        for (rvec, tvec) in &pose_specs {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut image_points = Vec::with_capacity(4);
            for p in &object {
                let pc = r * p.to_na() + tvec;
                let x_cam = Vector3::from_na(&pc);
                let uv = cam.project(x_cam).expect("in front of camera");
                image_points.push(uv);
            }
            views.push(CalibrationView { image_points });
        }

        let (overall, per_view) = rms_reprojection(&object, &views, &cam, &pose_specs);
        assert!(
            overall < 1e-10,
            "perfect synthetic overall RMS should be ~0, got {overall}"
        );
        assert_eq!(per_view.len(), 3);
        for (i, r) in per_view.iter().enumerate() {
            assert!(*r < 1e-10, "view {i} RMS={r}");
        }

        // Pack/unpack the same state and recompute RMS.
        let opts = CalibrateOptions {
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 5,
            ..Default::default()
        };
        let layout = CalibLayout::new(&opts, pose_specs.len(), cx, cy);
        let packed = pack_state(&layout, &cam, &pose_specs);
        let (cam2, poses2) = unpack_state(&packed, &layout).unwrap();
        let (overall2, _) = rms_reprojection(&object, &views, &cam2, &poses2);
        assert!(overall2 < 1e-10, "roundtrip RMS={overall2}");
    }

    #[test]
    fn unpack_rejects_wrong_param_len() {
        let opts = CalibrateOptions::default();
        let layout = CalibLayout::new(&opts, 2, 320.0, 240.0);
        let bad = vec![0.0; layout.n_params() - 1];
        assert_eq!(unpack_state(&bad, &layout), Err(PnpError::SolverFailed));
    }

    #[test]
    fn pose_cv_rvec_tvec_roundtrip() {
        let rvec = [0.1, -0.2, 0.3];
        let tvec = NaVector3::new(0.05, -0.02, 0.8);
        let pose = rvec_tvec_to_pose_cv(&rvec, &tvec);
        let (r2, t2) = pose_cv_to_rvec_tvec(&pose);
        let m1 = rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let m2 = rodrigues::rvec_to_rotation_matrix(&r2).to_na();
        assert!((m1 - m2).norm() < 1e-10);
        assert!((t2 - tvec).norm() < 1e-12);
    }

    /// Build a planar 3×3 grid on Z=0 (meters), centered on origin.
    fn planar_grid_3x3(half: f64) -> Vec<Vector3> {
        let mut pts = Vec::with_capacity(9);
        for iy in 0..3 {
            for ix in 0..3 {
                let x = -half + ix as f64 * half;
                let y = half - iy as f64 * half;
                pts.push(Vector3::new(x, y, 0.0));
            }
        }
        pts
    }

    #[test]
    fn refine_pinhole_from_noisy_init() {
        // True pinhole camera (noise-free synthetic).
        let fx_true = 800.0;
        let fy_true = 820.0;
        let cx_true = 320.0;
        let cy_true = 240.0;
        let cam_true = Camera::pinhole(fx_true, fy_true, cx_true, cy_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let object = planar_grid_3x3(0.1);

        // Varied tilts / distances (OpenCV object-in-camera).
        let true_poses: [CvRvecTvec; 6] = [
            ([0.15, -0.10, 0.05], NaVector3::new(0.02, -0.01, 0.55)),
            ([-0.20, 0.18, -0.08], NaVector3::new(-0.03, 0.02, 0.62)),
            ([0.10, 0.25, 0.12], NaVector3::new(0.01, 0.0, 0.48)),
            ([0.30, -0.05, -0.15], NaVector3::new(-0.02, 0.03, 0.70)),
            ([-0.12, -0.22, 0.08], NaVector3::new(0.04, -0.02, 0.58)),
            ([0.05, 0.12, -0.20], NaVector3::new(0.0, 0.01, 0.52)),
        ];

        let mut views = Vec::with_capacity(true_poses.len());
        for (rvec, tvec) in &true_poses {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut image_points = Vec::with_capacity(object.len());
            for p in &object {
                let pc = r * p.to_na() + tvec;
                let uv = cam_true
                    .project(Vector3::from_na(&pc))
                    .expect("in front of camera");
                image_points.push(uv);
            }
            views.push(CalibrationView { image_points });
        }

        // K off by ~5%; poses slightly perturbed.
        let cam0 =
            Camera::pinhole(fx_true * 1.05, fy_true * 0.95, cx_true + 5.0, cy_true - 4.0).unwrap();
        let poses0: Vec<CvRvecTvec> = true_poses
            .iter()
            .map(|(r, t)| {
                (
                    [r[0] + 0.02, r[1] - 0.015, r[2] + 0.01],
                    NaVector3::new(t.x + 0.005, t.y - 0.004, t.z * 1.03),
                )
            })
            .collect();

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: false,
            fix_principal_point: false,
            dist_len: 0,
            max_iterations: 100,
            function_tolerance: 1e-12,
            rms_success_threshold: None,
        };

        let (cam_est, poses_gl, rms, per_view) = refine_calibration_lm(
            &object,
            &views,
            image_width,
            image_height,
            &opts,
            cam0,
            poses0,
        )
        .expect("LM refine should succeed");

        assert!(rms < 1e-2, "overall RMS={rms}");
        assert_eq!(per_view.len(), true_poses.len());
        for (i, r) in per_view.iter().enumerate() {
            assert!(*r < 1e-2, "view {i} RMS={r}");
        }

        let fx_rel = (cam_est.fx - fx_true).abs() / fx_true;
        let fy_rel = (cam_est.fy - fy_true).abs() / fy_true;
        assert!(
            fx_rel < 1e-3,
            "fx rel err={fx_rel} (est={}, true={fx_true})",
            cam_est.fx
        );
        assert!(
            fy_rel < 1e-3,
            "fy rel err={fy_rel} (est={}, true={fy_true})",
            cam_est.fy
        );
        assert!(
            (cam_est.cx - cx_true).abs() < 0.5,
            "cx err={}",
            (cam_est.cx - cx_true).abs()
        );
        assert!(
            (cam_est.cy - cy_true).abs() < 0.5,
            "cy err={}",
            (cam_est.cy - cy_true).abs()
        );
        assert!(cam_est.dist.is_empty());
        assert_eq!(poses_gl.len(), true_poses.len());
    }

    #[test]
    fn calibrate_camera_recovers_pinhole_synthetic() {
        // N=10 views, varied tilt, 3×3 grid on Z=0, noise-free pinhole.
        let fx_true = 800.0;
        let fy_true = 800.0; // fix_aspect_ratio=true
        let cx_true = 320.0;
        let cy_true = 240.0;
        let cam_true = Camera::pinhole(fx_true, fy_true, cx_true, cy_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let object = planar_grid_3x3(0.1);

        let true_poses: [CvRvecTvec; 10] = [
            ([0.15, -0.10, 0.05], NaVector3::new(0.02, -0.01, 0.55)),
            ([-0.20, 0.18, -0.08], NaVector3::new(-0.03, 0.02, 0.62)),
            ([0.10, 0.25, 0.12], NaVector3::new(0.01, 0.0, 0.48)),
            ([0.30, -0.05, -0.15], NaVector3::new(-0.02, 0.03, 0.70)),
            ([-0.12, -0.22, 0.08], NaVector3::new(0.04, -0.02, 0.58)),
            ([0.05, 0.12, -0.20], NaVector3::new(0.0, 0.01, 0.52)),
            ([0.28, 0.15, 0.10], NaVector3::new(-0.01, 0.02, 0.60)),
            ([-0.25, -0.15, -0.05], NaVector3::new(0.03, -0.01, 0.65)),
            ([0.18, -0.28, 0.15], NaVector3::new(-0.02, 0.0, 0.50)),
            ([-0.08, 0.30, -0.12], NaVector3::new(0.01, 0.03, 0.57)),
        ];

        let mut views = Vec::with_capacity(true_poses.len());
        for (rvec, tvec) in &true_poses {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut image_points = Vec::with_capacity(object.len());
            for p in &object {
                let pc = r * p.to_na() + tvec;
                let uv = cam_true
                    .project(Vector3::from_na(&pc))
                    .expect("in front of camera");
                image_points.push(uv);
            }
            views.push(CalibrationView { image_points });
        }

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 0,
            max_iterations: 100,
            function_tolerance: 1e-12,
            rms_success_threshold: None,
        };

        let result = calibrate_camera(&object, &views, image_width, image_height, &opts)
            .expect("calibrate_camera should succeed");

        assert_eq!(result.views_used, true_poses.len());
        assert_eq!(result.object_poses.len(), true_poses.len());
        assert_eq!(result.per_view_rms.len(), true_poses.len());
        assert!(
            result.rms_reprojection_error < 1e-2,
            "RMS={}",
            result.rms_reprojection_error
        );
        for (i, r) in result.per_view_rms.iter().enumerate() {
            assert!(*r < 1e-2, "view {i} RMS={r}");
        }

        let fx_rel = (result.camera.fx - fx_true).abs() / fx_true;
        let fy_rel = (result.camera.fy - fy_true).abs() / fy_true;
        assert!(
            fx_rel < 1e-3,
            "fx rel err={fx_rel} (est={}, true={fx_true})",
            result.camera.fx
        );
        assert!(
            fy_rel < 1e-3,
            "fy rel err={fy_rel} (est={}, true={fy_true})",
            result.camera.fy
        );
        assert!(
            (result.camera.cx - cx_true).abs() < 0.5,
            "cx err={}",
            (result.camera.cx - cx_true).abs()
        );
        assert!(
            (result.camera.cy - cy_true).abs() < 0.5,
            "cy err={}",
            (result.camera.cy - cy_true).abs()
        );
        assert!(result.camera.dist.is_empty());
        // Fixed aspect: fy == fx after estimate.
        assert!((result.camera.fx - result.camera.fy).abs() < 1e-9);
    }

    #[test]
    fn calibrate_camera_recovers_distortion_len5_synthetic() {
        // Ground-truth Brown–Conrady (length 5); recover with dist_len=5.
        // Strong tilt diversity so radial distortion is identifiable.
        let fx_true = 800.0;
        let fy_true = 800.0;
        let cx_true = 320.0;
        let cy_true = 240.0;
        let dist_true = [0.12, -0.05, 0.001, -0.002, 0.01];
        let cam_true = Camera::new(fx_true, fy_true, cx_true, cy_true, &dist_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let object = planar_grid_3x3(0.1);

        // Strong yaw/pitch diversity, varied distance — needed for k1.
        let true_poses: [CvRvecTvec; 12] = [
            ([0.35, -0.28, 0.08], NaVector3::new(0.02, -0.01, 0.45)),
            ([-0.40, 0.32, -0.12], NaVector3::new(-0.03, 0.02, 0.55)),
            ([0.28, 0.38, 0.15], NaVector3::new(0.01, 0.0, 0.42)),
            ([0.42, -0.15, -0.20], NaVector3::new(-0.02, 0.03, 0.62)),
            ([-0.25, -0.35, 0.10], NaVector3::new(0.04, -0.02, 0.50)),
            ([0.12, 0.30, -0.28], NaVector3::new(0.0, 0.01, 0.48)),
            ([0.38, 0.22, 0.18], NaVector3::new(-0.01, 0.02, 0.58)),
            ([-0.35, -0.20, -0.08], NaVector3::new(0.03, -0.01, 0.65)),
            ([0.22, -0.40, 0.20], NaVector3::new(-0.02, 0.0, 0.44)),
            ([-0.18, 0.42, -0.15], NaVector3::new(0.01, 0.03, 0.52)),
            ([0.30, 0.10, 0.25], NaVector3::new(-0.015, 0.015, 0.60)),
            ([-0.32, 0.18, -0.22], NaVector3::new(0.02, -0.015, 0.47)),
        ];

        let mut views = Vec::with_capacity(true_poses.len());
        for (rvec, tvec) in &true_poses {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut image_points = Vec::with_capacity(object.len());
            for p in &object {
                let pc = r * p.to_na() + tvec;
                let uv = cam_true
                    .project(Vector3::from_na(&pc))
                    .expect("in front of camera");
                image_points.push(uv);
            }
            views.push(CalibrationView { image_points });
        }

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 5,
            max_iterations: 150,
            function_tolerance: 1e-12,
            rms_success_threshold: None,
        };

        let result = calibrate_camera(&object, &views, image_width, image_height, &opts)
            .expect("calibrate_camera with dist_len=5 should succeed");

        assert_eq!(result.views_used, true_poses.len());
        assert!(
            result.rms_reprojection_error < 1e-2,
            "RMS={}",
            result.rms_reprojection_error
        );
        for (i, r) in result.per_view_rms.iter().enumerate() {
            assert!(*r < 1e-2, "view {i} RMS={r}");
        }

        // Stored packing: dist_len=5 → [k1,k2,p1,p2,k3]
        assert_eq!(result.camera.dist.len(), 5);
        let k1_err = (result.camera.dist[0] - dist_true[0]).abs();
        assert!(
            k1_err < 0.02,
            "k1 err={k1_err} (est={}, true={})",
            result.camera.dist[0],
            dist_true[0]
        );

        // Intrinsics should still be reasonable with free distortion.
        let fx_rel = (result.camera.fx - fx_true).abs() / fx_true;
        assert!(
            fx_rel < 0.02,
            "fx rel err={fx_rel} (est={}, true={fx_true})",
            result.camera.fx
        );
        assert!((result.camera.fx - result.camera.fy).abs() < 1e-9);
    }

    #[test]
    fn refine_distortion_from_noisy_init() {
        // Direct LM path: free dist_len=5 from a perturbed seed.
        let fx_true = 800.0;
        let fy_true = 800.0;
        let cx_true = 320.0;
        let cy_true = 240.0;
        let dist_true = [0.10, -0.04, 0.0, 0.0, 0.008];
        let cam_true = Camera::new(fx_true, fy_true, cx_true, cy_true, &dist_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let object = planar_grid_3x3(0.1);

        let true_poses: [CvRvecTvec; 8] = [
            ([0.30, -0.25, 0.08], NaVector3::new(0.02, -0.01, 0.48)),
            ([-0.35, 0.28, -0.10], NaVector3::new(-0.03, 0.02, 0.58)),
            ([0.22, 0.35, 0.12], NaVector3::new(0.01, 0.0, 0.45)),
            ([0.38, -0.12, -0.18], NaVector3::new(-0.02, 0.03, 0.65)),
            ([-0.20, -0.32, 0.10], NaVector3::new(0.04, -0.02, 0.52)),
            ([0.15, 0.28, -0.25], NaVector3::new(0.0, 0.01, 0.50)),
            ([0.32, 0.18, 0.15], NaVector3::new(-0.01, 0.02, 0.60)),
            ([-0.28, -0.18, -0.08], NaVector3::new(0.03, -0.01, 0.55)),
        ];

        let mut views = Vec::with_capacity(true_poses.len());
        for (rvec, tvec) in &true_poses {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut image_points = Vec::with_capacity(object.len());
            for p in &object {
                let pc = r * p.to_na() + tvec;
                let uv = cam_true
                    .project(Vector3::from_na(&pc))
                    .expect("in front of camera");
                image_points.push(uv);
            }
            views.push(CalibrationView { image_points });
        }

        // Zero-dist seed (pipeline style) + mild pose perturbation.
        let cam0 = Camera::new(
            fx_true * 1.03,
            fy_true * 1.03,
            cx_true + 3.0,
            cy_true - 2.0,
            &[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let poses0: Vec<CvRvecTvec> = true_poses
            .iter()
            .map(|(r, t)| {
                (
                    [r[0] + 0.015, r[1] - 0.01, r[2] + 0.008],
                    NaVector3::new(t.x + 0.003, t.y - 0.002, t.z * 1.02),
                )
            })
            .collect();

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 5,
            max_iterations: 150,
            function_tolerance: 1e-12,
            rms_success_threshold: None,
        };

        let (cam_est, _, rms, _) = refine_calibration_lm(
            &object,
            &views,
            image_width,
            image_height,
            &opts,
            cam0,
            poses0,
        )
        .expect("LM refine with free dist should succeed");

        assert!(rms < 1e-2, "overall RMS={rms}");
        assert_eq!(cam_est.dist.len(), 5);
        let k1_err = (cam_est.dist[0] - dist_true[0]).abs();
        assert!(
            k1_err < 0.02,
            "k1 err={k1_err} (est={}, true={})",
            cam_est.dist[0],
            dist_true[0]
        );
    }

    #[test]
    fn calibrate_from_square_views_noise_free() {
        // Noise-free pinhole recovery via the square convenience API.
        let fx_true = 800.0;
        let fy_true = 800.0;
        let cx_true = 320.0;
        let cy_true = 240.0;
        let cam_true = Camera::pinhole(fx_true, fy_true, cx_true, cy_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let physical_size = 0.2;
        let object = square_object_points(physical_size).unwrap();

        let true_poses: [CvRvecTvec; 10] = [
            ([0.15, -0.10, 0.05], NaVector3::new(0.02, -0.01, 0.55)),
            ([-0.20, 0.18, -0.08], NaVector3::new(-0.03, 0.02, 0.62)),
            ([0.10, 0.25, 0.12], NaVector3::new(0.01, 0.0, 0.48)),
            ([0.30, -0.05, -0.15], NaVector3::new(-0.02, 0.03, 0.70)),
            ([-0.12, -0.22, 0.08], NaVector3::new(0.04, -0.02, 0.58)),
            ([0.05, 0.12, -0.20], NaVector3::new(0.0, 0.01, 0.52)),
            ([0.28, 0.15, 0.10], NaVector3::new(-0.01, 0.02, 0.60)),
            ([-0.25, -0.15, -0.05], NaVector3::new(0.03, -0.01, 0.65)),
            ([0.18, -0.28, 0.15], NaVector3::new(-0.02, 0.0, 0.50)),
            ([-0.08, 0.30, -0.12], NaVector3::new(0.01, 0.03, 0.57)),
        ];

        let mut corners_per_view: Vec<[Vector2; 4]> = Vec::with_capacity(true_poses.len());
        for (rvec, tvec) in &true_poses {
            let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
            let mut corners = [Vector2::new(0.0, 0.0); 4];
            for (i, p) in object.iter().enumerate() {
                let pc = r * p.to_na() + tvec;
                corners[i] = cam_true
                    .project(Vector3::from_na(&pc))
                    .expect("in front of camera");
            }
            corners_per_view.push(corners);
        }

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 0,
            max_iterations: 100,
            function_tolerance: 1e-12,
            rms_success_threshold: None,
        };

        let result = calibrate_from_square_views(
            &corners_per_view,
            physical_size,
            image_width,
            image_height,
            &opts,
        )
        .expect("calibrate_from_square_views should succeed");

        assert_eq!(result.views_used, true_poses.len());
        assert_eq!(result.object_poses.len(), true_poses.len());
        assert_eq!(result.per_view_rms.len(), true_poses.len());
        assert!(
            result.rms_reprojection_error < 1e-2,
            "RMS={}",
            result.rms_reprojection_error
        );
        for (i, r) in result.per_view_rms.iter().enumerate() {
            assert!(*r < 1e-2, "view {i} RMS={r}");
        }

        let fx_rel = (result.camera.fx - fx_true).abs() / fx_true;
        let fy_rel = (result.camera.fy - fy_true).abs() / fy_true;
        assert!(
            fx_rel < 1e-3,
            "fx rel err={fx_rel} (est={}, true={fx_true})",
            result.camera.fx
        );
        assert!(
            fy_rel < 1e-3,
            "fy rel err={fy_rel} (est={}, true={fy_true})",
            result.camera.fy
        );
        assert!(
            (result.camera.cx - cx_true).abs() < 0.5,
            "cx err={}",
            (result.camera.cx - cx_true).abs()
        );
        assert!(
            (result.camera.cy - cy_true).abs() < 0.5,
            "cy err={}",
            (result.camera.cy - cy_true).abs()
        );
        assert!(result.camera.dist.is_empty());
        assert!((result.camera.fx - result.camera.fy).abs() < 1e-9);
    }

    #[test]
    fn calibrate_from_square_views_rejects_bad_size() {
        assert!(
            calibrate_from_square_views(&[], 0.0, 640, 480, &CalibrateOptions::default()).is_err()
        );
        assert!(
            calibrate_from_square_views(&[], -0.1, 640, 480, &CalibrateOptions::default()).is_err()
        );
        assert!(
            calibrate_from_square_views(&[], f64::NAN, 640, 480, &CalibrateOptions::default())
                .is_err()
        );
        assert!(calibrate_from_square_views(
            &[],
            f64::INFINITY,
            640,
            480,
            &CalibrateOptions::default()
        )
        .is_err());
    }

    #[test]
    fn calibrate_rejects_identical_poses_cleanly() {
        // All views share one pose — degenerate for Zhang; must not panic.
        let fx_true = 800.0;
        let fy_true = 800.0;
        let cx_true = 320.0;
        let cy_true = 240.0;
        let cam_true = Camera::pinhole(fx_true, fy_true, cx_true, cy_true).unwrap();

        let image_width = 640u32;
        let image_height = 480u32;
        let physical_size = 0.2;
        let object = square_object_points(physical_size).unwrap();

        let rvec = [0.15, -0.10, 0.05];
        let tvec = NaVector3::new(0.02, -0.01, 0.55);
        let r = rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let mut corners = [Vector2::new(0.0, 0.0); 4];
        for (i, p) in object.iter().enumerate() {
            let pc = r * p.to_na() + tvec;
            corners[i] = cam_true
                .project(Vector3::from_na(&pc))
                .expect("in front of camera");
        }

        let corners_per_view = [corners; 5];

        let opts = CalibrateOptions {
            min_views: 3,
            fix_aspect_ratio: true,
            fix_principal_point: false,
            dist_len: 0,
            max_iterations: 50,
            function_tolerance: 1e-12,
            // Fail if the degenerate solution only achieves a high RMS.
            rms_success_threshold: Some(1e-2),
            ..Default::default()
        };

        // Either a clean PnpError or (if accepted) would need low RMS — with
        // identical poses Zhang is singular / inconsistent, so Err is expected.
        // In all cases we must not panic.
        match calibrate_from_square_views(
            &corners_per_view,
            physical_size,
            image_width,
            image_height,
            &opts,
        ) {
            Err(PnpError::SolverFailed) | Err(PnpError::InsufficientPoints) => {}
            Err(e) => {
                // Other PnpError variants are also acceptable (clean failure).
                let _ = e;
            }
            Ok(result) => {
                // If somehow accepted, RMS must still be under the threshold
                // (pipeline already enforced rms_success_threshold).
                assert!(
                    result.rms_reprojection_error < 1e-2,
                    "unexpected success with high RMS={}",
                    result.rms_reprojection_error
                );
            }
        }
    }
}
