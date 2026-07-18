//! Multi-view monocular camera calibration: public types, validation, and
//! square planar object points.
//!
//! Full Zhang-style initialization and joint bundle adjustment land in later
//! tasks. This module currently exposes configuration / result types and
//! input checks shared by the calibration pipeline.

use crate::camera::Camera;
use crate::types::{PnpError, Pose, Vector2, Vector3};
use alloc::vec::Vec;

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
}
