//! Midpoint triangulation for calibrated stereo pairs.
//!
//! Reconstructs a 3D point in the **left camera OpenCV frame** as the midpoint
//! of the common perpendicular between the left and right unprojected rays.

use crate::pose_tools::transform_point;
use crate::stereo::StereoRig;
use crate::types::{PnpError, Pose, Vector2, Vector3};

/// Minimum length of `D_l × D_r` before rays are treated as parallel.
const PARALLEL_EPS: f64 = 1e-12;
/// Minimum direction length for normalization.
const DIR_EPS: f64 = 1e-15;

/// Triangulate a point from a stereo correspondence via the skew-line midpoint.
///
/// # Frames
/// - Input pixels are OpenCV image coordinates (distorted OK; undistorted via camera).
/// - Output is in the **left camera OpenCV** frame (+Z forward).
///
/// # Errors
/// Returns [`PnpError::SolverFailed`] if rays are nearly parallel, depths are
/// non-positive, or the result is non-finite.
pub fn triangulate_midpoint(
    rig: &StereoRig,
    left_px: Vector2,
    right_px: Vector2,
) -> Result<Vector3, PnpError> {
    let ray_l = rig.left.unproject_opencv_ray(left_px);
    let ray_r = rig.right.unproject_opencv_ray(right_px);

    let o_l = Vector3::new(0.0, 0.0, 0.0);
    let d_l = normalize(ray_l.direction).ok_or(PnpError::SolverFailed)?;

    // Right ray expressed in left frame:
    // origin = right camera origin in left = right_from_left.position
    // direction = R * d_right (rotate free vector only)
    let o_r = rig.right_from_left.position;
    let rot_only = Pose::new(Vector3::new(0.0, 0.0, 0.0), rig.right_from_left.rotation);
    let d_r = normalize(transform_point(&rot_only, ray_r.direction)).ok_or(PnpError::SolverFailed)?;

    skew_line_midpoint(o_l, d_l, o_r, d_r)
}

/// Midpoint of the common perpendicular between two skew rays.
///
/// Rays are `O + s * D` with unit `D`. Rejects nearly parallel directions and
/// non-positive parameters `s`, `t` (behind either camera).
fn skew_line_midpoint(
    o1: Vector3,
    d1: Vector3,
    o2: Vector3,
    d2: Vector3,
) -> Result<Vector3, PnpError> {
    let cross_d = cross(d1, d2);
    let cross_len = length(cross_d);
    if !cross_len.is_finite() || cross_len < PARALLEL_EPS {
        return Err(PnpError::SolverFailed);
    }

    // Standard closest-points parameters for skew lines (Hartley/Zisserman / wiki).
    // w0 = O1 - O2
    let w0 = sub(o1, o2);
    let a = dot(d1, d1); // ~1 for unit dirs
    let b = dot(d1, d2);
    let c = dot(d2, d2); // ~1
    let d = dot(d1, w0);
    let e = dot(d2, w0);
    let denom = a * c - b * b;
    if !denom.is_finite() || denom.abs() < PARALLEL_EPS {
        return Err(PnpError::SolverFailed);
    }

    let s = (b * e - c * d) / denom;
    let t = (a * e - b * d) / denom;

    // Positive depth along both camera rays
    if !s.is_finite() || !t.is_finite() || s <= 0.0 || t <= 0.0 {
        return Err(PnpError::SolverFailed);
    }

    let p1 = add(o1, scale(d1, s));
    let p2 = add(o2, scale(d2, t));
    let mid = scale(add(p1, p2), 0.5);

    if !is_finite_vec3(&mid) {
        return Err(PnpError::SolverFailed);
    }
    Ok(mid)
}

fn normalize(v: Vector3) -> Option<Vector3> {
    let len = length(v);
    if !len.is_finite() || len < DIR_EPS {
        None
    } else {
        Some(scale(v, 1.0 / len))
    }
}

fn add(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn scale(v: Vector3, s: f64) -> Vector3 {
    Vector3::new(v.x * s, v.y * s, v.z * s)
}

fn dot(a: Vector3, b: Vector3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn length(v: Vector3) -> f64 {
    libm::sqrt(dot(v, v))
}

fn is_finite_vec3(v: &Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Quaternion;
    use crate::Camera;

    #[test]
    fn triangulate_known_point_in_front() {
        // Left at origin; right translated +0.1 m on X; identical K.
        // Point at (0, 0, 2) in left frame projects to both principal-ish rays with disparity.
        let left = Camera::pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let rig = StereoRig::new(
            left,
            right,
            Pose::new(Vector3::new(0.1, 0.0, 0.0), Quaternion::identity()),
        )
        .unwrap();

        let _p = Vector3::new(0.0, 0.0, 2.0);
        // Project with pinhole OpenCV: u = fx*X/Z+cx, v = fy*Y/Z+cy
        let ul = Vector2::new(320.0, 240.0); // principal point for (0,0,2)
        // In right camera: X_r = X_l - 0.1 => (-0.1, 0, 2)
        let ur = Vector2::new(500.0 * (-0.1) / 2.0 + 320.0, 240.0);

        let est = triangulate_midpoint(&rig, ul, ur).unwrap();
        assert!((est.x - 0.0).abs() < 1e-6);
        assert!((est.y - 0.0).abs() < 1e-6);
        assert!((est.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn triangulate_rejects_parallel_rays() {
        // Parallel cameras (identity R): principal points share direction (0,0,1)
        // after transform → |D_l × D_r| = 0 → SolverFailed.
        let left = Camera::pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let rig = StereoRig::new(
            left,
            right,
            Pose::new(Vector3::new(0.1, 0.0, 0.0), Quaternion::identity()),
        )
        .unwrap();

        let ul = Vector2::new(320.0, 240.0);
        let ur = Vector2::new(320.0, 240.0);
        let err = triangulate_midpoint(&rig, ul, ur).unwrap_err();
        assert_eq!(err, PnpError::SolverFailed);
    }
}
