//! Absolute orientation (Umeyama) for rigid 3D–3D registration.
//!
//! Finds a rigid transform `R, t` (scale fixed to 1) that maps source points
//! onto destination points in the least-squares sense:
//! `R * src_i + t ≈ dst_i`.

use crate::types::{
    rotation_matrix_to_quaternion, Matrix3x3, PnpError, Pose, Vector3,
};
use nalgebra::{Matrix3, Vector3 as NaVector3};

/// Minimum number of correspondences.
const MIN_POINTS: usize = 3;
/// Relative threshold for collinearity (cross-product length vs span²).
const COLLINEAR_EPS: f64 = 1e-12;
/// Minimum singular value ratio when forming the rotation (numerical floor).
const SVD_EPS: f64 = 1e-15;

/// Estimate the rigid transform mapping `src` points to `dst` points.
///
/// Uses the Umeyama / Kabsch procedure with **scale fixed to 1**:
/// centroids → cross-covariance → SVD → nearest proper rotation → translation.
///
/// # Errors
/// - [`PnpError::MismatchedCounts`] if slice lengths differ
/// - [`PnpError::InsufficientPoints`] if fewer than three pairs
/// - [`PnpError::SolverFailed`] if points are collinear (or otherwise degenerate)
///   or the SVD fails
pub fn absolute_orientation(src: &[Vector3], dst: &[Vector3]) -> Result<Pose, PnpError> {
    if src.len() != dst.len() {
        return Err(PnpError::MismatchedCounts);
    }
    let n = src.len();
    if n < MIN_POINTS {
        return Err(PnpError::InsufficientPoints);
    }

    if !src.iter().all(is_finite_vec3) || !dst.iter().all(is_finite_vec3) {
        return Err(PnpError::SolverFailed);
    }

    if points_are_collinear(src) || points_are_collinear(dst) {
        return Err(PnpError::SolverFailed);
    }

    let mu_src = centroid(src);
    let mu_dst = centroid(dst);

    // Cross-covariance H = Σ (dst_i - μ_dst)(src_i - μ_src)^T  (3×3)
    // so that y ≈ R x with R from SVD(H).
    let mut h = Matrix3::<f64>::zeros();
    for i in 0..n {
        let xs = NaVector3::new(src[i].x - mu_src.x, src[i].y - mu_src.y, src[i].z - mu_src.z);
        let yd = NaVector3::new(dst[i].x - mu_dst.x, dst[i].y - mu_dst.y, dst[i].z - mu_dst.z);
        h += yd * xs.transpose();
    }

    let svd = h.svd(true, true);
    let u = svd.u.ok_or(PnpError::SolverFailed)?;
    let vt = svd.v_t.ok_or(PnpError::SolverFailed)?;

    // Proper rotation: R = U * diag(1,1,det(U V^T)) * V^T
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        let mut d = Matrix3::<f64>::identity();
        d[(2, 2)] = -1.0;
        r = u * d * vt;
    }

    // Reject near-singular H (degenerate point set after centering)
    let singular = svd.singular_values;
    if singular[0] < SVD_EPS {
        return Err(PnpError::SolverFailed);
    }

    let t = mu_dst.to_na() - r * mu_src.to_na();
    if !t.x.is_finite() || !t.y.is_finite() || !t.z.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    let rot = rotation_matrix_to_quaternion(&Matrix3x3::from_na(&r)).normalize();
    Ok(Pose::new(Vector3::from_na(&t), rot))
}

fn centroid(pts: &[Vector3]) -> Vector3 {
    let n = pts.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for p in pts {
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    Vector3::new(sx / n, sy / n, sz / n)
}

fn is_finite_vec3(v: &Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// True if all points lie (approximately) on a single line.
fn points_are_collinear(pts: &[Vector3]) -> bool {
    if pts.len() < 3 {
        return true;
    }
    // Reference direction: first pair with meaningful separation.
    let mut dir = NaVector3::zeros();
    let mut span2 = 0.0;
    let p0 = pts[0].to_na();
    for p in pts.iter().skip(1) {
        let d = p.to_na() - p0;
        let len2 = d.norm_squared();
        if len2 > span2 {
            span2 = len2;
            dir = d;
        }
    }
    if span2 < COLLINEAR_EPS {
        // All points coincide.
        return true;
    }
    let scale = span2; // length²
    for p in pts.iter().skip(1) {
        let d = p.to_na() - p0;
        let cross = dir.cross(&d);
        if cross.norm_squared() > COLLINEAR_EPS * scale * scale {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose_tools::transform_point;
    use crate::types::Quaternion;
    use nalgebra::{Rotation3, Unit};

    const POS_TOL: f64 = 1e-9;
    const ROT_TOL: f64 = 1e-9;
    const MAP_TOL: f64 = 1e-9;

    fn apply_pose(pose: &Pose, pts: &[Vector3]) -> alloc::vec::Vec<Vector3> {
        pts.iter().map(|p| transform_point(pose, *p)).collect()
    }

    fn quat_angle(a: &Quaternion, b: &Quaternion) -> f64 {
        let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        2.0 * libm::acos(libm::fabs(dot).min(1.0))
    }

    fn poses_close(a: &Pose, b: &Pose) -> bool {
        let dp = Vector3::new(
            a.position.x - b.position.x,
            a.position.y - b.position.y,
            a.position.z - b.position.z,
        );
        dp.length() < POS_TOL && quat_angle(&a.rotation, &b.rotation) < ROT_TOL
    }

    #[test]
    fn recovers_known_rotation_and_translation() {
        let src = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.3, 0.2),
        ];
        let axis = Unit::new_normalize(NaVector3::new(0.2, 0.7, -0.4));
        let r = Rotation3::from_axis_angle(&axis, 0.85);
        let t = Vector3::new(0.3, -0.4, 1.2);
        let true_pose = Pose::new(
            t,
            rotation_matrix_to_quaternion(&Matrix3x3::from_na(r.matrix())).normalize(),
        );
        let dst = apply_pose(&true_pose, &src);

        let est = absolute_orientation(&src, &dst).expect("umeyama");
        assert!(
            poses_close(&est, &true_pose),
            "pos err {:?} rot err {}",
            (
                est.position.x - true_pose.position.x,
                est.position.y - true_pose.position.y,
                est.position.z - true_pose.position.z,
            ),
            quat_angle(&est.rotation, &true_pose.rotation)
        );

        for (s, d) in src.iter().zip(dst.iter()) {
            let mapped = transform_point(&est, *s);
            let err = Vector3::new(mapped.x - d.x, mapped.y - d.y, mapped.z - d.z).length();
            assert!(err < MAP_TOL, "map residual {err}");
        }
    }

    #[test]
    fn identity_on_identical_clouds() {
        let pts = [
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(-1.0, 0.5, 0.0),
            Vector3::new(0.0, -2.0, 1.0),
            Vector3::new(2.0, 1.0, -0.5),
        ];
        let pose = absolute_orientation(&pts, &pts).expect("identity");
        assert!(pose.position.length() < POS_TOL);
        assert!(quat_angle(&pose.rotation, &Quaternion::identity()) < ROT_TOL);
    }

    #[test]
    fn pure_translation() {
        let src = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let t = Vector3::new(5.0, -2.0, 3.0);
        let dst: alloc::vec::Vec<_> = src
            .iter()
            .map(|p| Vector3::new(p.x + t.x, p.y + t.y, p.z + t.z))
            .collect();
        let pose = absolute_orientation(&src, &dst).expect("translation");
        assert!((pose.position.x - t.x).abs() < POS_TOL);
        assert!((pose.position.y - t.y).abs() < POS_TOL);
        assert!((pose.position.z - t.z).abs() < POS_TOL);
        assert!(quat_angle(&pose.rotation, &Quaternion::identity()) < ROT_TOL);
    }

    #[test]
    fn rejects_too_few_points() {
        let src = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let dst = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)];
        assert_eq!(
            absolute_orientation(&src, &dst).unwrap_err(),
            PnpError::InsufficientPoints
        );
    }

    #[test]
    fn rejects_mismatched_counts() {
        let src = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let dst = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        assert_eq!(
            absolute_orientation(&src, &dst).unwrap_err(),
            PnpError::MismatchedCounts
        );
    }

    #[test]
    fn rejects_collinear_points() {
        let src = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
        let dst = src;
        assert_eq!(
            absolute_orientation(&src, &dst).unwrap_err(),
            PnpError::SolverFailed
        );
    }
}
