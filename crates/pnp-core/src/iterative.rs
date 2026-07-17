//! Iterative (Levenberg-Marquardt) PnP solver.
//!
//! Minimizes reprojection error starting from an initial estimate (typically EPnP).

use crate::rodrigues;
use crate::types::{PnpError, Vector2, Vector3};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3 as NaVector3};

/// Solve PnP using iterative Levenberg-Marquardt optimization.
///
/// If no initial estimate is provided, EPnP is used to compute one.
/// Returns (rotation_matrix, translation_vector) in OpenCV convention.
pub fn solve_iterative(
    object_points: &[Vector3],
    image_points: &[Vector2],
    camera_matrix: &Matrix3<f64>,
    initial_rvec: Option<&[f64; 3]>,
    initial_tvec: Option<&NaVector3<f64>>,
) -> Result<(Matrix3<f64>, NaVector3<f64>), PnpError> {
    let n = object_points.len();
    if n < 4 {
        return Err(PnpError::InsufficientPoints);
    }

    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    // Get initial estimate
    let (mut rvec, mut tvec) = match (initial_rvec, initial_tvec) {
        (Some(rv), Some(tv)) => (*rv, *tv),
        _ => {
            // Use EPnP for initial estimate
            let (r, t) = crate::epnp::solve_epnp(object_points, image_points, camera_matrix)?;
            let rm = crate::types::Matrix3x3::from_na(&r);
            let rv = rodrigues::rotation_matrix_to_rvec(&rm);
            (rv, t)
        }
    };

    // LM parameters
    let max_iterations = 100;
    let mut lambda = 1e-3;
    let lambda_factor = 10.0;
    let convergence_threshold = 1e-8;

    let mut prev_cost = compute_cost(object_points, image_points, &rvec, &tvec, fx, fy, cx, cy);

    for _iter in 0..max_iterations {
        // Compute residuals and Jacobian
        let (residuals, jacobian) = compute_residuals_and_jacobian(
            object_points,
            image_points,
            &rvec,
            &tvec,
            fx,
            fy,
            cx,
            cy,
        );

        // J^T * J and J^T * r
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // Solve (J^T*J + lambda*diag(J^T*J)) * delta = -J^T*r
        let mut a = jtj.clone();
        for i in 0..6 {
            a[(i, i)] += lambda * jtj[(i, i)].max(1e-10);
        }

        let neg_jtr = -&jtr;
        let lu = a.lu();
        let delta = match lu.solve(&neg_jtr) {
            Some(d) => d,
            None => break,
        };

        // Check convergence
        let delta_norm = delta.norm();
        if delta_norm < convergence_threshold {
            break;
        }

        // Try the update
        let new_rvec = [rvec[0] + delta[0], rvec[1] + delta[1], rvec[2] + delta[2]];
        let new_tvec = NaVector3::new(tvec.x + delta[3], tvec.y + delta[4], tvec.z + delta[5]);

        let new_cost = compute_cost(
            object_points,
            image_points,
            &new_rvec,
            &new_tvec,
            fx,
            fy,
            cx,
            cy,
        );

        if new_cost < prev_cost {
            // Accept step, decrease lambda
            rvec = new_rvec;
            tvec = new_tvec;
            lambda /= lambda_factor;
            lambda = lambda.max(1e-10);

            if (prev_cost - new_cost) / prev_cost.max(1e-15) < convergence_threshold {
                break;
            }
            prev_cost = new_cost;
        } else {
            // Reject step, increase lambda
            lambda *= lambda_factor;
            if lambda > 1e16 {
                break;
            }
        }
    }

    // Convert rvec to rotation matrix
    let r = rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
    Ok((r, tvec))
}

/// Compute the total reprojection cost (sum of squared residuals).
fn compute_cost(
    object_points: &[Vector3],
    image_points: &[Vector2],
    rvec: &[f64; 3],
    tvec: &NaVector3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> f64 {
    let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
    let n = object_points.len();
    let mut cost = 0.0;

    for i in 0..n {
        let pw = NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z);
        let pc = r * pw + tvec;

        if pc.z.abs() < 1e-15 {
            return f64::MAX;
        }

        let u_proj = fx * pc.x / pc.z + cx;
        let v_proj = fy * pc.y / pc.z + cy;

        let du = u_proj - image_points[i].x;
        let dv = v_proj - image_points[i].y;
        cost += du * du + dv * dv;
    }

    cost
}

/// Compute residuals (2n×1) and Jacobian (2n×6) w.r.t. [rvec; tvec].
fn compute_residuals_and_jacobian(
    object_points: &[Vector3],
    image_points: &[Vector2],
    rvec: &[f64; 3],
    tvec: &NaVector3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> (DVector<f64>, DMatrix<f64>) {
    let n = object_points.len();
    let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();

    let mut residuals = DVector::zeros(2 * n);
    let mut jacobian = DMatrix::zeros(2 * n, 6);

    // Compute dR/drvec using finite differences
    let eps = 1e-6;
    let mut dr_drvec = [Matrix3::<f64>::zeros(); 3];
    for k in 0..3 {
        let mut rv_plus = *rvec;
        let mut rv_minus = *rvec;
        rv_plus[k] += eps;
        rv_minus[k] -= eps;
        let r_plus = rodrigues::rvec_to_rotation_matrix(&rv_plus).to_na();
        let r_minus = rodrigues::rvec_to_rotation_matrix(&rv_minus).to_na();
        dr_drvec[k] = (r_plus - r_minus) / (2.0 * eps);
    }

    for i in 0..n {
        let pw = NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z);
        let pc = r * pw + tvec;

        if pc.z.abs() < 1e-15 {
            continue;
        }

        let inv_z = 1.0 / pc.z;
        let inv_z2 = inv_z * inv_z;

        let u_proj = fx * pc.x * inv_z + cx;
        let v_proj = fy * pc.y * inv_z + cy;

        residuals[2 * i] = u_proj - image_points[i].x;
        residuals[2 * i + 1] = v_proj - image_points[i].y;

        // Jacobian of projection w.r.t. pc = [x_c, y_c, z_c]
        // du/dxc = fx / zc
        // du/dyc = 0
        // du/dzc = -fx * xc / zc^2
        // dv/dxc = 0
        // dv/dyc = fy / zc
        // dv/dzc = -fy * yc / zc^2

        // d(pc)/d(rvec_k) = dR/drvec_k * pw
        for k in 0..3 {
            let dpc = dr_drvec[k] * pw;

            jacobian[(2 * i, k)] = fx * (dpc.x * inv_z - pc.x * dpc.z * inv_z2);
            jacobian[(2 * i + 1, k)] = fy * (dpc.y * inv_z - pc.y * dpc.z * inv_z2);
        }

        // d(pc)/d(tvec) = I
        // du/dtx = fx / zc
        jacobian[(2 * i, 3)] = fx * inv_z;
        // du/dty = 0
        jacobian[(2 * i, 4)] = 0.0;
        // du/dtz = -fx * xc / zc^2
        jacobian[(2 * i, 5)] = -fx * pc.x * inv_z2;
        // dv/dtx = 0
        jacobian[(2 * i + 1, 3)] = 0.0;
        // dv/dty = fy / zc
        jacobian[(2 * i + 1, 4)] = fy * inv_z;
        // dv/dtz = -fy * yc / zc^2
        jacobian[(2 * i + 1, 5)] = -fy * pc.y * inv_z2;
    }

    (residuals, jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Vector2, Vector3};

    const FX: f64 = 815.8511;
    const FY: f64 = 815.8511;
    const CX: f64 = 960.0;
    const CY: f64 = 540.0;

    fn camera_matrix() -> Matrix3<f64> {
        Matrix3::new(FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0)
    }

    fn project_points(pts: &[Vector3], r: &Matrix3<f64>, t: &NaVector3<f64>) -> Vec<Vector2> {
        pts.iter()
            .map(|p| {
                let pw = NaVector3::new(p.x, p.y, p.z);
                let pc = r * pw + t;
                Vector2::new(FX * pc.x / pc.z + CX, FY * pc.y / pc.z + CY)
            })
            .collect()
    }

    fn rotation_error(r1: &Matrix3<f64>, r2: &Matrix3<f64>) -> f64 {
        let r_diff = r1.transpose() * r2;
        let trace = r_diff[(0, 0)] + r_diff[(1, 1)] + r_diff[(2, 2)];
        let cos_angle = ((trace - 1.0) / 2.0).max(-1.0).min(1.0);
        libm::acos(cos_angle)
    }

    fn coplanar_landmarks() -> Vec<Vector3> {
        vec![
            Vector3::new(-0.15, -0.15, 0.0),
            Vector3::new(0.15, -0.15, 0.0),
            Vector3::new(0.15, 0.15, 0.0),
            Vector3::new(-0.15, 0.15, 0.0),
        ]
    }

    #[test]
    fn test_reprojection_zero_at_ground_truth() {
        let pts = coplanar_landmarks();
        let rvec = [-0.785398, -0.000001, 0.0];
        let r = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t = NaVector3::new(0.0, 0.0, 1.0);
        let img = project_points(&pts, &r, &t);
        let cost = compute_cost(&pts, &img, &rvec, &t, FX, FY, CX, CY);
        assert!(
            cost < 1e-10,
            "cost at ground truth should be ~0, got {}",
            cost
        );
    }

    #[test]
    fn test_reprojection_nonzero_at_perturbed() {
        let pts = coplanar_landmarks();
        let rvec_true = [-0.785398, 0.0, 0.0];
        let t_true = NaVector3::new(0.0, 0.0, 1.0);
        let r = crate::rodrigues::rvec_to_rotation_matrix(&rvec_true).to_na();
        let img = project_points(&pts, &r, &t_true);

        let rvec_perturbed = [-0.785398, 0.1, 0.0];
        let cost = compute_cost(&pts, &img, &rvec_perturbed, &t_true, FX, FY, CX, CY);
        assert!(
            cost > 1.0,
            "cost at perturbed pose should be > 0, got {}",
            cost
        );
    }

    #[test]
    fn test_jacobian_vs_finite_difference() {
        let pts = coplanar_landmarks();
        let rvec = [0.1, -0.2, 0.3];
        let tvec = NaVector3::new(0.0, 0.0, 1.5);
        let r = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let img = project_points(&pts, &r, &tvec);

        let (_, jacobian) =
            compute_residuals_and_jacobian(&pts, &img, &rvec, &tvec, FX, FY, CX, CY);

        // Numerical Jacobian
        let eps = 1e-6;
        let n = pts.len();
        let mut num_jac = DMatrix::zeros(2 * n, 6);

        for k in 0..6 {
            let mut params_plus = [rvec[0], rvec[1], rvec[2], tvec.x, tvec.y, tvec.z];
            let mut params_minus = params_plus;
            params_plus[k] += eps;
            params_minus[k] -= eps;

            let rv_p = [params_plus[0], params_plus[1], params_plus[2]];
            let tv_p = NaVector3::new(params_plus[3], params_plus[4], params_plus[5]);
            let rv_m = [params_minus[0], params_minus[1], params_minus[2]];
            let tv_m = NaVector3::new(params_minus[3], params_minus[4], params_minus[5]);

            let (res_p, _) =
                compute_residuals_and_jacobian(&pts, &img, &rv_p, &tv_p, FX, FY, CX, CY);
            let (res_m, _) =
                compute_residuals_and_jacobian(&pts, &img, &rv_m, &tv_m, FX, FY, CX, CY);

            for i in 0..(2 * n) {
                num_jac[(i, k)] = (res_p[i] - res_m[i]) / (2.0 * eps);
            }
        }

        // Compare
        for i in 0..(2 * n) {
            for k in 0..6 {
                let diff = (jacobian[(i, k)] - num_jac[(i, k)]).abs();
                let scale = jacobian[(i, k)].abs().max(1.0);
                assert!(
                    diff / scale < 1e-3,
                    "Jacobian mismatch at ({},{}): analytical={}, numerical={}, diff={}",
                    i,
                    k,
                    jacobian[(i, k)],
                    num_jac[(i, k)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_jacobian_shape() {
        let pts = coplanar_landmarks();
        let rvec = [0.0, 0.0, 0.0];
        let tvec = NaVector3::new(0.0, 0.0, 1.0);
        let r = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let img = project_points(&pts, &r, &tvec);

        let (res, jac) = compute_residuals_and_jacobian(&pts, &img, &rvec, &tvec, FX, FY, CX, CY);
        assert_eq!(res.len(), 8);
        assert_eq!(jac.nrows(), 8);
        assert_eq!(jac.ncols(), 6);
    }

    #[test]
    fn test_lm_converges_from_good_initial() {
        let pts = coplanar_landmarks();
        let rvec_true = [0.0, 0.3, 0.0];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec_true).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 1.5);
        let img = project_points(&pts, &r_true, &t_true);

        // Start with small perturbation
        let rvec_init = [0.01, 0.31, -0.01];
        let t_init = NaVector3::new(0.01, 0.01, 1.51);

        let cam = camera_matrix();
        let (r, t) = solve_iterative(&pts, &img, &cam, Some(&rvec_init), Some(&t_init)).unwrap();

        let pos_err = (t - t_true).norm();
        let rot_err = rotation_error(&r, &r_true);
        assert!(pos_err < 1e-3, "position error: {}", pos_err);
        assert!(rot_err < 1e-3, "rotation error: {} rad", rot_err);
    }

    #[test]
    fn test_lm_converges_from_epnp() {
        let pts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];
        let rvec_true = [0.1, -0.2, 0.15];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec_true).to_na();
        let t_true = NaVector3::new(0.5, 0.3, 3.0);
        let img = project_points(&pts, &r_true, &t_true);

        let cam = camera_matrix();
        // No initial estimate → uses EPnP
        let (r, t) = solve_iterative(&pts, &img, &cam, None, None).unwrap();

        let pos_err = (t - t_true).norm();
        let rot_err = rotation_error(&r, &r_true);
        assert!(pos_err < 1e-3, "position error: {}", pos_err);
        assert!(rot_err < 1e-3, "rotation error: {} rad", rot_err);
    }

    #[test]
    fn test_lm_reference_set0() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(849.3577, 461.7641),
            Vector2::new(1070.642, 461.7641),
            Vector2::new(1096.898, 636.8014),
            Vector2::new(823.1021, 636.8014),
        ];
        let cam = camera_matrix();

        let (r, t) = solve_iterative(&pts, &img, &cam, None, None).unwrap();

        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 0: {}", reproj);
    }

    #[test]
    fn test_lm_reference_set1() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(1324.333, 208.0732),
            Vector2::new(1604.393, 129.3065),
            Vector2::new(1604.393, 403.1022),
            Vector2::new(1324.333, 429.3577),
        ];
        let cam = camera_matrix();
        let (r, t) = solve_iterative(&pts, &img, &cam, None, None).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 1: {}", reproj);
    }

    #[test]
    fn test_lm_reference_set2() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(379.0064, 743.9628),
            Vector2::new(552.0745, 570.8946),
            Vector2::new(725.1426, 743.9628),
            Vector2::new(552.0745, 917.0309),
        ];
        let cam = camera_matrix();
        let (r, t) = solve_iterative(&pts, &img, &cam, None, None).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 2: {}", reproj);
    }

    #[test]
    fn test_lm_max_iterations() {
        // Bad initial estimate — should complete without panic
        let pts = coplanar_landmarks();
        let rvec_true = [0.0, 0.3, 0.0];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec_true).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 1.5);
        let img = project_points(&pts, &r_true, &t_true);

        let rvec_bad = [2.0, -1.0, 1.5];
        let t_bad = NaVector3::new(5.0, 5.0, 10.0);

        let cam = camera_matrix();
        let _ = solve_iterative(&pts, &img, &cam, Some(&rvec_bad), Some(&t_bad));
        // Just verify no panic
    }
}
