//! SQPnP (Sequential Quadratic Programming) solver.
//!
//! Reference: Terzakis & Lourakis (2020)
//! "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem"

use crate::types::{PnpError, Vector2, Vector3};
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3 as NaVector3};

type Matrix9<T> = SMatrix<T, 9, 9>;

/// Solve PnP using the SQPnP algorithm.
///
/// Handles >= 3 points, works for both coplanar and non-coplanar configurations.
/// Returns (rotation_matrix, translation_vector) in OpenCV convention.
pub fn solve_sqpnp(
    object_points: &[Vector3],
    image_points: &[Vector2],
    camera_matrix: &Matrix3<f64>,
) -> Result<(Matrix3<f64>, NaVector3<f64>), PnpError> {
    let n = object_points.len();
    if n < 3 {
        return Err(PnpError::InsufficientPoints);
    }

    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    // Step 1: Normalize image points
    let normalized: alloc::vec::Vec<(f64, f64)> = image_points
        .iter()
        .map(|p| ((p.x - cx) / fx, (p.y - cy) / fy))
        .collect();

    // Step 2: Build the 9x9 Omega matrix
    let omega = build_omega(object_points, &normalized);

    // Step 3: Eigendecompose Omega
    let eigen = omega.symmetric_eigen();
    let eigenvalues = &eigen.eigenvalues;
    let eigenvectors = &eigen.eigenvectors;

    // Sort eigenvalues ascending
    let mut indices: [usize; 9] = core::array::from_fn(|i| i);
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    // Find rank of Omega (number of non-zero eigenvalues)
    let rank_threshold = 1e-10 * eigenvalues[indices[8]].abs().max(1e-15);
    let null_dim = indices
        .iter()
        .take_while(|&&i| eigenvalues[i].abs() < rank_threshold)
        .count();

    // Step 4: Generate initial rotation candidates from null space
    let mut candidates: alloc::vec::Vec<Matrix3<f64>> = alloc::vec::Vec::new();

    // Try each eigenvector with small eigenvalue as a rotation candidate
    for i in 0..3.min(9 - null_dim.max(0) + 1) {
        let ev = eigenvectors.column(indices[i]);
        let r_vec: SVector<f64, 9> = SVector::from_column_slice(ev.as_slice());

        // Reshape to 3x3 and project onto SO(3)
        let r_candidate = vec9_to_mat3(&r_vec);
        if let Some(r_nearest) = nearest_rotation_matrix(&r_candidate) {
            candidates.push(r_nearest);
        }
    }

    // Also add identity and some standard rotations as fallback
    candidates.push(Matrix3::identity());

    // Step 5: SQP refinement for each candidate
    let mut best_r = Matrix3::<f64>::identity();
    let mut best_t = NaVector3::zeros();
    let mut best_cost = f64::MAX;

    for r_init in &candidates {
        if let Some((r, t, cost)) =
            sqp_refine(&omega, r_init, object_points, &normalized, fx, fy, cx, cy)
        {
            if cost < best_cost {
                best_r = r;
                best_t = t;
                best_cost = cost;
            }
        }
    }

    if best_cost == f64::MAX {
        return Err(PnpError::SolverFailed);
    }

    Ok((best_r, best_t))
}

/// Build the 9x9 Omega matrix from point correspondences.
///
/// The projection equations for each point give a 2x12 linear system:
///   [Q_i | P_i] * [r; t] = 0
/// where Q_i is 2x9 (rotation coefficients) and P_i is 2x3 (translation coefficients).
///
/// Stacking all points: [Q | P] * [r; t] = 0
/// Eliminating t: Omega = Q^T * (I - P*(P^T*P)^{-1}*P^T) * Q
fn build_omega(object_points: &[Vector3], normalized_points: &[(f64, f64)]) -> Matrix9<f64> {
    let n = object_points.len();

    // Build Q (2n x 9) and P (2n x 3)
    let mut q_mat = DMatrix::zeros(2 * n, 9);
    let mut p_mat = DMatrix::zeros(2 * n, 3);

    for i in 0..n {
        let px = object_points[i].x;
        let py = object_points[i].y;
        let pz = object_points[i].z;
        let (nx, ny) = normalized_points[i];

        // Row-vectorized R: r = [R00, R01, R02, R10, R11, R12, R20, R21, R22]
        // Equation 1: r1^T*p + tx - nx*(r3^T*p + tz) = 0
        //   => [px, py, pz, 0, 0, 0, -nx*px, -nx*py, -nx*pz] * r + [1, 0, -nx] * t = 0
        q_mat[(2 * i, 0)] = px;
        q_mat[(2 * i, 1)] = py;
        q_mat[(2 * i, 2)] = pz;
        q_mat[(2 * i, 6)] = -nx * px;
        q_mat[(2 * i, 7)] = -nx * py;
        q_mat[(2 * i, 8)] = -nx * pz;

        p_mat[(2 * i, 0)] = 1.0;
        p_mat[(2 * i, 2)] = -nx;

        // Equation 2: r2^T*p + ty - ny*(r3^T*p + tz) = 0
        //   => [0, 0, 0, px, py, pz, -ny*px, -ny*py, -ny*pz] * r + [0, 1, -ny] * t = 0
        q_mat[(2 * i + 1, 3)] = px;
        q_mat[(2 * i + 1, 4)] = py;
        q_mat[(2 * i + 1, 5)] = pz;
        q_mat[(2 * i + 1, 6)] = -ny * px;
        q_mat[(2 * i + 1, 7)] = -ny * py;
        q_mat[(2 * i + 1, 8)] = -ny * pz;

        p_mat[(2 * i + 1, 1)] = 1.0;
        p_mat[(2 * i + 1, 2)] = -ny;
    }

    // Compute projection: P_perp = I - P*(P^T*P)^{-1}*P^T
    let ptp = p_mat.transpose() * &p_mat; // 3x3
    let ptp_inv = match ptp.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            // Fallback: use SVD pseudo-inverse
            let svd = ptp.svd(true, true);
            svd.pseudo_inverse(1e-10)
                .unwrap_or_else(|_| DMatrix::identity(3, 3))
        }
    };

    // P_perp * Q = Q - P*(P^T*P)^{-1}*P^T*Q
    let ptq = p_mat.transpose() * &q_mat; // 3 x 9
    let p_pinv_ptq = &ptp_inv * ptq; // 3 x 9
    let p_proj_q = &p_mat * p_pinv_ptq; // 2n x 9
    let perp_q = &q_mat - p_proj_q; // 2n x 9

    // Omega = Q^T * P_perp * Q = Q^T * perp_Q
    let omega_dyn = q_mat.transpose() * perp_q; // 9 x 9

    // Convert to static matrix
    let mut omega = Matrix9::<f64>::zeros();
    for i in 0..9 {
        for j in 0..9 {
            omega[(i, j)] = omega_dyn[(i, j)];
        }
    }

    omega
}

/// Project a 3x3 matrix onto SO(3) using SVD.
fn nearest_rotation_matrix(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;

    let det = (u * vt).determinant();
    if det < 0.0 {
        let mut d = Matrix3::<f64>::identity();
        d[(2, 2)] = -1.0;
        Some(u * d * vt)
    } else {
        Some(u * vt)
    }
}

/// Convert a 9-vector to a 3x3 matrix (row-major vectorization).
fn vec9_to_mat3(v: &SVector<f64, 9>) -> Matrix3<f64> {
    Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
}

/// Convert a 3x3 matrix to a 9-vector (row-major vectorization).
fn mat3_to_vec9(m: &Matrix3<f64>) -> SVector<f64, 9> {
    SVector::<f64, 9>::from_row_slice(&[
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 0)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 0)],
        m[(2, 1)],
        m[(2, 2)],
    ])
}

/// Recover translation given rotation matrix R, object points, and normalized image points.
///
/// Solves the overdetermined system: for each point i,
///   nx_i * (r3^T * p_i + tz) = r1^T * p_i + tx
///   ny_i * (r3^T * p_i + tz) = r2^T * p_i + ty
///
/// Rearranged: [1, 0, -nx_i] * t = nx_i * r3^T*p_i - r1^T*p_i
fn recover_translation(
    r: &Matrix3<f64>,
    object_points: &[Vector3],
    normalized_points: &[(f64, f64)],
) -> Option<NaVector3<f64>> {
    let n = object_points.len();
    let mut a = DMatrix::zeros(2 * n, 3);
    let mut b = DVector::zeros(2 * n);

    let r1 = r.row(0).transpose();
    let r2 = r.row(1).transpose();
    let r3 = r.row(2).transpose();

    for i in 0..n {
        let p = NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z);
        let (nx, ny) = normalized_points[i];

        let r1p = r1.dot(&p);
        let r2p = r2.dot(&p);
        let r3p = r3.dot(&p);

        // Equation 1: tx - nx*tz = nx*r3p - r1p
        a[(2 * i, 0)] = 1.0;
        a[(2 * i, 1)] = 0.0;
        a[(2 * i, 2)] = -nx;
        b[2 * i] = nx * r3p - r1p;

        // Equation 2: ty - ny*tz = ny*r3p - r2p
        a[(2 * i + 1, 0)] = 0.0;
        a[(2 * i + 1, 1)] = 1.0;
        a[(2 * i + 1, 2)] = -ny;
        b[2 * i + 1] = ny * r3p - r2p;
    }

    let svd = a.svd(true, true);
    let solution = svd.solve(&b, 1e-10).ok()?;
    Some(NaVector3::new(solution[0], solution[1], solution[2]))
}

/// SQP refinement starting from an initial rotation estimate.
fn sqp_refine(
    omega: &Matrix9<f64>,
    r_init: &Matrix3<f64>,
    object_points: &[Vector3],
    normalized_points: &[(f64, f64)],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Option<(Matrix3<f64>, NaVector3<f64>, f64)> {
    let max_iter = 15;
    let convergence = 1e-10;

    let mut r = *r_init;

    for _iter in 0..max_iter {
        let r_vec = mat3_to_vec9(&r);

        // Compute the gradient: g = 2 * Omega * r
        let g = 2.0 * omega * r_vec;

        // Compute the 6 orthogonality constraints: R^T*R - I = 0
        // Using the Jacobian of constraints w.r.t. r (9-vector)
        let (h, jac_h) = rotation_constraints_and_jacobian(&r);

        // SQP step: solve the KKT system
        // [2*Omega  J^T] [delta_r] = [-g      ]
        // [J         0 ] [lambda ] = [-h       ]

        let j = &jac_h; // 6x9
        let jt = j.transpose(); // 9x6

        // Form the 15x15 KKT system
        let mut kkt = DMatrix::zeros(15, 15);
        let omega_scaled = 2.0 * omega;

        // Top-left: 2*Omega (9x9)
        for row in 0..9 {
            for col in 0..9 {
                kkt[(row, col)] = omega_scaled[(row, col)];
            }
        }

        // Top-right: J^T (9x6)
        for row in 0..9 {
            for col in 0..6 {
                kkt[(row, 9 + col)] = jt[(row, col)];
            }
        }

        // Bottom-left: J (6x9)
        for row in 0..6 {
            for col in 0..9 {
                kkt[(9 + row, col)] = j[(row, col)];
            }
        }

        // RHS
        let mut rhs = DVector::zeros(15);
        for i in 0..9 {
            rhs[i] = -g[i];
        }
        for i in 0..6 {
            rhs[9 + i] = -h[i];
        }

        // Solve KKT system
        let lu = kkt.lu();
        let solution = lu.solve(&rhs)?;

        // Extract delta_r
        let mut delta_r = SVector::<f64, 9>::zeros();
        for i in 0..9 {
            delta_r[i] = solution[i];
        }

        // Update r
        let new_r_vec = r_vec + delta_r;
        let new_r_mat = vec9_to_mat3(&new_r_vec);

        // Project back onto SO(3)
        let new_r = nearest_rotation_matrix(&new_r_mat)?;

        // Check convergence
        let diff = mat3_to_vec9(&new_r) - mat3_to_vec9(&r);
        if diff.norm() < convergence {
            r = new_r;
            break;
        }

        r = new_r;
    }

    // Recover translation
    let t = recover_translation(&r, object_points, normalized_points)?;

    // Check that points are in front of camera
    let z_positive = object_points.iter().all(|p| {
        let pw = NaVector3::new(p.x, p.y, p.z);
        let pc = r * pw + t;
        pc.z > 0.0
    });

    if !z_positive {
        // Try negating the rotation (the other solution)
        return None;
    }

    // Compute reprojection error
    let err = crate::epnp::reprojection_error(
        object_points,
        &normalized_points
            .iter()
            .map(|(nx, ny)| crate::types::Vector2::new(nx * fx + cx, ny * fy + cy))
            .collect::<alloc::vec::Vec<_>>(),
        &r,
        &t,
        fx,
        fy,
        cx,
        cy,
    );

    Some((r, t, err))
}

/// Compute the 6 rotation constraints (R^T*R - I vectorized) and their 6x9 Jacobian.
fn rotation_constraints_and_jacobian(r: &Matrix3<f64>) -> (SVector<f64, 6>, SMatrix<f64, 6, 9>) {
    // Constraints: R^T*R - I = 0 (symmetric, 6 unique entries)
    let rtr = r.transpose() * r;
    let mut h = SVector::<f64, 6>::zeros();

    // Upper triangle of R^T*R - I:
    // (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    h[0] = rtr[(0, 0)] - 1.0;
    h[1] = rtr[(0, 1)];
    h[2] = rtr[(0, 2)];
    h[3] = rtr[(1, 1)] - 1.0;
    h[4] = rtr[(1, 2)];
    h[5] = rtr[(2, 2)] - 1.0;

    // Jacobian: dh/dr where r = [R00, R01, R02, R10, R11, R12, R20, R21, R22]
    // h[0] = R00^2 + R10^2 + R20^2 - 1  (column 0 dot column 0)
    // h[1] = R00*R01 + R10*R11 + R20*R21  (col 0 dot col 1)
    // h[2] = R00*R02 + R10*R12 + R20*R22  (col 0 dot col 2)
    // h[3] = R01^2 + R11^2 + R21^2 - 1   (col 1 dot col 1)
    // h[4] = R01*R02 + R11*R12 + R21*R22  (col 1 dot col 2)
    // h[5] = R02^2 + R12^2 + R22^2 - 1   (col 2 dot col 2)

    // Wait - our r vectorization is row-major: r = [R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2), R(2,0), R(2,1), R(2,2)]
    // R^T*R constraint involves column operations.
    // Column j of R = [R(0,j), R(1,j), R(2,j)]
    // In our vec: R(0,j) = r[j], R(1,j) = r[3+j], R(2,j) = r[6+j]

    let mut jac = SMatrix::<f64, 6, 9>::zeros();

    // h[0] = r[0]^2 + r[3]^2 + r[6]^2 - 1
    jac[(0, 0)] = 2.0 * r[(0, 0)];
    jac[(0, 3)] = 2.0 * r[(1, 0)];
    jac[(0, 6)] = 2.0 * r[(2, 0)];

    // h[1] = r[0]*r[1] + r[3]*r[4] + r[6]*r[7]
    jac[(1, 0)] = r[(0, 1)];
    jac[(1, 1)] = r[(0, 0)];
    jac[(1, 3)] = r[(1, 1)];
    jac[(1, 4)] = r[(1, 0)];
    jac[(1, 6)] = r[(2, 1)];
    jac[(1, 7)] = r[(2, 0)];

    // h[2] = r[0]*r[2] + r[3]*r[5] + r[6]*r[8]
    jac[(2, 0)] = r[(0, 2)];
    jac[(2, 2)] = r[(0, 0)];
    jac[(2, 3)] = r[(1, 2)];
    jac[(2, 5)] = r[(1, 0)];
    jac[(2, 6)] = r[(2, 2)];
    jac[(2, 8)] = r[(2, 0)];

    // h[3] = r[1]^2 + r[4]^2 + r[7]^2 - 1
    jac[(3, 1)] = 2.0 * r[(0, 1)];
    jac[(3, 4)] = 2.0 * r[(1, 1)];
    jac[(3, 7)] = 2.0 * r[(2, 1)];

    // h[4] = r[1]*r[2] + r[4]*r[5] + r[7]*r[8]
    jac[(4, 1)] = r[(0, 2)];
    jac[(4, 2)] = r[(0, 1)];
    jac[(4, 4)] = r[(1, 2)];
    jac[(4, 5)] = r[(1, 1)];
    jac[(4, 7)] = r[(2, 2)];
    jac[(4, 8)] = r[(2, 1)];

    // h[5] = r[2]^2 + r[5]^2 + r[8]^2 - 1
    jac[(5, 2)] = 2.0 * r[(0, 2)];
    jac[(5, 5)] = 2.0 * r[(1, 2)];
    jac[(5, 8)] = 2.0 * r[(2, 2)];

    (h, jac)
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
    fn test_omega_symmetric() {
        let pts = coplanar_landmarks();
        let norm: Vec<_> = pts.iter().map(|_| (0.1, 0.2)).collect();
        let omega = build_omega(&pts, &norm);

        for i in 0..9 {
            for j in 0..9 {
                assert!(
                    (omega[(i, j)] - omega[(j, i)]).abs() < 1e-12,
                    "Omega not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_omega_positive_semidefinite() {
        let pts = coplanar_landmarks();
        let norm: Vec<_> = vec![(0.1, 0.2), (-0.1, 0.3), (0.2, -0.1), (-0.2, -0.2)];
        let omega = build_omega(&pts, &norm);
        let eigen = omega.symmetric_eigen();
        for ev in eigen.eigenvalues.iter() {
            assert!(*ev > -1e-10, "Omega has negative eigenvalue: {}", ev);
        }
    }

    #[test]
    fn test_omega_dimensions() {
        let pts = coplanar_landmarks();
        let norm: Vec<_> = vec![(0.0, 0.0); 4];
        let omega = build_omega(&pts, &norm);
        assert_eq!(omega.nrows(), 9);
        assert_eq!(omega.ncols(), 9);
    }

    #[test]
    fn test_nearest_rotation() {
        let r = Matrix3::new(1.01, 0.02, -0.01, -0.02, 0.99, 0.03, 0.01, -0.03, 1.01);
        let nr = nearest_rotation_matrix(&r).unwrap();

        // Check det = 1
        assert!((nr.determinant() - 1.0).abs() < 1e-10, "det should be 1");

        // Check R^T*R = I
        let rtr = nr.transpose() * nr;
        let diff = (rtr - Matrix3::identity()).norm();
        assert!(diff < 1e-10, "R^T*R should be I, diff = {}", diff);
    }

    #[test]
    fn test_sqpnp_known_pose_noncoplanar() {
        let pts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];

        let rvec = [0.1, -0.2, 0.15];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.5, 0.3, 3.0);
        let img = project_points(&pts, &r_true, &t_true);
        let cam = camera_matrix();

        let (r, t) = solve_sqpnp(&pts, &img, &cam).unwrap();

        let pos_err = (t - t_true).norm();
        let rot_err = rotation_error(&r, &r_true);
        assert!(pos_err < 0.1, "position error: {}", pos_err);
        assert!(rot_err < 0.1, "rotation error: {} rad", rot_err);
    }

    #[test]
    fn test_sqpnp_known_pose_coplanar() {
        let pts = coplanar_landmarks();
        let rvec = [0.0, 0.3, 0.0];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 1.5);
        let img = project_points(&pts, &r_true, &t_true);
        let cam = camera_matrix();

        let result = solve_sqpnp(&pts, &img, &cam);
        assert!(
            result.is_ok(),
            "SQPnP should succeed for coplanar: {:?}",
            result.err()
        );

        let (r, t) = result.unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 5.0, "reprojection error: {}", reproj);
    }

    #[test]
    fn test_sqpnp_reference_set0() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(849.3577, 461.7641),
            Vector2::new(1070.642, 461.7641),
            Vector2::new(1096.898, 636.8014),
            Vector2::new(823.1021, 636.8014),
        ];
        let cam = camera_matrix();
        let (r, t) = solve_sqpnp(&pts, &img, &cam).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 0: {}", reproj);
    }

    #[test]
    fn test_sqpnp_reference_set1() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(1324.333, 208.0732),
            Vector2::new(1604.393, 129.3065),
            Vector2::new(1604.393, 403.1022),
            Vector2::new(1324.333, 429.3577),
        ];
        let cam = camera_matrix();
        let (r, t) = solve_sqpnp(&pts, &img, &cam).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 1: {}", reproj);
    }

    #[test]
    fn test_sqpnp_reference_set2() {
        let pts = coplanar_landmarks();
        let img = vec![
            Vector2::new(379.0064, 743.9628),
            Vector2::new(552.0745, 570.8946),
            Vector2::new(725.1426, 743.9628),
            Vector2::new(552.0745, 917.0309),
        ];
        let cam = camera_matrix();
        let (r, t) = solve_sqpnp(&pts, &img, &cam).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(reproj < 1.0, "reprojection error set 2: {}", reproj);
    }

    #[test]
    fn test_sqpnp_minimum_3_points() {
        // 3 non-collinear points
        let pts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let rvec = [0.05, -0.1, 0.05];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.5, 0.5, 4.0);
        let img = project_points(&pts, &r_true, &t_true);
        let cam = camera_matrix();

        let result = solve_sqpnp(&pts, &img, &cam);
        assert!(result.is_ok(), "SQPnP should work with 3 points");
    }

    #[test]
    fn test_sqpnp_many_points() {
        let mut pts = alloc::vec::Vec::new();
        let seed: u64 = 12345;
        let mut rng = seed;
        for _ in 0..20 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let z = ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            pts.push(Vector3::new(x, y, z));
        }

        let rvec = [0.2, -0.3, 0.1];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 5.0);
        let img = project_points(&pts, &r_true, &t_true);
        let cam = camera_matrix();

        let (r, t) = solve_sqpnp(&pts, &img, &cam).unwrap();
        let reproj = crate::epnp::reprojection_error(&pts, &img, &r, &t, FX, FY, CX, CY);
        assert!(
            reproj < 1.0,
            "reprojection error for many points: {}",
            reproj
        );
    }
}
