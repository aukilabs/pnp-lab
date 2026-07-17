//! EPnP (Efficient Perspective-n-Point) solver.
//!
//! Reference: Lepetit, Moreno-Noguer, Fua (2009)
//! "EPnP: An Accurate O(n) Solution to the PnP Problem"

use crate::types::{PnpError, Vector2, Vector3};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3 as NaVector3};

/// Solve PnP using the EPnP algorithm.
///
/// Returns (rotation_matrix, translation_vector) in OpenCV convention.
pub fn solve_epnp(
    object_points: &[Vector3],
    image_points: &[Vector2],
    camera_matrix: &Matrix3<f64>,
) -> Result<(Matrix3<f64>, NaVector3<f64>), PnpError> {
    let n = object_points.len();
    if n < 4 {
        return Err(PnpError::InsufficientPoints);
    }

    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    // Step 1: Choose control points and compute barycentric coordinates
    let (control_points, alphas, num_cp) = compute_control_points_and_barycentrics(object_points);
    let dim = num_cp * 3; // 9 for planar (3 control points), 12 for general (4)

    // Step 2: Build M matrix (2n x dim)
    let m_matrix = build_m_matrix(&alphas, image_points, fx, fy, cx, cy, num_cp);

    // Step 3: Compute M^T * M and its eigendecomposition
    let mtm = m_matrix.transpose() * &m_matrix;

    // Eigendecomposition
    let eigen = mtm.symmetric_eigen();
    let eigenvalues = &eigen.eigenvalues;
    let eigenvectors = &eigen.eigenvectors;

    // Sort eigenvectors by eigenvalue (ascending)
    let mut indices: alloc::vec::Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    // Extract the null space vectors
    let max_null = 4.min(dim);
    let mut null_vecs: alloc::vec::Vec<DVector<f64>> = alloc::vec::Vec::new();
    for i in 0..max_null {
        null_vecs.push(eigenvectors.column(indices[i]).clone_owned());
    }

    // Step 4: Try different numbers of null space vectors
    let mut best_r = Matrix3::<f64>::identity();
    let mut best_t = NaVector3::zeros();
    let mut best_error = f64::MAX;

    // Compute world-frame pairwise distances between control points
    let world_dists = pairwise_distances_n(&control_points[..num_cp]);

    for num_null in 1..=max_null.min(3) {
        if let Some((r, t, err)) = try_betas(
            &null_vecs[..num_null],
            &world_dists,
            num_cp,
            &control_points[..num_cp],
            &alphas,
            num_cp,
            object_points,
            image_points,
            fx,
            fy,
            cx,
            cy,
        ) {
            if err < best_error {
                best_r = r;
                best_t = t;
                best_error = err;
            }
        }
    }

    if best_error == f64::MAX {
        return Err(PnpError::SolverFailed);
    }

    Ok((best_r, best_t))
}

/// Compute control points and barycentric coordinates.
/// Detects coplanar configuration and uses 3 or 4 control points accordingly.
/// Returns (control_points, alphas, num_control_points).
fn compute_control_points_and_barycentrics(
    points: &[Vector3],
) -> (
    [NaVector3<f64>; 4],
    alloc::vec::Vec<alloc::vec::Vec<f64>>,
    usize,
) {
    let n = points.len();

    // Control point 0: centroid
    let mut centroid = NaVector3::zeros();
    for p in points {
        centroid += NaVector3::new(p.x, p.y, p.z);
    }
    centroid /= n as f64;

    // Compute covariance matrix of centered points
    let mut cov = Matrix3::<f64>::zeros();
    for p in points {
        let d = NaVector3::new(p.x - centroid.x, p.y - centroid.y, p.z - centroid.z);
        cov += d * d.transpose();
    }
    cov /= n as f64;

    let eigen = cov.symmetric_eigen();

    // Sort eigenvalues descending to detect planarity
    let mut ev_indices = [0usize, 1, 2];
    ev_indices.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap()
    });

    let sorted_eigenvalues: [f64; 3] = [
        eigen.eigenvalues[ev_indices[0]],
        eigen.eigenvalues[ev_indices[1]],
        eigen.eigenvalues[ev_indices[2]],
    ];

    // Detect planarity: smallest eigenvalue << other two
    let is_planar = sorted_eigenvalues[2].abs() < 1e-6 * sorted_eigenvalues[0].max(1e-15);

    let mut control_points = [NaVector3::zeros(); 4];
    control_points[0] = centroid;

    if is_planar {
        // Planar case: use 3 control points (centroid + 2 in-plane directions)
        let num_cp = 3;
        for i in 0..2 {
            control_points[i + 1] =
                centroid + eigen.eigenvectors.column(ev_indices[i]).into_owned();
        }

        // Barycentric coordinates: p = a0*c0 + a1*c1 + a2*c2, a0+a1+a2 = 1
        let c0 = control_points[0];
        let v1 = control_points[1] - c0;
        let v2 = control_points[2] - c0;

        // Build 2-component system (project onto plane)
        // For each point: p-c0 = a1*(c1-c0) + a2*(c2-c0)
        // Since points are coplanar, we can solve the overdetermined 3x2 system
        let mut basis = DMatrix::zeros(3, 2);
        for j in 0..3 {
            basis[(j, 0)] = v1[j];
        }
        for j in 0..3 {
            basis[(j, 1)] = v2[j];
        }
        let svd = basis.svd(true, true);

        let mut alphas = alloc::vec::Vec::with_capacity(n);
        for p in points {
            let pv = DVector::from_column_slice(&[p.x - c0.x, p.y - c0.y, p.z - c0.z]);
            let bary = svd.solve(&pv, 1e-12).unwrap_or(DVector::zeros(2));
            let a0 = 1.0 - bary[0] - bary[1];
            alphas.push(alloc::vec![a0, bary[0], bary[1]]);
        }

        (control_points, alphas, num_cp)
    } else {
        // General case: use 4 control points
        let num_cp = 4;
        for i in 0..3 {
            control_points[i + 1] =
                centroid + eigen.eigenvectors.column(ev_indices[i]).into_owned();
        }

        let c0 = control_points[0];
        let mut basis = Matrix3::<f64>::zeros();
        for j in 0..3 {
            let col = control_points[j + 1] - c0;
            basis.set_column(j, &col);
        }

        let svd = basis.svd(true, true);

        let mut alphas = alloc::vec::Vec::with_capacity(n);
        for p in points {
            let pv = NaVector3::new(p.x, p.y, p.z) - c0;
            let bary = svd.solve(&pv, 1e-12).unwrap_or(NaVector3::zeros());
            let a0 = 1.0 - bary[0] - bary[1] - bary[2];
            alphas.push(alloc::vec![a0, bary[0], bary[1], bary[2]]);
        }

        (control_points, alphas, num_cp)
    }
}

/// Build the 2n x (num_cp*3) M matrix from projection equations.
fn build_m_matrix(
    alphas: &[alloc::vec::Vec<f64>],
    image_points: &[Vector2],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    num_cp: usize,
) -> DMatrix<f64> {
    let n = alphas.len();
    let dim = num_cp * 3;
    let mut m = DMatrix::zeros(2 * n, dim);

    for i in 0..n {
        let u = image_points[i].x;
        let v = image_points[i].y;
        let a = &alphas[i];

        for j in 0..num_cp {
            let col = j * 3;
            m[(2 * i, col)] = a[j] * fx;
            m[(2 * i, col + 1)] = 0.0;
            m[(2 * i, col + 2)] = a[j] * (cx - u);

            m[(2 * i + 1, col)] = 0.0;
            m[(2 * i + 1, col + 1)] = a[j] * fy;
            m[(2 * i + 1, col + 2)] = a[j] * (cy - v);
        }
    }

    m
}

/// Extract control point positions in camera frame from a vector.
fn extract_camera_control_points_dyn(
    v: &DVector<f64>,
    num_cp: usize,
) -> alloc::vec::Vec<NaVector3<f64>> {
    (0..num_cp)
        .map(|i| NaVector3::new(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]))
        .collect()
}

/// Compute pairwise distances between control points.
fn pairwise_distances_n(pts: &[NaVector3<f64>]) -> alloc::vec::Vec<f64> {
    let n = pts.len();
    let mut dists = alloc::vec::Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push((pts[i] - pts[j]).norm_squared());
        }
    }
    dists
}

/// Try solving with `num_null` null space vectors.
fn try_betas(
    null_vecs: &[DVector<f64>],
    world_dists: &[f64],
    num_cp: usize,
    _control_points: &[NaVector3<f64>],
    alphas: &[alloc::vec::Vec<f64>],
    _num_cp_alphas: usize,
    object_points: &[Vector3],
    image_points: &[Vector2],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Option<(Matrix3<f64>, NaVector3<f64>, f64)> {
    let num_null = null_vecs.len();
    let num_dists = world_dists.len();

    // Extract control point positions from each null vector
    let v_pts: alloc::vec::Vec<alloc::vec::Vec<NaVector3<f64>>> = null_vecs
        .iter()
        .map(|v| extract_camera_control_points_dyn(v, num_cp))
        .collect();

    // Number of beta products: num_null*(num_null+1)/2
    let num_beta_products = num_null * (num_null + 1) / 2;

    // Build linearized system for beta products
    let mut l_mat = DMatrix::zeros(num_dists, num_beta_products);
    let mut rhs = DVector::zeros(num_dists);

    // Generate pair indices
    let mut pair_indices = alloc::vec::Vec::new();
    for i in 0..num_cp {
        for j in (i + 1)..num_cp {
            pair_indices.push((i, j));
        }
    }

    for (dist_idx, &(pi, pj)) in pair_indices.iter().enumerate() {
        // Compute difference vectors for each null vector
        let dv: alloc::vec::Vec<NaVector3<f64>> =
            (0..num_null).map(|k| v_pts[k][pi] - v_pts[k][pj]).collect();

        // Fill L matrix columns: beta_a * beta_b terms
        let mut col = 0;
        for a in 0..num_null {
            for b in a..num_null {
                let factor = if a == b { 1.0 } else { 2.0 };
                l_mat[(dist_idx, col)] = factor * dv[a].dot(&dv[b]);
                col += 1;
            }
        }

        rhs[dist_idx] = world_dists[dist_idx];
    }

    // Solve least squares for beta products
    let lls = l_mat.svd(true, true);
    let beta_products = lls.solve(&rhs, 1e-10).ok()?;

    // Extract betas from products
    let betas = extract_betas_from_products(&beta_products, num_null);

    // Try all sign combinations
    let num_signs = 1 << num_null;
    let mut best: Option<(Matrix3<f64>, NaVector3<f64>, f64)> = None;

    for sign_mask in 0..num_signs {
        let signed_betas: alloc::vec::Vec<f64> = (0..num_null)
            .map(|i| {
                if (sign_mask >> i) & 1 == 1 {
                    -betas[i]
                } else {
                    betas[i]
                }
            })
            .collect();

        // Compute camera control points
        let mut cam_cp = alloc::vec::Vec::new();
        for cp_idx in 0..num_cp {
            let mut p = NaVector3::zeros();
            for (k, &beta) in signed_betas.iter().enumerate() {
                p += beta * v_pts[k][cp_idx];
            }
            cam_cp.push(p);
        }

        if let Some((r, t)) = recover_pose(&cam_cp, alphas, num_cp, object_points) {
            let err = reprojection_error(object_points, image_points, &r, &t, fx, fy, cx, cy);
            if best.is_none() || err < best.as_ref().unwrap().2 {
                best = Some((r, t, err));
            }
        }
    }

    best
}

/// Extract individual betas from the beta product vector [b1^2, b1*b2, b2^2, ...].
fn extract_betas_from_products(products: &DVector<f64>, num_null: usize) -> alloc::vec::Vec<f64> {
    let mut betas = alloc::vec![0.0; num_null];

    if num_null == 0 {
        return betas;
    }

    // b1^2 is the first product
    let b11 = products[0].max(0.0);
    betas[0] = libm::sqrt(b11);

    if num_null >= 2 {
        // b1*b2 is products[1]
        if betas[0] > 1e-10 {
            betas[1] = products[1] / betas[0];
        } else {
            // b2^2 is products[2]
            let b22 = products[2].max(0.0);
            betas[1] = libm::sqrt(b22);
        }
    }

    if num_null >= 3 {
        // b1*b3 is products[2] for N=3 layout: [b11, b12, b13, b22, b23, b33]
        let b13_idx = 2;
        if betas[0] > 1e-10 {
            betas[2] = products[b13_idx] / betas[0];
        } else if betas[1] > 1e-10 {
            // b2*b3 is products[4]
            betas[2] = products[4] / betas[1];
        } else {
            let b33_idx = 5;
            let b33 = products[b33_idx].max(0.0);
            betas[2] = libm::sqrt(b33);
        }
    }

    betas
}

/// Recover R, t from camera-frame control points using Horn's absolute orientation.
fn recover_pose(
    camera_control_points: &[NaVector3<f64>],
    alphas: &[alloc::vec::Vec<f64>],
    num_cp: usize,
    object_points: &[Vector3],
) -> Option<(Matrix3<f64>, NaVector3<f64>)> {
    let n = object_points.len();

    // Recover 3D points in camera frame
    let mut cam_points = alloc::vec::Vec::with_capacity(n);
    for i in 0..n {
        let mut p = NaVector3::zeros();
        for j in 0..num_cp {
            p += alphas[i][j] * camera_control_points[j];
        }
        cam_points.push(p);
    }

    // Check that points are in front of camera (positive z)
    let z_mean: f64 = cam_points.iter().map(|p| p.z).sum::<f64>() / n as f64;
    let sign = if z_mean < 0.0 { -1.0 } else { 1.0 };
    let cam_points: alloc::vec::Vec<_> = cam_points.iter().map(|p| sign * p).collect();

    // Compute centroids
    let mut world_centroid = NaVector3::zeros();
    let mut cam_centroid = NaVector3::zeros();
    for i in 0..n {
        world_centroid +=
            NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z);
        cam_centroid += cam_points[i];
    }
    world_centroid /= n as f64;
    cam_centroid /= n as f64;

    // Build the cross-covariance matrix H
    let mut h = Matrix3::<f64>::zeros();
    for i in 0..n {
        let pw = NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z)
            - world_centroid;
        let pc = cam_points[i] - cam_centroid;
        h += pw * pc.transpose();
    }

    // SVD of H
    let svd = h.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;

    // R = V * U^T
    let mut r = vt.transpose() * u.transpose();

    // Ensure proper rotation (det = +1)
    if r.determinant() < 0.0 {
        let mut d = Matrix3::<f64>::identity();
        d[(2, 2)] = -1.0;
        r = vt.transpose() * d * u.transpose();
    }

    // t = cam_centroid - R * world_centroid
    let t = cam_centroid - r * world_centroid;

    Some((r, t))
}

/// Compute reprojection error for a given R, t.
pub fn reprojection_error(
    object_points: &[Vector3],
    image_points: &[Vector2],
    r: &Matrix3<f64>,
    t: &NaVector3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> f64 {
    let n = object_points.len();
    let mut total_error = 0.0;

    for i in 0..n {
        let pw = NaVector3::new(object_points[i].x, object_points[i].y, object_points[i].z);
        let pc = r * pw + t;

        if pc.z.abs() < 1e-10 {
            return f64::MAX;
        }

        let u_proj = fx * pc.x / pc.z + cx;
        let v_proj = fy * pc.y / pc.z + cy;

        let du = u_proj - image_points[i].x;
        let dv = v_proj - image_points[i].y;
        total_error += du * du + dv * dv;
    }

    total_error / n as f64
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

    fn coplanar_landmarks() -> Vec<Vector3> {
        vec![
            Vector3::new(-0.15, -0.15, 0.0),
            Vector3::new(0.15, -0.15, 0.0),
            Vector3::new(0.15, 0.15, 0.0),
            Vector3::new(-0.15, 0.15, 0.0),
        ]
    }

    fn project_points(
        pts: &[Vector3],
        r: &Matrix3<f64>,
        t: &NaVector3<f64>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
    ) -> Vec<Vector2> {
        pts.iter()
            .map(|p| {
                let pw = NaVector3::new(p.x, p.y, p.z);
                let pc = r * pw + t;
                Vector2::new(fx * pc.x / pc.z + cx, fy * pc.y / pc.z + cy)
            })
            .collect()
    }

    fn rotation_error(r1: &Matrix3<f64>, r2: &Matrix3<f64>) -> f64 {
        let r_diff = r1.transpose() * r2;
        let trace = r_diff[(0, 0)] + r_diff[(1, 1)] + r_diff[(2, 2)];
        let cos_angle = ((trace - 1.0) / 2.0).max(-1.0).min(1.0);
        libm::acos(cos_angle)
    }

    #[test]
    fn test_control_points_centroid() {
        let points = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ];
        let (cp, _, _) = compute_control_points_and_barycentrics(&points);
        assert!((cp[0].x).abs() < 1e-10, "centroid x should be 0");
        assert!((cp[0].y).abs() < 1e-10, "centroid y should be 0");
        assert!((cp[0].z).abs() < 1e-10, "centroid z should be 0");
    }

    #[test]
    fn test_barycentric_sum_to_one() {
        let points = coplanar_landmarks();
        let (_, alphas, _) = compute_control_points_and_barycentrics(&points);
        for (i, a) in alphas.iter().enumerate() {
            let sum: f64 = a.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "barycentric sum for point {} = {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_barycentric_reconstruction() {
        let points = coplanar_landmarks();
        let (cp, alphas, num_cp) = compute_control_points_and_barycentrics(&points);
        for (i, p) in points.iter().enumerate() {
            let mut reconstructed = NaVector3::zeros();
            for j in 0..num_cp {
                reconstructed += alphas[i][j] * cp[j];
            }
            let orig = NaVector3::new(p.x, p.y, p.z);
            assert!(
                (reconstructed - orig).norm() < 1e-10,
                "reconstruction error for point {}: {}",
                i,
                (reconstructed - orig).norm()
            );
        }
    }

    #[test]
    fn test_coplanar_detection() {
        let points = coplanar_landmarks();
        let (_, _, num_cp) = compute_control_points_and_barycentrics(&points);
        assert_eq!(num_cp, 3, "coplanar points should use 3 control points");
    }

    #[test]
    fn test_noncoplanar_detection() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let (_, _, num_cp) = compute_control_points_and_barycentrics(&points);
        assert_eq!(num_cp, 4, "non-coplanar points should use 4 control points");
    }

    #[test]
    fn test_m_matrix_dimensions() {
        let points = coplanar_landmarks();
        let (_, alphas, num_cp) = compute_control_points_and_barycentrics(&points);
        let img = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        let m = build_m_matrix(&alphas, &img, FX, FY, CX, CY, num_cp);
        assert_eq!(m.nrows(), 8);
        assert_eq!(m.ncols(), num_cp * 3); // 9 for planar, 12 for general
    }

    #[test]
    fn test_epnp_known_pose_noncoplanar() {
        let object_points = vec![
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

        let image_points = project_points(&object_points, &r_true, &t_true, FX, FY, CX, CY);
        let cam = camera_matrix();

        let (r, t) = solve_epnp(&object_points, &image_points, &cam).unwrap();

        let pos_err = (t - t_true).norm();
        let rot_err = rotation_error(&r, &r_true);

        assert!(pos_err < 0.1, "position error too large: {}", pos_err);
        assert!(rot_err < 0.1, "rotation error too large: {} rad", rot_err);
    }

    #[test]
    fn test_epnp_known_pose_coplanar() {
        let object_points = coplanar_landmarks();

        let rvec = [0.0, 0.3, 0.0];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 1.5);

        let image_points = project_points(&object_points, &r_true, &t_true, FX, FY, CX, CY);
        let cam = camera_matrix();

        let result = solve_epnp(&object_points, &image_points, &cam);
        assert!(
            result.is_ok(),
            "EPnP should succeed for coplanar points: {:?}",
            result.err()
        );

        let (r, t) = result.unwrap();
        let reproj_err = reprojection_error(&object_points, &image_points, &r, &t, FX, FY, CX, CY);
        // EPnP with 4 coplanar points may have moderate error
        assert!(
            reproj_err < 100.0,
            "reprojection error too large: {}",
            reproj_err
        );
    }

    #[test]
    fn test_epnp_reference_set1() {
        let object_points = coplanar_landmarks();
        let image_points = vec![
            Vector2::new(1324.333, 208.0732),
            Vector2::new(1604.393, 129.3065),
            Vector2::new(1604.393, 403.1022),
            Vector2::new(1324.333, 429.3577),
        ];
        let cam = camera_matrix();

        let result = solve_epnp(&object_points, &image_points, &cam);
        assert!(
            result.is_ok(),
            "EPnP should succeed for set 1: {:?}",
            result.err()
        );

        let (r, t) = result.unwrap();
        let reproj_err = reprojection_error(&object_points, &image_points, &r, &t, FX, FY, CX, CY);
        // OpenCV EPnP gets <1px on this set
        assert!(
            reproj_err < 5.0,
            "reprojection error for set 1: {}",
            reproj_err
        );
    }

    #[test]
    fn test_epnp_reference_set2() {
        let object_points = coplanar_landmarks();
        let image_points = vec![
            Vector2::new(379.0064, 743.9628),
            Vector2::new(552.0745, 570.8946),
            Vector2::new(725.1426, 743.9628),
            Vector2::new(552.0745, 917.0309),
        ];
        let cam = camera_matrix();

        let result = solve_epnp(&object_points, &image_points, &cam);
        assert!(
            result.is_ok(),
            "EPnP should succeed for set 2: {:?}",
            result.err()
        );

        let (r, t) = result.unwrap();
        let reproj_err = reprojection_error(&object_points, &image_points, &r, &t, FX, FY, CX, CY);
        assert!(
            reproj_err < 5.0,
            "reprojection error for set 2: {}",
            reproj_err
        );
    }

    #[test]
    fn test_epnp_many_points() {
        let mut object_points = alloc::vec::Vec::new();
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
            object_points.push(Vector3::new(x, y, z));
        }

        let rvec = [0.2, -0.3, 0.1];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.0, 0.0, 5.0);

        let image_points = project_points(&object_points, &r_true, &t_true, FX, FY, CX, CY);
        let cam = camera_matrix();

        let (r, t) = solve_epnp(&object_points, &image_points, &cam).unwrap();
        let reproj_err = reprojection_error(&object_points, &image_points, &r, &t, FX, FY, CX, CY);
        assert!(
            reproj_err < 1.0,
            "reprojection error for many points: {}",
            reproj_err
        );
    }

    #[test]
    fn test_epnp_minimum_points() {
        let object_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        let rvec = [0.1, 0.2, -0.1];
        let r_true = crate::rodrigues::rvec_to_rotation_matrix(&rvec).to_na();
        let t_true = NaVector3::new(0.5, 0.5, 4.0);

        let image_points = project_points(&object_points, &r_true, &t_true, FX, FY, CX, CY);
        let cam = camera_matrix();

        let result = solve_epnp(&object_points, &image_points, &cam);
        assert!(result.is_ok(), "EPnP should work with exactly 4 points");
    }
}
