//! Rodrigues rotation vector ↔ matrix conversion (OpenCV-compatible).

use crate::types::Matrix3x3;
use nalgebra::{Matrix3, Vector3};

/// Convert a Rodrigues rotation vector to a 3×3 rotation matrix.
///
/// The rotation vector encodes the axis of rotation (direction) and
/// the angle of rotation (magnitude in radians).
///
/// Algorithm:
/// θ = ||rvec||
/// if θ ≈ 0: R = I + [rvec]×
/// else: k = rvec/θ, K = skew(k), R = I + sin(θ)·K + (1-cos(θ))·K²
pub fn rvec_to_rotation_matrix(rvec: &[f64; 3]) -> Matrix3x3 {
    let theta = libm::sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);

    if theta < 1e-12 {
        // Small angle: R ≈ I + [rvec]×
        let mut m = Matrix3x3::identity();
        // skew symmetric of rvec:
        // [  0  -rz  ry ]
        // [  rz  0  -rx ]
        // [ -ry  rx  0  ]
        // In column-major storage (col, row):
        m.set(0, 1, rvec[2]); // col 0, row 1 = rz
        m.set(0, 2, -rvec[1]); // col 0, row 2 = -ry
        m.set(1, 0, -rvec[2]); // col 1, row 0 = -rz
        m.set(1, 2, rvec[0]); // col 1, row 2 = rx
        m.set(2, 0, rvec[1]); // col 2, row 0 = ry
        m.set(2, 1, -rvec[0]); // col 2, row 1 = -rx
        return m;
    }

    let k = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta];

    // Skew-symmetric matrix K of unit axis k
    let k_mat = Matrix3::new(0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0);

    let sin_t = libm::sin(theta);
    let cos_t = libm::cos(theta);

    let i = Matrix3::<f64>::identity();
    let k_sq = k_mat * k_mat;

    // Rodrigues formula: R = I + sin(θ)·K + (1-cos(θ))·K²
    let r = i + sin_t * k_mat + (1.0 - cos_t) * k_sq;

    Matrix3x3::from_na(&r)
}

/// Convert a 3x3 rotation matrix to a Rodrigues rotation vector.
///
/// Algorithm:
/// θ = acos(clamp((trace(R) - 1) / 2, -1, 1))
/// if θ ≈ 0: small-angle approximation
/// if |θ - π| < ε: near-180° handling
/// else: k = [R₂₁-R₁₂, R₀₂-R₂₀, R₁₀-R₀₁] / (2·sin(θ)), return k·θ
pub fn rotation_matrix_to_rvec(m: &Matrix3x3) -> [f64; 3] {
    let r = m.to_na();

    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let cos_theta = clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    let theta = libm::acos(cos_theta);

    if theta < 1e-12 {
        // Small angle: rvec ≈ [R₂₁-R₁₂, R₀₂-R₂₀, R₁₀-R₀₁] / 2
        return [
            (r[(2, 1)] - r[(1, 2)]) / 2.0,
            (r[(0, 2)] - r[(2, 0)]) / 2.0,
            (r[(1, 0)] - r[(0, 1)]) / 2.0,
        ];
    }

    if libm::fabs(theta - core::f64::consts::PI) < 1e-6 {
        // Near 180°: find the column of (R + I) with largest norm
        let r_plus_i = r + Matrix3::<f64>::identity();

        let mut best_col = 0;
        let mut best_norm = 0.0;
        for col in 0..3 {
            let n = r_plus_i[(0, col)] * r_plus_i[(0, col)]
                + r_plus_i[(1, col)] * r_plus_i[(1, col)]
                + r_plus_i[(2, col)] * r_plus_i[(2, col)];
            if n > best_norm {
                best_norm = n;
                best_col = col;
            }
        }

        let v = Vector3::new(
            r_plus_i[(0, best_col)],
            r_plus_i[(1, best_col)],
            r_plus_i[(2, best_col)],
        );
        let v_norm = v.norm();
        if v_norm < 1e-15 {
            return [core::f64::consts::PI, 0.0, 0.0];
        }
        let k = v / v_norm;
        return [k[0] * theta, k[1] * theta, k[2] * theta];
    }

    // General case
    let sin_theta = libm::sin(theta);
    let k = [
        (r[(2, 1)] - r[(1, 2)]) / (2.0 * sin_theta),
        (r[(0, 2)] - r[(2, 0)]) / (2.0 * sin_theta),
        (r[(1, 0)] - r[(0, 1)]) / (2.0 * sin_theta),
    ];

    [k[0] * theta, k[1] * theta, k[2] * theta]
}

#[inline]
fn clamp(val: f64, min: f64, max: f64) -> f64 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    const EPS: f64 = 1e-10;

    fn matrices_close(a: &Matrix3x3, b: &Matrix3x3, tol: f64) -> bool {
        for i in 0..9 {
            if libm::fabs(a.m[i] - b.m[i]) > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_identity_rvec() {
        let r = rvec_to_rotation_matrix(&[0.0, 0.0, 0.0]);
        let id = Matrix3x3::identity();
        assert!(matrices_close(&r, &id, EPS));
    }

    #[test]
    fn test_90deg_x() {
        let r = rvec_to_rotation_matrix(&[FRAC_PI_2, 0.0, 0.0]);
        // Expected: [[1,0,0],[0,0,-1],[0,1,0]]
        // col-major: col0=[1,0,0], col1=[0,0,1], col2=[0,-1,0]
        assert!((r.get(0, 0) - 1.0).abs() < EPS);
        assert!(r.get(0, 1).abs() < EPS);
        assert!(r.get(0, 2).abs() < EPS);
        assert!(r.get(1, 0).abs() < EPS);
        assert!(r.get(1, 1).abs() < EPS);
        assert!((r.get(1, 2) - 1.0).abs() < EPS);
        assert!(r.get(2, 0).abs() < EPS);
        assert!((r.get(2, 1) - (-1.0)).abs() < EPS);
        assert!(r.get(2, 2).abs() < EPS);
    }

    #[test]
    fn test_90deg_y() {
        let r = rvec_to_rotation_matrix(&[0.0, FRAC_PI_2, 0.0]);
        // Expected: [[0,0,1],[0,1,0],[-1,0,0]]
        // col-major: col0=[0,0,-1], col1=[0,1,0], col2=[1,0,0]
        assert!(r.get(0, 0).abs() < EPS);
        assert!(r.get(0, 1).abs() < EPS);
        assert!((r.get(0, 2) - (-1.0)).abs() < EPS);
        assert!(r.get(1, 0).abs() < EPS);
        assert!((r.get(1, 1) - 1.0).abs() < EPS);
        assert!(r.get(1, 2).abs() < EPS);
        assert!((r.get(2, 0) - 1.0).abs() < EPS);
        assert!(r.get(2, 1).abs() < EPS);
        assert!(r.get(2, 2).abs() < EPS);
    }

    #[test]
    fn test_90deg_z() {
        let r = rvec_to_rotation_matrix(&[0.0, 0.0, FRAC_PI_2]);
        // Expected: [[0,-1,0],[1,0,0],[0,0,1]]
        // col-major: col0=[0,1,0], col1=[-1,0,0], col2=[0,0,1]
        assert!(r.get(0, 0).abs() < EPS);
        assert!((r.get(0, 1) - 1.0).abs() < EPS);
        assert!(r.get(0, 2).abs() < EPS);
        assert!((r.get(1, 0) - (-1.0)).abs() < EPS);
        assert!(r.get(1, 1).abs() < EPS);
        assert!(r.get(1, 2).abs() < EPS);
        assert!(r.get(2, 0).abs() < EPS);
        assert!(r.get(2, 1).abs() < EPS);
        assert!((r.get(2, 2) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_arbitrary_rotation() {
        use nalgebra::Rotation3;
        let rvec = [0.1, -0.2, 0.3];
        let our_r = rvec_to_rotation_matrix(&rvec);
        let na_r = Rotation3::new(nalgebra::Vector3::new(rvec[0], rvec[1], rvec[2]));
        let expected = Matrix3x3::from_na(na_r.matrix());
        assert!(matrices_close(&our_r, &expected, EPS));
    }

    #[test]
    fn test_roundtrip() {
        let test_rvecs = [
            [0.5, -0.3, 0.7],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.5],
            [0.1, 0.1, 0.1],
            [2.5, -1.2, 0.8],
            [-0.7, 0.4, -0.2],
            [0.01, -0.02, 0.03],
            [1.5, 1.5, 0.0],
            [-2.0, 0.5, 1.0],
        ];
        for rvec in &test_rvecs {
            let m = rvec_to_rotation_matrix(rvec);
            let back = rotation_matrix_to_rvec(&m);
            let m2 = rvec_to_rotation_matrix(&back);
            assert!(
                matrices_close(&m, &m2, EPS),
                "round-trip failed for rvec {:?}",
                rvec
            );
        }
    }

    #[test]
    fn test_180deg() {
        let r = rvec_to_rotation_matrix(&[PI, 0.0, 0.0]);
        // 180° about X: [[1,0,0],[0,-1,0],[0,0,-1]]
        assert!((r.get(0, 0) - 1.0).abs() < EPS);
        assert!((r.get(1, 1) - (-1.0)).abs() < EPS);
        assert!((r.get(2, 2) - (-1.0)).abs() < EPS);
        // Roundtrip
        let back = rotation_matrix_to_rvec(&r);
        let m2 = rvec_to_rotation_matrix(&back);
        assert!(matrices_close(&r, &m2, EPS));
    }

    #[test]
    fn test_small_angle() {
        let tiny = 1e-15;
        let r = rvec_to_rotation_matrix(&[tiny, tiny, tiny]);
        let id = Matrix3x3::identity();
        assert!(matrices_close(&r, &id, 1e-10));
    }

    #[test]
    fn test_against_nalgebra_rodrigues() {
        // Test against nalgebra's Rotation3 for various angles
        use nalgebra::Rotation3;
        let test_rvecs = [[1.2, -0.5, 0.8], [0.0, 0.0, 2.5], [-1.0, 1.0, -1.0]];
        for rvec in &test_rvecs {
            let our_r = rvec_to_rotation_matrix(rvec);
            let na_r = Rotation3::new(nalgebra::Vector3::new(rvec[0], rvec[1], rvec[2]));
            let expected = Matrix3x3::from_na(na_r.matrix());
            assert!(
                matrices_close(&our_r, &expected, EPS),
                "mismatch for rvec {:?}",
                rvec
            );
        }
    }
}
