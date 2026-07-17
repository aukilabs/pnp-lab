use alloc::string::String;
use nalgebra::{Matrix3, Quaternion as NaQuaternion, UnitQuaternion, Vector3 as NaVector3};

/// 2D point (image coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

impl Vector2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn length(&self) -> f64 {
        libm::sqrt(self.x * self.x + self.y * self.y)
    }
}

/// 3D point (world coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn length(&self) -> f64 {
        libm::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn to_na(&self) -> NaVector3<f64> {
        NaVector3::new(self.x, self.y, self.z)
    }

    pub fn from_na(v: &NaVector3<f64>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

/// 3D ray in world/tracking coordinates.
///
/// The Ark square pose solver accepts one ray per QR corner, ordered
/// top-left, top-right, bottom-right, bottom-left. Ray directions may be
/// unnormalized; the solver normalizes them before estimating corner
/// distances. Units must match the physical QR size passed to the solver
/// (meters in the AR calibration flow).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray3 {
    pub origin: Vector3,
    pub direction: Vector3,
}

/// Quaternion in Hamilton convention (x, y, z, w) where w is the scalar part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quaternion {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn norm(&self) -> f64 {
        libm::sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return Self::identity();
        }
        Self {
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
            w: self.w / n,
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Hamilton product: self * other
    pub fn multiply(&self, other: &Quaternion) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    /// Convert to nalgebra UnitQuaternion.
    /// nalgebra stores quaternion as (w, x, y, z) internally.
    pub fn to_na_unit(&self) -> UnitQuaternion<f64> {
        let q = NaQuaternion::new(self.w, self.x, self.y, self.z);
        UnitQuaternion::from_quaternion(q)
    }

    /// Create from nalgebra UnitQuaternion.
    pub fn from_na_unit(uq: &UnitQuaternion<f64>) -> Self {
        let q = uq.quaternion();
        Self {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        }
    }
}

/// 3x3 matrix in column-major order.
/// Storage: m[col * 3 + row], so m[0..3] is column 0, m[3..6] is column 1, etc.
/// Field naming: M<col><row> — M00 = column 0, row 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3x3 {
    pub m: [f64; 9],
}

impl Matrix3x3 {
    /// Create from column-major array.
    pub fn new(m: [f64; 9]) -> Self {
        Self { m }
    }

    /// Create identity matrix.
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create zero matrix.
    pub fn zeros() -> Self {
        Self { m: [0.0; 9] }
    }

    /// Get element at (col, row) — column-major.
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> f64 {
        self.m[col * 3 + row]
    }

    /// Set element at (col, row) — column-major.
    #[inline]
    pub fn set(&mut self, col: usize, row: usize, val: f64) {
        self.m[col * 3 + row] = val;
    }

    /// Transpose: swap rows and columns.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros();
        for col in 0..3 {
            for row in 0..3 {
                result.set(row, col, self.get(col, row));
            }
        }
        result
    }

    /// Convert to nalgebra Matrix3 (column-major, which matches our storage).
    pub fn to_na(&self) -> Matrix3<f64> {
        // nalgebra::Matrix3 is column-major, same as our storage
        Matrix3::from_column_slice(&self.m)
    }

    /// Create from nalgebra Matrix3.
    pub fn from_na(m: &Matrix3<f64>) -> Self {
        let mut result = Self::zeros();
        for col in 0..3 {
            for row in 0..3 {
                result.set(col, row, m[(row, col)]);
            }
        }
        result
    }

    /// Create a camera intrinsics matrix from fx, fy, cx, cy.
    /// Stored in column-major order so that when transposed to row-major,
    /// we get [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    pub fn camera_matrix(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        // Column-major: the Matrix3x3 stores the transposed version of the
        // row-major camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        // M00=fx, M10=0, M20=cx => column 0 row 0 = fx, col 1 row 0 = 0, col 2 row 0 = cx
        // But in the reference code, Mij means column i, row j.
        // The JS test stores: M00=fx, M10=0, M20=cx, M01=0, M11=fy, M21=cy, M02=0, M12=0, M22=1
        // In column-major storage: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        let mut m = Self::zeros();
        m.set(0, 0, fx); // M00 = fx
        m.set(1, 1, fy); // M11 = fy
        m.set(2, 0, cx); // M20 = cx
        m.set(2, 1, cy); // M21 = cy
        m.set(2, 2, 1.0); // M22 = 1
        m
    }
}

/// Convert a rotation matrix to a quaternion using nalgebra.
pub fn rotation_matrix_to_quaternion(m: &Matrix3x3) -> Quaternion {
    let na_mat = m.to_na();
    let rotation = nalgebra::Rotation3::from_matrix_unchecked(na_mat);
    let uq = UnitQuaternion::from_rotation_matrix(&rotation);
    Quaternion::from_na_unit(&uq)
}

/// 6-DoF pose: position + rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: Vector3,
    pub rotation: Quaternion,
}

impl Pose {
    pub fn new(position: Vector3, rotation: Quaternion) -> Self {
        Self { position, rotation }
    }

    pub fn identity() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
        }
    }
}

/// Pose estimate recovered from four square-corner rays.
///
/// `pose` is in the same world/tracking coordinate frame as the input rays.
/// Its rotation follows Unity Ark's convention: local +X points from the
/// top-left corner to the top-right corner, local +Y points upward on the QR
/// face, and local +Z is `cross(+X, +Y)`.
///
/// `confidence` is a bounded [0, 1] score derived from the normalized residual
/// error; exact synthetic geometry should be close to 1. `ray_distances`
/// contains the optimized positive distance from each input ray origin to its
/// corresponding corner in top-left, top-right, bottom-right, bottom-left
/// order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquarePoseEstimate {
    pub pose: Pose,
    pub confidence: f64,
    pub normalized_corner_error: f64,
    pub ray_distances: [f64; 4],
}

/// A known 3D point in world coordinates.
#[derive(Debug, Clone)]
pub struct Landmark {
    pub id: String,
    pub position: Vector3,
}

/// A 2D observation of a landmark in image coordinates.
#[derive(Debug, Clone)]
pub struct LandmarkObservation {
    pub id: String,
    pub position: Vector2,
}

/// Solver method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolvePnpMethod {
    EPnP,
    Iterative,
    SQPnP,
}

/// Errors that can occur during PnP solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnpError {
    InsufficientPoints,
    SolverFailed,
    MismatchedCounts,
}

impl core::fmt::Display for PnpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PnpError::InsufficientPoints => write!(f, "insufficient points for solver"),
            PnpError::SolverFailed => write!(f, "solver failed to converge"),
            PnpError::MismatchedCounts => write!(f, "landmark and observation counts do not match"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_matrix3x3_identity() {
        let m = Matrix3x3::identity();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(
                        (m.get(i, j) - 1.0).abs() < EPS,
                        "diagonal ({},{}) should be 1",
                        i,
                        j
                    );
                } else {
                    assert!(
                        m.get(i, j).abs() < EPS,
                        "off-diagonal ({},{}) should be 0",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_matrix3x3_transpose() {
        let mut m = Matrix3x3::zeros();
        m.set(0, 1, 1.0); // col 0, row 1
        m.set(1, 2, 2.0); // col 1, row 2
        m.set(2, 0, 3.0); // col 2, row 0
        let t = m.transpose();
        assert!((t.get(1, 0) - 1.0).abs() < EPS); // was col 0 row 1, now col 1 row 0
        assert!((t.get(2, 1) - 2.0).abs() < EPS);
        assert!((t.get(0, 2) - 3.0).abs() < EPS);
    }

    #[test]
    fn test_matrix3x3_na_roundtrip() {
        let original = Matrix3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let na = original.to_na();
        let back = Matrix3x3::from_na(&na);
        for i in 0..9 {
            assert!(
                (original.m[i] - back.m[i]).abs() < EPS,
                "element {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert!((q.x).abs() < EPS);
        assert!((q.y).abs() < EPS);
        assert!((q.z).abs() < EPS);
        assert!((q.w - 1.0).abs() < EPS);
    }

    #[test]
    fn test_quaternion_normalize() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let n = q.normalize();
        assert!((n.norm() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_quaternion_multiply() {
        // 90° around X * 90° around Y
        let half = core::f64::consts::FRAC_PI_4;
        let s = libm::sin(half);
        let c = libm::cos(half);
        let qx = Quaternion::new(s, 0.0, 0.0, c);
        let qy = Quaternion::new(0.0, s, 0.0, c);
        let result = qx.multiply(&qy);
        // Verify via nalgebra
        let na_result = qx.to_na_unit() * qy.to_na_unit();
        let expected = Quaternion::from_na_unit(&na_result);
        assert!((result.x - expected.x).abs() < EPS);
        assert!((result.y - expected.y).abs() < EPS);
        assert!((result.z - expected.z).abs() < EPS);
        assert!((result.w - expected.w).abs() < EPS);
    }

    #[test]
    fn test_quaternion_conjugate() {
        // Use a unit quaternion so q * conjugate(q) = identity
        let q = Quaternion::new(0.267, 0.535, 0.802, 0.0).normalize();
        let conj = q.conjugate();
        let product = q.multiply(&conj);
        // Should be approximately identity
        assert!((product.x).abs() < 1e-10);
        assert!((product.y).abs() < 1e-10);
        assert!((product.z).abs() < 1e-10);
        assert!((product.w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_na_roundtrip() {
        let original = Quaternion::new(0.1, 0.2, 0.3, 0.9).normalize();
        let na = original.to_na_unit();
        let back = Quaternion::from_na_unit(&na);
        assert!((original.x - back.x).abs() < EPS);
        assert!((original.y - back.y).abs() < EPS);
        assert!((original.z - back.z).abs() < EPS);
        assert!((original.w - back.w).abs() < EPS);
    }

    #[test]
    fn test_camera_matrix_convention() {
        let fx = 815.8511;
        let fy = 815.8511;
        let cx = 960.0;
        let cy = 540.0;
        let m = Matrix3x3::camera_matrix(fx, fy, cx, cy);
        // Column-major Matrix3x3 to_na() should directly produce
        // the standard camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
        let na = m.to_na();
        // nalgebra Matrix3 is accessed as na[(row, col)]
        assert!((na[(0, 0)] - fx).abs() < EPS);
        assert!((na[(0, 1)]).abs() < EPS);
        assert!((na[(0, 2)] - cx).abs() < EPS);
        assert!((na[(1, 0)]).abs() < EPS);
        assert!((na[(1, 1)] - fy).abs() < EPS);
        assert!((na[(1, 2)] - cy).abs() < EPS);
        assert!((na[(2, 0)]).abs() < EPS);
        assert!((na[(2, 1)]).abs() < EPS);
        assert!((na[(2, 2)] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_identity_matrix_to_quaternion() {
        let m = Matrix3x3::identity();
        let q = rotation_matrix_to_quaternion(&m);
        assert!((q.x).abs() < 1e-10);
        assert!((q.y).abs() < 1e-10);
        assert!((q.z).abs() < 1e-10);
        assert!((q.w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_90deg_x_to_quaternion() {
        // 90° rotation about X: R = [[1,0,0],[0,0,-1],[0,1,0]]
        let m = Matrix3x3::new([
            1.0, 0.0, 0.0, // col 0
            0.0, 0.0, 1.0, // col 1
            0.0, -1.0, 0.0, // col 2
        ]);
        let q = rotation_matrix_to_quaternion(&m);
        let expected_s = libm::sin(core::f64::consts::FRAC_PI_4);
        let expected_c = libm::cos(core::f64::consts::FRAC_PI_4);
        // q or -q are equivalent, so check angle
        let angle = 2.0 * libm::acos(q.w.abs().min(1.0));
        assert!((angle - core::f64::consts::FRAC_PI_2).abs() < 1e-10);
        // Axis should be X
        let norm = libm::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        if norm > 1e-10 {
            assert!((q.x / norm).abs() > 0.99);
        }
    }

    #[test]
    fn test_roundtrip_matrix_quat_matrix() {
        use nalgebra::Rotation3;
        let axis = nalgebra::Unit::new_normalize(NaVector3::new(1.0, 2.0, 3.0));
        let rot = Rotation3::from_axis_angle(&axis, 1.2);
        let m = Matrix3x3::from_na(rot.matrix());
        let q = rotation_matrix_to_quaternion(&m);
        let uq = q.to_na_unit();
        let back = uq.to_rotation_matrix();
        let m_back = Matrix3x3::from_na(back.matrix());
        for i in 0..9 {
            assert!(
                (m.m[i] - m_back.m[i]).abs() < 1e-10,
                "element {} mismatch: {} vs {}",
                i,
                m.m[i],
                m_back.m[i]
            );
        }
    }

    #[test]
    fn test_against_nalgebra() {
        use nalgebra::Rotation3;
        let axis = nalgebra::Unit::new_normalize(NaVector3::new(0.5, -0.3, 0.8));
        let rot = Rotation3::from_axis_angle(&axis, 2.1);
        let m = Matrix3x3::from_na(rot.matrix());
        let our_q = rotation_matrix_to_quaternion(&m);
        let na_q = UnitQuaternion::from_rotation_matrix(&rot);
        let na_qv = Quaternion::from_na_unit(&na_q);
        // Account for sign ambiguity
        let dot = our_q.x * na_qv.x + our_q.y * na_qv.y + our_q.z * na_qv.z + our_q.w * na_qv.w;
        assert!(
            dot.abs() > 1.0 - 1e-10,
            "quaternions should match (up to sign)"
        );
    }
}
