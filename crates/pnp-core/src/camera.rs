//! Pinhole camera model with optional Brown–Conrady distortion.
//!
//! Distortion coefficients follow OpenCV order:
//! `[k1, k2, p1, p2]` (4), `[k1, k2, p1, p2, k3]` (5), or
//! `[k1, k2, p1, p2, k3, k4, k5, k6]` (8, rational model).

use crate::types::{Matrix3x3, PnpError, Ray3, Vector2, Vector3};
use alloc::vec::Vec;
use nalgebra::Matrix3;

const MIN_FOCAL: f64 = 1e-12;
const UNDISTORT_ITERS: usize = 10;
const UNDISTORT_EPS: f64 = 1e-12;

/// Calibrated monocular camera: pinhole intrinsics + optional distortion.
///
/// Focal lengths and principal point are in **pixels**. Distortion uses the
/// OpenCV Brown–Conrady (and optional rational) model.
///
/// # Example
///
/// ```
/// use pnp_core::Camera;
/// let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
/// assert!(!cam.has_distortion());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// OpenCV-ordered coefficients; empty means ideal pinhole (no distortion).
    pub dist: Vec<f64>,
}

impl Camera {
    /// Build a pinhole camera with no distortion.
    ///
    /// # Errors
    ///
    /// Returns [`PnpError::SolverFailed`] if focal lengths are non-finite or
    /// near zero, or if the principal point is non-finite.
    pub fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> Result<Self, PnpError> {
        Self::new(fx, fy, cx, cy, &[])
    }

    /// Build a camera from focal lengths, principal point, and OpenCV distortion.
    ///
    /// `dist` must be empty or length 4, 5, or 8.
    ///
    /// # Errors
    ///
    /// Returns [`PnpError::SolverFailed`] on invalid intrinsics or coefficient
    /// count / non-finite values.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, dist: &[f64]) -> Result<Self, PnpError> {
        if !fx.is_finite() || !fy.is_finite() || !cx.is_finite() || !cy.is_finite() {
            return Err(PnpError::SolverFailed);
        }
        if fx.abs() < MIN_FOCAL || fy.abs() < MIN_FOCAL {
            return Err(PnpError::SolverFailed);
        }
        if !matches!(dist.len(), 0 | 4 | 5 | 8) {
            return Err(PnpError::SolverFailed);
        }
        for c in dist {
            if !c.is_finite() {
                return Err(PnpError::SolverFailed);
            }
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            dist: dist.to_vec(),
        })
    }

    /// Build from a column-major camera matrix `K` plus optional distortion.
    ///
    /// Reads `fx = K₀₀`, `fy = K₁₁`, `cx = K₀₂`, `cy = K₁₂`.
    pub fn from_matrix(matrix: &Matrix3x3, dist: &[f64]) -> Result<Self, PnpError> {
        let na = matrix.to_na();
        Self::new(na[(0, 0)], na[(1, 1)], na[(0, 2)], na[(1, 2)], dist)
    }

    /// `true` if any stored distortion coefficient is non-zero.
    pub fn has_distortion(&self) -> bool {
        self.dist.iter().any(|c| c.abs() > 0.0)
    }

    /// Column-major 3×3 intrinsics matrix `K` only (distortion not embedded).
    pub fn matrix(&self) -> Matrix3x3 {
        Matrix3x3::camera_matrix(self.fx, self.fy, self.cx, self.cy)
    }

    /// nalgebra form of `K` for the algebraic solvers.
    pub fn matrix_na(&self) -> Matrix3<f64> {
        self.matrix().to_na()
    }

    /// Project a camera-space 3D point to a **distorted** image pixel.
    ///
    /// Expects OpenCV camera coordinates (looking down +Z). Returns `None` if
    /// the point is at or behind the camera plane (`z ≈ 0` or non-finite).
    pub fn project(&self, point_cam: Vector3) -> Option<Vector2> {
        if !point_cam.z.is_finite() || point_cam.z.abs() < MIN_FOCAL {
            return None;
        }
        let x = point_cam.x / point_cam.z;
        let y = point_cam.y / point_cam.z;
        let (xd, yd) = self.distort_normalized(x, y);
        Some(Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy))
    }

    /// Undistort a distorted image pixel to the ideal pinhole pixel.
    ///
    /// Identity when [`Self::has_distortion`] is false. Uses fixed-point
    /// iteration on the normalized plane (OpenCV-style).
    pub fn undistort_pixel(&self, pixel: Vector2) -> Vector2 {
        if !self.has_distortion() {
            return pixel;
        }
        let xd = (pixel.x - self.cx) / self.fx;
        let yd = (pixel.y - self.cy) / self.fy;
        let (xn, yn) = self.undistort_normalized(xd, yd);
        Vector2::new(self.fx * xn + self.cx, self.fy * yn + self.cy)
    }

    /// Undistort each pixel with [`Self::undistort_pixel`].
    pub fn undistort_pixels(&self, pixels: &[Vector2]) -> Vec<Vector2> {
        pixels.iter().map(|p| self.undistort_pixel(*p)).collect()
    }

    /// Pixel → camera-space ray for **OpenGL-style** cameras (look down −Z, Y up).
    ///
    /// After optional undistortion:
    /// `x = (u − cx) / fx`, `y = −(v − cy) / fy`, `z = −1`.
    ///
    /// Matches the Expo `cameraPixelToCameraRay` helper and is used by
    /// [`crate::estimate_square_pose_from_pixels`]. Directions are left
    /// unnormalized; the square-pose solver normalizes them.
    pub fn unproject_opengl_ray(&self, pixel: Vector2) -> Ray3 {
        let p = self.undistort_pixel(pixel);
        Ray3 {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new((p.x - self.cx) / self.fx, -(p.y - self.cy) / self.fy, -1.0),
        }
    }

    /// Pixel → ray in **OpenCV** camera frame (look +Z, Y down in image).
    ///
    /// After optional undistortion: direction
    /// `((u − cx) / fx, (v − cy) / fy, 1)`, origin at the camera.
    ///
    /// Used by stereo triangulation. Do **not** use
    /// [`Self::unproject_opengl_ray`] for OpenCV-frame stereo math (that
    /// flips Y and uses `z = −1`).
    pub fn unproject_opencv_ray(&self, pixel: Vector2) -> Ray3 {
        let p = self.undistort_pixel(pixel);
        Ray3 {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(
                (p.x - self.cx) / self.fx,
                (p.y - self.cy) / self.fy,
                1.0,
            ),
        }
    }

    /// Normalized-plane distortion (OpenCV Brown–Conrady / rational).
    fn distort_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        if !self.has_distortion() {
            return (x, y);
        }
        let k1 = self.coeff(0);
        let k2 = self.coeff(1);
        let p1 = self.coeff(2);
        let p2 = self.coeff(3);
        let k3 = self.coeff(4);
        let k4 = self.coeff(5);
        let k5 = self.coeff(6);
        let k6 = self.coeff(7);

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let mut radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        if self.dist.len() >= 8 {
            let denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
            if denom.abs() > MIN_FOCAL {
                radial /= denom;
            }
        }
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        (xd, yd)
    }

    /// Invert distortion on the normalized plane via fixed-point iteration.
    fn undistort_normalized(&self, xd: f64, yd: f64) -> (f64, f64) {
        if !self.has_distortion() {
            return (xd, yd);
        }
        let mut x = xd;
        let mut y = yd;
        for _ in 0..UNDISTORT_ITERS {
            let (px, py) = self.distort_normalized(x, y);
            let err_x = px - xd;
            let err_y = py - yd;
            x -= err_x;
            y -= err_y;
            if err_x * err_x + err_y * err_y < UNDISTORT_EPS {
                break;
            }
        }
        (x, y)
    }

    fn coeff(&self, index: usize) -> f64 {
        self.dist.get(index).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn pinhole_project_roundtrip_principal_point() {
        let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let pixel = cam.project(Vector3::new(0.0, 0.0, 2.0)).expect("project");
        assert!((pixel.x - 320.0).abs() < EPS);
        assert!((pixel.y - 240.0).abs() < EPS);
    }

    #[test]
    fn distort_undistort_roundtrip() {
        let cam = Camera::new(
            815.8511,
            815.8511,
            960.0,
            540.0,
            &[0.12, -0.05, 0.001, -0.002, 0.01],
        )
        .unwrap();

        let ideal = Vector2::new(1020.0, 480.0);
        let xn = (ideal.x - cam.cx) / cam.fx;
        let yn = (ideal.y - cam.cy) / cam.fy;
        let (xd, yd) = cam.distort_normalized(xn, yn);
        let distorted = Vector2::new(cam.fx * xd + cam.cx, cam.fy * yd + cam.cy);
        let recovered = cam.undistort_pixel(distorted);

        assert!((recovered.x - ideal.x).abs() < 1e-6);
        assert!((recovered.y - ideal.y).abs() < 1e-6);
    }

    #[test]
    fn opengl_ray_principal_point_is_forward() {
        let cam = Camera::pinhole(100.0, 200.0, 50.0, 60.0).unwrap();
        let ray = cam.unproject_opengl_ray(Vector2::new(50.0, 60.0));
        assert!((ray.direction.x).abs() < EPS);
        assert!((ray.direction.y).abs() < EPS);
        assert!((ray.direction.z + 1.0).abs() < EPS);
    }

    #[test]
    fn opencv_ray_principal_point_is_forward_z() {
        let cam = Camera::pinhole(100.0, 100.0, 50.0, 40.0).unwrap();
        let ray = cam.unproject_opencv_ray(Vector2::new(50.0, 40.0));
        assert!((ray.direction.x).abs() < 1e-12);
        assert!((ray.direction.y).abs() < 1e-12);
        assert!((ray.direction.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_focal_and_coeff_count() {
        assert!(Camera::pinhole(0.0, 100.0, 0.0, 0.0).is_err());
        assert!(Camera::new(100.0, 100.0, 0.0, 0.0, &[1.0, 2.0, 3.0]).is_err());
    }
}
