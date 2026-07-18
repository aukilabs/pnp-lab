//! Calibrated stereo rig: two monocular cameras + fixed extrinsics.
//!
//! # Frames
//! - **Left camera** is the primary frame for triangulation and stereo PnP seeds.
//! - `right_from_left` is the pose of the **right** camera expressed in the
//!   **left** camera frame: a point in left coords maps to right as
//!   `X_right = R * X_left + t` where `(R,t)` come from `right_from_left`.
//! - **Stereo internal math uses OpenCV-frame poses** for `right_from_left`
//!   (+Z forward, Y down in image). Pixel→ray conversion for stereo must use
//!   [`Camera::unproject_opencv_ray`], not [`Camera::unproject_opengl_ray`].
//!
//! Callers building a rig from device calibration must convert into this
//! convention before constructing [`StereoRig`].

use crate::camera::Camera;
use crate::types::{PnpError, Pose, Vector2};
use alloc::string::String;

const MIN_BASELINE: f64 = 1e-6;

/// Calibrated stereo pair.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoRig {
    pub left: Camera,
    pub right: Camera,
    /// Pose of the right camera in the left camera frame.
    pub right_from_left: Pose,
}

/// Per-landmark observations in one or both eyes (pixel coords, possibly distorted).
#[derive(Debug, Clone, PartialEq)]
pub struct StereoLandmarkObservation {
    pub id: String,
    pub left: Option<Vector2>,
    pub right: Option<Vector2>,
}

impl StereoRig {
    pub fn new(
        left: Camera,
        right: Camera,
        right_from_left: Pose,
    ) -> Result<Self, PnpError> {
        let baseline = right_from_left.position.length();
        if !baseline.is_finite() || baseline < MIN_BASELINE {
            return Err(PnpError::SolverFailed);
        }
        // Rotation should be finite unit-ish; normalize if needed later.
        if !right_from_left.rotation.norm().is_finite() {
            return Err(PnpError::SolverFailed);
        }
        Ok(Self {
            left,
            right,
            right_from_left,
        })
    }

    /// Euclidean length of the left→right camera translation.
    pub fn baseline_length(&self) -> f64 {
        self.right_from_left.position.length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Quaternion, Vector3};
    use crate::Camera;

    #[test]
    fn stereo_rig_rejects_near_zero_baseline() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let right_from_left = Pose::identity(); // baseline 0
        assert!(StereoRig::new(left, right, right_from_left).is_err());
    }

    #[test]
    fn stereo_rig_accepts_horizontal_baseline() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let right_from_left = Pose::new(
            Vector3::new(0.12, 0.0, 0.0), // 12 cm baseline along +X in left frame
            Quaternion::identity(),
        );
        let rig = StereoRig::new(left, right, right_from_left).unwrap();
        assert!((rig.baseline_length() - 0.12).abs() < 1e-12);
    }
}
