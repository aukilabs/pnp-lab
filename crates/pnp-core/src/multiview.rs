//! Multi-view camera rig: N ≥ 1 calibrated views with fixed extrinsics relative
//! to a primary view.
//!
//! # Frames
//! - **`views[0]` is primary.** Pose results and joint residuals are expressed
//!   in the primary OpenCV camera frame (+Z forward).
//! - Per-view `from_primary` maps a 3D point in primary coords into that view:
//!   `X_view = R * X_primary + t`. For the primary view this is identity
//!   (forced on construction).
//! - Stereo maps as left → primary, `right_from_left` → right `from_primary`.

use crate::camera::Camera;
use crate::stereo::StereoRig;
use crate::types::{PnpError, Pose, Vector2};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// One calibrated view in a multi-view rig.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraView {
    pub camera: Camera,
    /// Maps primary OpenCV coords → this view: `X_view = R * X_primary + t`.
    /// Identity for the primary view.
    pub from_primary: Pose,
}

/// Ordered multi-view rig. `views[0]` is primary; length ≥ 1 after construction.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiViewRig {
    /// `views[0]` is primary. Length ≥ 1.
    pub views: Vec<CameraView>,
}

/// Per-landmark sparse observations across rig views (pixel coords).
///
/// `pixels.len()` must equal `rig.views.len()`. `None` means not observed in
/// that view.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiViewObservation {
    pub id: String,
    /// Length must equal `rig.views.len()`. `None` = not observed in that view.
    pub pixels: Vec<Option<Vector2>>,
}

impl MultiViewRig {
    /// Build a rig from ordered views (`views[0]` = primary).
    ///
    /// Returns [`PnpError::SolverFailed`] if `views` is empty or a non-primary
    /// view has a non-finite `from_primary` rotation. View 0's `from_primary`
    /// is **forced to identity** regardless of the value supplied.
    pub fn new(mut views: Vec<CameraView>) -> Result<Self, PnpError> {
        if views.is_empty() {
            return Err(PnpError::SolverFailed);
        }

        // Primary extrinsics are defined to be identity.
        views[0].from_primary = Pose::identity();

        for view in views.iter().skip(1) {
            if !view.from_primary.rotation.norm().is_finite() {
                return Err(PnpError::SolverFailed);
            }
            if !view.from_primary.position.x.is_finite()
                || !view.from_primary.position.y.is_finite()
                || !view.from_primary.position.z.is_finite()
            {
                return Err(PnpError::SolverFailed);
            }
        }

        Ok(Self { views })
    }

    /// Primary view (`views[0]`).
    pub fn primary(&self) -> &CameraView {
        &self.views[0]
    }

    /// Number of views in the rig.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// Two-view rig with left as primary and `right_from_left` as the secondary
    /// extrinsic. Infallible for any valid [`StereoRig`].
    pub fn from_stereo(stereo: &StereoRig) -> Self {
        MultiViewRig::new(vec![
            CameraView {
                camera: stereo.left.clone(),
                from_primary: Pose::identity(),
            },
            CameraView {
                camera: stereo.right.clone(),
                from_primary: stereo.right_from_left,
            },
        ])
        .expect("StereoRig always yields a valid two-view MultiViewRig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Quaternion, Vector3};
    use crate::{Camera, Pose, StereoRig};

    #[test]
    fn multiview_rejects_empty_views() {
        assert!(MultiViewRig::new(vec![]).is_err());
    }

    #[test]
    fn multiview_primary_from_primary_forced_to_identity() {
        // Plan prefers force-set identity for view 0 rather than hard-reject.
        let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let bad = CameraView {
            camera: cam,
            from_primary: Pose::new(Vector3::new(1.0, 0.0, 0.0), Quaternion::identity()),
        };
        let rig = MultiViewRig::new(vec![bad]).unwrap();
        assert!((rig.views[0].from_primary.position.length()).abs() < 1e-15);
        assert!((rig.primary().from_primary.rotation.w - 1.0).abs() < 1e-15);
    }

    #[test]
    fn stereo_to_multiview_has_two_views() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let stereo = StereoRig::new(
            left,
            right,
            Pose::new(Vector3::new(0.12, 0.0, 0.0), Quaternion::identity()),
        )
        .unwrap();
        let mv = stereo.to_multiview();
        assert_eq!(mv.num_views(), 2);
        assert!((mv.views[0].from_primary.position.length()).abs() < 1e-15);
        assert!((mv.views[1].from_primary.position.x - 0.12).abs() < 1e-15);
    }
}
