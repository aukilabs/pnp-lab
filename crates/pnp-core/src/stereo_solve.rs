//! Joint stereo Perspective-n-Point as a thin N=2 multiview wrapper.
//!
//! # Pipeline
//! 1. Convert [`StereoRig`] → [`crate::multiview::MultiViewRig`] (left = primary).
//! 2. Convert [`StereoLandmarkObservation`]s → [`MultiViewObservation`]s.
//! 3. Delegate to [`solve_pnp_multiview`] / [`solve_pnp_multiview_camera_pose`].
//!
//! # Frames
//! - Public results: OpenGL object pose (primary view = left), same as mono.
//! - `rig.right_from_left` is treated as OpenCV extrinsics
//!   (`X_right = R_rl * X_left + t_rl`), identical to multiview `from_primary`.

use crate::multiview::MultiViewObservation;
use crate::multiview_solve::{solve_pnp_multiview, solve_pnp_multiview_camera_pose};
use crate::stereo::{StereoLandmarkObservation, StereoRig};
use crate::types::{Landmark, PnpError, Pose, SolvePnpMethod};
use alloc::vec;
use alloc::vec::Vec;

/// Convert stereo observations to multiview form: pixels = `[left, right]`.
fn stereo_obs_to_multiview(
    observations: &[StereoLandmarkObservation],
) -> Vec<MultiViewObservation> {
    observations
        .iter()
        .map(|o| MultiViewObservation {
            id: o.id.clone(),
            pixels: vec![o.left, o.right],
        })
        .collect()
}

/// Estimate the **object** pose from stereo landmark observations.
///
/// # Inputs
/// - `landmarks` / `observations`: matched by string `id` (same length required)
/// - `rig`: calibrated stereo pair; `right_from_left` in **OpenCV** convention
/// - `method`: monocular solver used for the seed (left preferred; right if
///   left is sparse and right has enough points)
///
/// Missing left or right pixels are skipped in the joint residual. The seed
/// prefers the left view when it has enough points for `method`; otherwise
/// seeds from the right and transports the pose into the left (primary) frame.
///
/// # Returns
/// Object pose in **OpenGL** (primary view = left camera), same meaning as
/// [`crate::solve::solve_pnp`].
///
/// # Errors
/// - [`PnpError::MismatchedCounts`] — length or id mismatch
/// - [`PnpError::InsufficientPoints`] — too few projections / seed points
/// - [`PnpError::SolverFailed`] — seed or numerical failure
///
/// Implemented as a thin wrapper over [`solve_pnp_multiview`] (N=2).
pub fn solve_pnp_stereo(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let mv_rig = rig.to_multiview();
    let mv_obs = stereo_obs_to_multiview(observations);
    solve_pnp_multiview(landmarks, &mv_obs, &mv_rig, method)
}

/// Stereo object pose inverted to camera pose (same convention as mono).
///
/// Equivalent to [`solve_pnp_stereo`] followed by
/// [`crate::solve::camera_pose_from_solve_pnp_pose`].
///
/// Implemented as a thin wrapper over [`solve_pnp_multiview_camera_pose`] (N=2).
pub fn solve_pnp_stereo_camera_pose(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let mv_rig = rig.to_multiview();
    let mv_obs = stereo_obs_to_multiview(observations);
    solve_pnp_multiview_camera_pose(landmarks, &mv_obs, &mv_rig, method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::pose_tools::{self, transform_point};
    use crate::rodrigues;
    use crate::types::{rotation_matrix_to_quaternion, Matrix3x3, Quaternion, Vector2, Vector3};
    use alloc::string::ToString;
    use nalgebra::Vector3 as NaVector3;

    fn square_landmarks() -> Vec<Landmark> {
        vec![
            Landmark {
                id: "0".to_string(),
                position: Vector3::new(-0.1, -0.1, 0.0),
            },
            Landmark {
                id: "1".to_string(),
                position: Vector3::new(0.1, -0.1, 0.0),
            },
            Landmark {
                id: "2".to_string(),
                position: Vector3::new(0.1, 0.1, 0.0),
            },
            Landmark {
                id: "3".to_string(),
                position: Vector3::new(-0.1, 0.1, 0.0),
            },
        ]
    }

    fn make_rig() -> StereoRig {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        StereoRig::new(
            left,
            right,
            Pose::new(Vector3::new(0.12, 0.0, 0.0), Quaternion::identity()),
        )
        .unwrap()
    }

    fn pose_to_rvec_tvec(pose: &Pose) -> ([f64; 3], NaVector3<f64>) {
        let r_mat = pose.rotation.normalize().to_na_unit().to_rotation_matrix();
        let m = Matrix3x3::from_na(r_mat.matrix());
        let rvec = rodrigues::rotation_matrix_to_rvec(&m);
        let tvec = NaVector3::new(pose.position.x, pose.position.y, pose.position.z);
        (rvec, tvec)
    }

    fn rvec_tvec_to_pose(rvec: &[f64; 3], tvec: &NaVector3<f64>) -> Pose {
        let rot_m = rodrigues::rvec_to_rotation_matrix(rvec);
        let q = rotation_matrix_to_quaternion(&rot_m);
        Pose::new(Vector3::new(tvec.x, tvec.y, tvec.z), q)
    }

    fn project_pinhole(cam: &Camera, pc: Vector3) -> Vector2 {
        Vector2::new(cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy)
    }

    fn transform_object_to_left(rvec: &[f64; 3], tvec: &NaVector3<f64>, pw: Vector3) -> Vector3 {
        let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
        let p = r * NaVector3::new(pw.x, pw.y, pw.z) + tvec;
        Vector3::new(p.x, p.y, p.z)
    }

    /// Known OpenCV object-in-left: small yaw + translation in front of camera.
    fn true_cv_pose() -> Pose {
        let rvec = [0.05, -0.15, 0.08];
        let t = NaVector3::new(0.02, -0.01, 1.5);
        rvec_tvec_to_pose(&rvec, &t)
    }

    fn project_stereo_obs(
        landmarks: &[Landmark],
        rig: &StereoRig,
        cv_pose: &Pose,
    ) -> Vec<StereoLandmarkObservation> {
        let (rvec, tvec) = pose_to_rvec_tvec(cv_pose);
        landmarks
            .iter()
            .map(|lm| {
                let x_left = transform_object_to_left(&rvec, &tvec, lm.position);
                let left = project_pinhole(&rig.left, x_left);
                let x_right = transform_point(&rig.right_from_left, x_left);
                let right = project_pinhole(&rig.right, x_right);
                StereoLandmarkObservation {
                    id: lm.id.clone(),
                    left: Some(left),
                    right: Some(right),
                }
            })
            .collect()
    }

    fn quaternion_angle(a: &Quaternion, b: &Quaternion) -> f64 {
        let dot = (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
            .abs()
            .min(1.0);
        2.0 * libm::acos(dot)
    }

    #[test]
    fn recovers_known_stereo_pose() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let cv_true = true_cv_pose();
        let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
        let obs = project_stereo_obs(&landmarks, &rig, &cv_true);

        let pose = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::Iterative).unwrap();

        let dp = Vector3::new(
            pose.position.x - gl_true.position.x,
            pose.position.y - gl_true.position.y,
            pose.position.z - gl_true.position.z,
        );
        let pos_err = dp.length();
        let rot_err = quaternion_angle(&pose.rotation, &gl_true.rotation);
        assert!(pos_err < 1e-3, "position error {}", pos_err);
        assert!(rot_err < 1e-3, "rotation error {} rad", rot_err);
    }

    #[test]
    fn left_only_still_works() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let cv_true = true_cv_pose();
        let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
        let mut obs = project_stereo_obs(&landmarks, &rig, &cv_true);
        for o in &mut obs {
            o.right = None;
        }

        let pose = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::Iterative).unwrap();

        let dp = Vector3::new(
            pose.position.x - gl_true.position.x,
            pose.position.y - gl_true.position.y,
            pose.position.z - gl_true.position.z,
        );
        assert!(dp.length() < 1e-3);
        assert!(quaternion_angle(&pose.rotation, &gl_true.rotation) < 1e-3);
    }

    #[test]
    fn mismatched_ids_error() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let obs = vec![
            StereoLandmarkObservation {
                id: "0".to_string(),
                left: Some(Vector2::new(100.0, 100.0)),
                right: Some(Vector2::new(90.0, 100.0)),
            },
            StereoLandmarkObservation {
                id: "1".to_string(),
                left: Some(Vector2::new(200.0, 100.0)),
                right: None,
            },
            StereoLandmarkObservation {
                id: "2".to_string(),
                left: Some(Vector2::new(200.0, 200.0)),
                right: None,
            },
            StereoLandmarkObservation {
                id: "99".to_string(),
                left: Some(Vector2::new(100.0, 200.0)),
                right: None,
            },
        ];
        let err = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::EPnP).unwrap_err();
        assert_eq!(err, PnpError::MismatchedCounts);
    }

    #[test]
    fn mismatched_counts_error() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let obs = vec![StereoLandmarkObservation {
            id: "0".to_string(),
            left: Some(Vector2::new(1.0, 1.0)),
            right: None,
        }];
        let err = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::EPnP).unwrap_err();
        assert_eq!(err, PnpError::MismatchedCounts);
    }

    #[test]
    fn camera_pose_is_inverse() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let cv_true = true_cv_pose();
        let obs = project_stereo_obs(&landmarks, &rig, &cv_true);

        let obj = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::SQPnP).unwrap();
        let cam =
            solve_pnp_stereo_camera_pose(&landmarks, &obs, &rig, SolvePnpMethod::SQPnP).unwrap();
        let back = pose_tools::invert_pose(&cam);

        let dp = Vector3::new(
            obj.position.x - back.position.x,
            obj.position.y - back.position.y,
            obj.position.z - back.position.z,
        );
        assert!(dp.length() < 1e-9);
        assert!(quaternion_angle(&obj.rotation, &back.rotation) < 1e-9);
    }
}
