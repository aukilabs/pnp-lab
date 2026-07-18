//! Top-level PnP API: correspondence matching, method dispatch, and pose frames.
//!
//! Callers typically use [`solve_pnp`] or [`solve_pnp_camera_pose`]. Image
//! observations are **distorted pixels**; [`Camera`] undistorts them before
//! the algebraic solvers run on an ideal pinhole model.

use crate::camera::Camera;
use crate::pose_tools;
use crate::types::*;
use alloc::vec::Vec;

/// Match landmarks with observations by shared string `id`.
///
/// Returns pairs in **observation order**: `(landmark.position, observation.position)`.
///
/// # Errors
///
/// - [`PnpError::MismatchedCounts`] if lengths differ or an observation id is
///   missing from `landmarks`
/// - [`PnpError::InsufficientPoints`] if fewer than three pairs remain
pub fn match_correspondences(
    landmarks: &[Landmark],
    observations: &[LandmarkObservation],
) -> Result<Vec<(Vector3, Vector2)>, PnpError> {
    if landmarks.len() != observations.len() {
        return Err(PnpError::MismatchedCounts);
    }

    let mut pairs = Vec::with_capacity(landmarks.len());

    for obs in observations {
        let landmark = landmarks.iter().find(|l| l.id == obs.id);
        match landmark {
            Some(l) => pairs.push((l.position, obs.position)),
            None => return Err(PnpError::MismatchedCounts),
        }
    }

    if pairs.len() < 3 {
        return Err(PnpError::InsufficientPoints);
    }

    Ok(pairs)
}

/// Estimate the **object** pose from 3D–2D correspondences.
///
/// # Inputs
///
/// - `landmarks` / `observations`: matched by `id` (see [`match_correspondences`])
/// - `camera`: monocular intrinsics; optional OpenCV distortion on image points
/// - `method`: [`SolvePnpMethod::EPnP`], [`SolvePnpMethod::Iterative`], or
///   [`SolvePnpMethod::SQPnP`] (minimum 4 points, or 3 for SQPnP)
///
/// Image observations are **distorted pixels** (OpenCV image convention). When
/// `camera.dist` is non-empty, points are undistorted before the chosen solver
/// runs with an ideal pinhole `K`.
///
/// # Returns
///
/// Object pose in **OpenGL** coordinates (Y-up, Z-backward). Internally the
/// solvers use the OpenCV camera frame; the result is converted via
/// [`pose_tools::from_opencv_to_opengl`].
///
/// For camera-from-world pose, use [`solve_pnp_camera_pose`] or invert with
/// [`camera_pose_from_solve_pnp_pose`].
pub fn solve_pnp(
    landmarks: &[Landmark],
    observations: &[LandmarkObservation],
    camera: &Camera,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pairs = match_correspondences(landmarks, observations)?;
    let cam = camera.matrix_na();

    let min_points = match method {
        SolvePnpMethod::SQPnP => 3,
        _ => 4,
    };

    if pairs.len() < min_points {
        return Err(PnpError::InsufficientPoints);
    }

    let object_points: Vec<_> = pairs.iter().map(|(p, _)| *p).collect();
    // Undistort observations so EPnP / iterative / SQPnP can use pinhole math.
    let image_points: Vec<_> = pairs
        .iter()
        .map(|(_, p)| camera.undistort_pixel(*p))
        .collect();

    let (rotation_matrix, tvec) = match method {
        SolvePnpMethod::EPnP => crate::epnp::solve_epnp(&object_points, &image_points, &cam)?,
        SolvePnpMethod::Iterative => {
            crate::iterative::solve_iterative(&object_points, &image_points, &cam, None, None)?
        }
        SolvePnpMethod::SQPnP => crate::sqpnp::solve_sqpnp(&object_points, &image_points, &cam)?,
    };

    let rot_m3x3 = Matrix3x3::from_na(&rotation_matrix);
    let q = rotation_matrix_to_quaternion(&rot_m3x3);
    let cv_pose = Pose::new(Vector3::new(tvec.x, tvec.y, tvec.z), q);
    Ok(pose_tools::from_opencv_to_opengl(&cv_pose))
}

/// Estimate the **camera** pose (world-from-camera inverse of the object pose).
///
/// Equivalent to [`solve_pnp`] followed by [`camera_pose_from_solve_pnp_pose`].
/// Same input conventions as [`solve_pnp`].
pub fn solve_pnp_camera_pose(
    landmarks: &[Landmark],
    observations: &[LandmarkObservation],
    camera: &Camera,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pose = solve_pnp(landmarks, observations, camera, method)?;
    Ok(camera_pose_from_solve_pnp_pose(&pose))
}

/// Invert a rigid object pose to obtain the camera pose (and vice versa).
///
/// Given object-in-camera `T = (R, t)`, returns `T^{-1} = (Rᵀ, -Rᵀ t)`.
pub fn camera_pose_from_solve_pnp_pose(solve_pnp_pose: &Pose) -> Pose {
    pose_tools::invert_pose(solve_pnp_pose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn make_landmarks() -> Vec<Landmark> {
        vec![
            Landmark {
                id: "0".to_string(),
                position: Vector3::new(-0.15, -0.15, 0.0),
            },
            Landmark {
                id: "1".to_string(),
                position: Vector3::new(0.15, -0.15, 0.0),
            },
            Landmark {
                id: "2".to_string(),
                position: Vector3::new(0.15, 0.15, 0.0),
            },
            Landmark {
                id: "3".to_string(),
                position: Vector3::new(-0.15, 0.15, 0.0),
            },
        ]
    }

    fn make_observations_set0() -> Vec<LandmarkObservation> {
        vec![
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(849.3577, 461.7641),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(1070.642, 461.7641),
            },
            LandmarkObservation {
                id: "2".to_string(),
                position: Vector2::new(1096.898, 636.8014),
            },
            LandmarkObservation {
                id: "3".to_string(),
                position: Vector2::new(823.1021, 636.8014),
            },
        ]
    }

    fn make_camera() -> Camera {
        Camera::pinhole(815.8511, 815.8511, 960.0, 540.0).unwrap()
    }

    fn make_observations_set1() -> Vec<LandmarkObservation> {
        vec![
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(1324.333, 208.0732),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(1604.393, 129.3065),
            },
            LandmarkObservation {
                id: "2".to_string(),
                position: Vector2::new(1604.393, 403.1022),
            },
            LandmarkObservation {
                id: "3".to_string(),
                position: Vector2::new(1324.333, 429.3577),
            },
        ]
    }

    fn make_observations_set2() -> Vec<LandmarkObservation> {
        vec![
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(379.0064, 743.9628),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(552.0745, 570.8946),
            },
            LandmarkObservation {
                id: "2".to_string(),
                position: Vector2::new(725.1426, 743.9628),
            },
            LandmarkObservation {
                id: "3".to_string(),
                position: Vector2::new(552.0745, 917.0309),
            },
        ]
    }

    #[test]
    fn test_match_correspondences_basic() {
        let landmarks = make_landmarks();
        let observations = make_observations_set0();
        let pairs = match_correspondences(&landmarks, &observations).unwrap();
        assert_eq!(pairs.len(), 4);
        assert!((pairs[0].0.x - (-0.15)).abs() < 1e-10);
        assert!((pairs[0].1.x - 849.3577).abs() < 1e-10);
    }

    #[test]
    fn test_match_correspondences_reordered() {
        let landmarks = make_landmarks();
        let observations = vec![
            LandmarkObservation {
                id: "2".to_string(),
                position: Vector2::new(1096.898, 636.8014),
            },
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(849.3577, 461.7641),
            },
            LandmarkObservation {
                id: "3".to_string(),
                position: Vector2::new(823.1021, 636.8014),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(1070.642, 461.7641),
            },
        ];
        let pairs = match_correspondences(&landmarks, &observations).unwrap();
        assert_eq!(pairs.len(), 4);
        assert!((pairs[0].0.x - 0.15).abs() < 1e-10);
        assert!((pairs[0].0.y - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_match_correspondences_missing() {
        let landmarks = make_landmarks();
        let observations = vec![
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(0.0, 0.0),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(0.0, 0.0),
            },
            LandmarkObservation {
                id: "2".to_string(),
                position: Vector2::new(0.0, 0.0),
            },
            LandmarkObservation {
                id: "99".to_string(),
                position: Vector2::new(0.0, 0.0),
            },
        ];
        let result = match_correspondences(&landmarks, &observations);
        assert_eq!(result.unwrap_err(), PnpError::MismatchedCounts);
    }

    #[test]
    fn test_match_correspondences_insufficient() {
        let landmarks = vec![
            Landmark {
                id: "0".to_string(),
                position: Vector3::new(0.0, 0.0, 0.0),
            },
            Landmark {
                id: "1".to_string(),
                position: Vector3::new(1.0, 0.0, 0.0),
            },
        ];
        let observations = vec![
            LandmarkObservation {
                id: "0".to_string(),
                position: Vector2::new(0.0, 0.0),
            },
            LandmarkObservation {
                id: "1".to_string(),
                position: Vector2::new(1.0, 0.0),
            },
        ];
        let result = match_correspondences(&landmarks, &observations);
        assert_eq!(result.unwrap_err(), PnpError::InsufficientPoints);
    }

    #[test]
    fn test_camera_matrix_na() {
        let cam = make_camera();
        let k = cam.matrix_na();
        assert!((k[(0, 0)] - 815.8511).abs() < 1e-10);
        assert!((k[(0, 2)] - 960.0).abs() < 1e-10);
        assert!((k[(1, 1)] - 815.8511).abs() < 1e-10);
        assert!((k[(1, 2)] - 540.0).abs() < 1e-10);
        assert!((k[(2, 2)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_pose_from_solve_pnp_pose() {
        let pose = Pose::new(
            Vector3::new(1.0, 2.0, 3.0),
            Quaternion::new(0.0, 0.0, 0.0, 1.0),
        );
        let cam_pose = camera_pose_from_solve_pnp_pose(&pose);
        assert!((cam_pose.position.x - (-1.0)).abs() < 1e-10);
        assert!((cam_pose.position.y - (-2.0)).abs() < 1e-10);
        assert!((cam_pose.position.z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solve_pnp_epnp_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::EPnP).unwrap();
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_iterative_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_sqpnp_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_camera_pose_iterative() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
        let cam_pose = camera_pose_from_solve_pnp_pose(&pose);
        assert!(!cam_pose.position.x.is_nan());
    }

    #[test]
    fn test_solve_pnp_with_distortion_still_succeeds() {
        // Synthetic: project with distortion, then solve with matching camera.
        // Even with modest distortion, undistort-then-solve should succeed.
        let cam = Camera::new(
            815.8511,
            815.8511,
            960.0,
            540.0,
            &[0.05, -0.02, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let landmarks = make_landmarks();
        // Use the known good observations as approximately-undistorted;
        // applying mild distortion then solving with the same camera should work.
        let ideal_obs = make_observations_set0();
        let mut distorted_obs = Vec::new();
        for o in &ideal_obs {
            let xn = (o.position.x - cam.cx) / cam.fx;
            let yn = (o.position.y - cam.cy) / cam.fy;
            // Manually distort via project of a unit-depth point
            let pc = Vector3::new(xn, yn, 1.0);
            let distorted = cam.project(pc).unwrap();
            distorted_obs.push(LandmarkObservation {
                id: o.id.clone(),
                position: distorted,
            });
        }
        let pose = solve_pnp(&landmarks, &distorted_obs, &cam, SolvePnpMethod::Iterative).unwrap();
        assert!(pose.position.x.is_finite());
        assert!(pose.rotation.w.is_finite());
    }

    #[test]
    fn test_solve_pnp_iterative_all_sets() {
        let landmarks = make_landmarks();
        let cam = make_camera();
        for obs in [
            make_observations_set0(),
            make_observations_set1(),
            make_observations_set2(),
        ] {
            let result = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative);
            assert!(result.is_ok(), "Iterative should succeed");
        }
    }

    #[test]
    fn test_solve_pnp_sqpnp_all_sets() {
        let landmarks = make_landmarks();
        let cam = make_camera();
        for obs in [
            make_observations_set0(),
            make_observations_set1(),
            make_observations_set2(),
        ] {
            let result = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP);
            assert!(result.is_ok(), "SQPnP should succeed");
        }
    }

    #[test]
    fn test_all_methods_agree() {
        let landmarks = make_landmarks();
        let cam = make_camera();

        for obs in [make_observations_set1(), make_observations_set2()] {
            let pose_iter = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
            let pose_sqpnp = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

            let pos_diff = ((pose_iter.position.x - pose_sqpnp.position.x).powi(2)
                + (pose_iter.position.y - pose_sqpnp.position.y).powi(2)
                + (pose_iter.position.z - pose_sqpnp.position.z).powi(2))
            .sqrt();
            assert!(
                pos_diff < 0.01,
                "Iterative vs SQPnP position diff: {}",
                pos_diff
            );

            let q1 = pose_iter.rotation;
            let q2 = pose_sqpnp.rotation;
            let dot = (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w).abs();
            let angle_diff = 2.0 * libm::acos(dot.min(1.0));
            assert!(
                angle_diff < 0.002,
                "Iterative vs SQPnP rotation diff: {} rad",
                angle_diff
            );
        }
    }
}
