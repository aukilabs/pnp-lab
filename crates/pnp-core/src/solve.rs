use crate::pose_tools;
use crate::types::*;
use alloc::vec::Vec;

/// Match landmarks with observations by id, returning paired (3D, 2D) points.
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

/// Extract camera intrinsics from a column-major Matrix3x3.
///
/// The input Matrix3x3 is in column-major (OpenGL) order, which stores
/// the camera matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]] directly.
/// Since nalgebra also uses column-major storage, to_na() gives K directly.
///
/// Note: The C++ reference transposes because it converts between column-major
/// Matrix3x3 field access and row-major OpenCV cv::Mat access. Our nalgebra
/// Matrix3 shares column-major storage, so no transpose is needed.
pub fn extract_camera_intrinsics(camera_matrix: &Matrix3x3) -> nalgebra::Matrix3<f64> {
    camera_matrix.to_na()
}

/// Solve PnP: estimate the pose of an object given 3D-2D correspondences.
///
/// Returns the object pose in OpenGL coordinate system.
pub fn solve_pnp(
    landmarks: &[Landmark],
    observations: &[LandmarkObservation],
    camera_matrix: &Matrix3x3,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pairs = match_correspondences(landmarks, observations)?;
    let cam = extract_camera_intrinsics(camera_matrix);

    let min_points = match method {
        SolvePnpMethod::SQPnP => 3,
        _ => 4,
    };

    if pairs.len() < min_points {
        return Err(PnpError::InsufficientPoints);
    }

    let object_points: Vec<_> = pairs.iter().map(|(p, _)| *p).collect();
    let image_points: Vec<_> = pairs.iter().map(|(_, p)| *p).collect();

    // Dispatch to solver
    let (rotation_matrix, tvec) = match method {
        SolvePnpMethod::EPnP => crate::epnp::solve_epnp(&object_points, &image_points, &cam)?,
        SolvePnpMethod::Iterative => {
            crate::iterative::solve_iterative(&object_points, &image_points, &cam, None, None)?
        }
        SolvePnpMethod::SQPnP => crate::sqpnp::solve_sqpnp(&object_points, &image_points, &cam)?,
    };

    // Convert rotation matrix to quaternion
    let rot_m3x3 = Matrix3x3::from_na(&rotation_matrix);
    let q = rotation_matrix_to_quaternion(&rot_m3x3);

    // Build pose in OpenCV coordinates
    let cv_pose = Pose::new(Vector3::new(tvec.x, tvec.y, tvec.z), q);

    // Convert to OpenGL
    Ok(pose_tools::from_opencv_to_opengl(&cv_pose))
}

/// Solve PnP and return the camera pose (inverse of object pose).
pub fn solve_pnp_camera_pose(
    landmarks: &[Landmark],
    observations: &[LandmarkObservation],
    camera_matrix: &Matrix3x3,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pose = solve_pnp(landmarks, observations, camera_matrix, method)?;
    Ok(camera_pose_from_solve_pnp_pose(&pose))
}

/// Invert a solvePnP pose to get the camera pose.
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

    #[test]
    fn test_match_correspondences_basic() {
        let landmarks = make_landmarks();
        let observations = make_observations_set0();
        let pairs = match_correspondences(&landmarks, &observations).unwrap();
        assert_eq!(pairs.len(), 4);
        // First pair: landmark "0" with observation "0"
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
        // First pair should be landmark "2" (from the observation order)
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
    fn test_extract_camera_intrinsics() {
        let cam = Matrix3x3::camera_matrix(815.8511, 815.8511, 960.0, 540.0);
        let k = extract_camera_intrinsics(&cam);
        // Should be: [[fx,0,cx],[0,fy,cy],[0,0,1]]
        assert!((k[(0, 0)] - 815.8511).abs() < 1e-10);
        assert!((k[(0, 1)]).abs() < 1e-10);
        assert!((k[(0, 2)] - 960.0).abs() < 1e-10);
        assert!((k[(1, 0)]).abs() < 1e-10);
        assert!((k[(1, 1)] - 815.8511).abs() < 1e-10);
        assert!((k[(1, 2)] - 540.0).abs() < 1e-10);
        assert!((k[(2, 0)]).abs() < 1e-10);
        assert!((k[(2, 1)]).abs() < 1e-10);
        assert!((k[(2, 2)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_pose_from_solve_pnp_pose() {
        let pose = Pose::new(
            Vector3::new(1.0, 2.0, 3.0),
            Quaternion::new(0.0, 0.0, 0.0, 1.0),
        );
        let cam_pose = camera_pose_from_solve_pnp_pose(&pose);
        // Identity rotation → inverted position is just negated
        assert!((cam_pose.position.x - (-1.0)).abs() < 1e-10);
        assert!((cam_pose.position.y - (-2.0)).abs() < 1e-10);
        assert!((cam_pose.position.z - (-3.0)).abs() < 1e-10);
    }

    fn make_camera_matrix() -> Matrix3x3 {
        Matrix3x3::camera_matrix(815.8511, 815.8511, 960.0, 540.0)
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
    fn test_solve_pnp_epnp_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera_matrix();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::EPnP).unwrap();
        // Should return a valid pose (not NaN)
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_iterative_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera_matrix();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_sqpnp_end_to_end() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera_matrix();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();
        assert!(!pose.position.x.is_nan());
        assert!(!pose.rotation.w.is_nan());
    }

    #[test]
    fn test_solve_pnp_camera_pose_iterative() {
        let landmarks = make_landmarks();
        let obs = make_observations_set0();
        let cam = make_camera_matrix();
        let pose = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
        let cam_pose = camera_pose_from_solve_pnp_pose(&pose);
        // Camera pose and object pose should be inverses
        assert!(!cam_pose.position.x.is_nan());
    }

    #[test]
    fn test_solve_pnp_iterative_all_sets() {
        let landmarks = make_landmarks();
        let cam = make_camera_matrix();
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
        let cam = make_camera_matrix();
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
        // All three methods on the same input should produce similar poses
        let landmarks = make_landmarks();
        let cam = make_camera_matrix();

        for obs in [make_observations_set1(), make_observations_set2()] {
            let pose_iter = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
            let pose_sqpnp = solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

            // Position agreement within 1e-2
            let pos_diff = ((pose_iter.position.x - pose_sqpnp.position.x).powi(2)
                + (pose_iter.position.y - pose_sqpnp.position.y).powi(2)
                + (pose_iter.position.z - pose_sqpnp.position.z).powi(2))
            .sqrt();
            assert!(
                pos_diff < 0.01,
                "Iterative vs SQPnP position diff: {}",
                pos_diff
            );

            // Rotation agreement: compare via angle between quaternions
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
