//! Integration tests for calibrated stereo PnP.

use pnp_core::pose_tools::{self, transform_point};
use pnp_core::rodrigues::rvec_to_rotation_matrix;
use pnp_core::types::rotation_matrix_to_quaternion;
use pnp_core::{
    estimate_square_pose_from_stereo_pixels, solve_pnp_stereo, Camera, Landmark, PnpError, Pose,
    Quaternion, SolvePnpMethod, StereoLandmarkObservation, StereoRig, Vector2, Vector3,
};

fn square_landmarks() -> Vec<Landmark> {
    vec![
        Landmark {
            id: "0".into(),
            position: Vector3::new(-0.1, -0.1, 0.0),
        },
        Landmark {
            id: "1".into(),
            position: Vector3::new(0.1, -0.1, 0.0),
        },
        Landmark {
            id: "2".into(),
            position: Vector3::new(0.1, 0.1, 0.0),
        },
        Landmark {
            id: "3".into(),
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

fn cv_pose_from_rvec_tvec(rvec: [f64; 3], t: [f64; 3]) -> Pose {
    let rot = rvec_to_rotation_matrix(&rvec);
    let q = rotation_matrix_to_quaternion(&rot);
    Pose::new(Vector3::new(t[0], t[1], t[2]), q)
}

fn transform_object_to_left(cv_pose: &Pose, pw: Vector3) -> Vector3 {
    let r_mat = cv_pose
        .rotation
        .normalize()
        .to_na_unit()
        .to_rotation_matrix();
    let r = r_mat.matrix();
    let x = r[(0, 0)] * pw.x + r[(0, 1)] * pw.y + r[(0, 2)] * pw.z + cv_pose.position.x;
    let y = r[(1, 0)] * pw.x + r[(1, 1)] * pw.y + r[(1, 2)] * pw.z + cv_pose.position.y;
    let z = r[(2, 0)] * pw.x + r[(2, 1)] * pw.y + r[(2, 2)] * pw.z + cv_pose.position.z;
    Vector3::new(x, y, z)
}

fn project_pinhole(cam: &Camera, pc: Vector3) -> Vector2 {
    Vector2::new(cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy)
}

fn project_stereo(
    landmarks: &[Landmark],
    rig: &StereoRig,
    cv_pose: &Pose,
) -> Vec<StereoLandmarkObservation> {
    landmarks
        .iter()
        .map(|lm| {
            let x_left = transform_object_to_left(cv_pose, lm.position);
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
    2.0 * dot.acos()
}

fn pose_errors(est: &Pose, gt: &Pose) -> (f64, f64) {
    let dp = Vector3::new(
        est.position.x - gt.position.x,
        est.position.y - gt.position.y,
        est.position.z - gt.position.z,
    );
    (dp.length(), quaternion_angle(&est.rotation, &gt.rotation))
}

#[test]
fn stereo_recovers_known_pose_within_tol() {
    let landmarks = square_landmarks();
    let rig = make_rig();
    // Object slightly rotated and ~1.5 m in front of left camera (OpenCV).
    let cv_true = cv_pose_from_rvec_tvec([0.05, -0.15, 0.08], [0.02, -0.01, 1.5]);
    let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
    let obs = project_stereo(&landmarks, &rig, &cv_true);

    for method in [
        SolvePnpMethod::Iterative,
        SolvePnpMethod::EPnP,
        SolvePnpMethod::SQPnP,
    ] {
        let pose = solve_pnp_stereo(&landmarks, &obs, &rig, method).expect("stereo solve");
        let (pos_err, rot_err) = pose_errors(&pose, &gl_true);
        assert!(
            pos_err < 1e-3,
            "{:?}: position error {} exceeds 1e-3",
            method,
            pos_err
        );
        assert!(
            rot_err < 1e-3,
            "{:?}: rotation error {} rad exceeds 1e-3",
            method,
            rot_err
        );
    }
}

#[test]
fn left_only_degrades_to_mono() {
    let landmarks = square_landmarks();
    let rig = make_rig();
    let cv_true = cv_pose_from_rvec_tvec([0.0, 0.2, 0.0], [0.0, 0.0, 1.2]);
    let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
    let mut obs = project_stereo(&landmarks, &rig, &cv_true);
    for o in &mut obs {
        o.right = None;
    }

    let pose =
        solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::Iterative).expect("left-only");
    let (pos_err, rot_err) = pose_errors(&pose, &gl_true);
    assert!(pos_err < 1e-3, "left-only pos err {}", pos_err);
    assert!(rot_err < 1e-3, "left-only rot err {}", rot_err);
}

#[test]
fn mismatched_ids_returns_mismatched_counts() {
    let landmarks = square_landmarks();
    let rig = make_rig();
    let obs = vec![
        StereoLandmarkObservation {
            id: "0".into(),
            left: Some(Vector2::new(100.0, 100.0)),
            right: Some(Vector2::new(90.0, 100.0)),
        },
        StereoLandmarkObservation {
            id: "1".into(),
            left: Some(Vector2::new(200.0, 100.0)),
            right: None,
        },
        StereoLandmarkObservation {
            id: "2".into(),
            left: Some(Vector2::new(200.0, 200.0)),
            right: None,
        },
        StereoLandmarkObservation {
            id: "nope".into(),
            left: Some(Vector2::new(100.0, 200.0)),
            right: None,
        },
    ];

    let err = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::EPnP).unwrap_err();
    assert_eq!(err, PnpError::MismatchedCounts);
}

#[test]
fn partial_right_views_still_recover() {
    // Two landmarks stereo, two left-only — joint residual still well-posed.
    let landmarks = square_landmarks();
    let rig = make_rig();
    let cv_true = cv_pose_from_rvec_tvec([-0.1, 0.05, 0.02], [0.01, 0.02, 1.8]);
    let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
    let mut obs = project_stereo(&landmarks, &rig, &cv_true);
    obs[2].right = None;
    obs[3].right = None;

    let pose = solve_pnp_stereo(&landmarks, &obs, &rig, SolvePnpMethod::Iterative).unwrap();
    let (pos_err, rot_err) = pose_errors(&pose, &gl_true);
    assert!(pos_err < 1e-3, "partial stereo pos err {}", pos_err);
    assert!(rot_err < 1e-3, "partial stereo rot err {}", rot_err);
}

/// Corner order for square pose: TL, TR, BR, BL in left OpenCV frame.
///
/// `right` / `up` are the marker's local +X / +Y axes expressed in left OpenCV
/// coordinates (for a frontal marker facing the camera, up ≈ (0, −1, 0)).
fn stereo_square_corners_cv(
    center: Vector3,
    right: Vector3,
    up: Vector3,
    half: f64,
) -> [Vector3; 4] {
    let scale_add =
        |a: Vector3, s: f64, b: Vector3| Vector3::new(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
    let corner = |sr: f64, su: f64| {
        let mut p = center;
        p = scale_add(p, sr * half, right);
        p = scale_add(p, su * half, up);
        p
    };
    [
        corner(-1.0, 1.0),  // TL = −right + up
        corner(1.0, 1.0),   // TR
        corner(1.0, -1.0),  // BR
        corner(-1.0, -1.0), // BL
    ]
}

fn project_corners_stereo(
    corners_left: &[Vector3; 4],
    rig: &StereoRig,
) -> ([Vector2; 4], [Vector2; 4]) {
    // Triangulation treats `right_from_left` as the right camera's pose in the left
    // frame: X_left = R * X_right + t, so X_right = Rᵀ (X_left − t).
    // (Stereo PnP residuals use the inverse convention for the same field; fixtures
    // for triangulation / square stereo must match triangulate_midpoint.)
    let left_from_right_inv = pose_tools::invert_pose(&rig.right_from_left);
    let mut left = [Vector2::new(0.0, 0.0); 4];
    let mut right = [Vector2::new(0.0, 0.0); 4];
    for i in 0..4 {
        left[i] = project_pinhole(&rig.left, corners_left[i]);
        let x_right = transform_point(&left_from_right_inv, corners_left[i]);
        right[i] = project_pinhole(&rig.right, x_right);
    }
    (left, right)
}

fn rotate_vector(q: &Quaternion, v: Vector3) -> Vector3 {
    let vector_quat = Quaternion::new(v.x, v.y, v.z, 0.0);
    let rotated = q.multiply(&vector_quat).multiply(&q.conjugate());
    Vector3::new(rotated.x, rotated.y, rotated.z)
}

fn normalize_v(v: Vector3) -> Vector3 {
    let len = v.length();
    Vector3::new(v.x / len, v.y / len, v.z / len)
}

#[test]
fn stereo_square_pose_recovers_frontal_marker() {
    let physical_size = 0.2;
    let half = physical_size * 0.5;
    let rig = make_rig();

    // Frontal square in left OpenCV frame: Z=1.5 m, centered near principal ray.
    let center_cv = Vector3::new(0.02, -0.01, 1.5);
    let right_cv = Vector3::new(1.0, 0.0, 0.0);
    // "Up" on the marker face in image sense is -Y in OpenCV when facing camera.
    let up_cv = Vector3::new(0.0, -1.0, 0.0);
    let corners = stereo_square_corners_cv(center_cv, right_cv, up_cv, half);
    let (left_px, right_px) = project_corners_stereo(&corners, &rig);

    let estimate =
        estimate_square_pose_from_stereo_pixels(left_px, right_px, physical_size, &rig).unwrap();

    let gl_true_pos = Vector3::new(center_cv.x, -center_cv.y, -center_cv.z);
    let pos_err = Vector3::new(
        estimate.pose.position.x - gl_true_pos.x,
        estimate.pose.position.y - gl_true_pos.y,
        estimate.pose.position.z - gl_true_pos.z,
    )
    .length();
    assert!(
        pos_err < 1e-4,
        "position error {} (est {:?} expected {:?})",
        pos_err,
        estimate.pose.position,
        gl_true_pos
    );
    assert!(
        estimate.confidence > 0.95,
        "confidence was {}",
        estimate.confidence
    );
    assert!(
        estimate.normalized_corner_error < 1e-6,
        "normalized corner error was {}",
        estimate.normalized_corner_error
    );

    // Local +X should map to OpenGL right ≈ (+1, 0, 0); +Y to OpenGL up ≈ (0, +1, 0).
    let est_right = normalize_v(rotate_vector(
        &estimate.pose.rotation,
        Vector3::new(1.0, 0.0, 0.0),
    ));
    let est_up = normalize_v(rotate_vector(
        &estimate.pose.rotation,
        Vector3::new(0.0, 1.0, 0.0),
    ));
    assert!(
        est_right.x > 0.99 && est_right.y.abs() < 1e-3 && est_right.z.abs() < 1e-3,
        "estimated right {:?}",
        est_right
    );
    assert!(
        est_up.y > 0.99 && est_up.x.abs() < 1e-3 && est_up.z.abs() < 1e-3,
        "estimated up {:?}",
        est_up
    );

    for i in 0..4 {
        let expected_d = corners[i].length();
        assert!(
            (estimate.ray_distances[i] - expected_d).abs() < 1e-4,
            "ray_distance[{}] = {} expected {}",
            i,
            estimate.ray_distances[i],
            expected_d
        );
    }
}

#[test]
fn stereo_square_pose_recovers_tilted_marker() {
    let physical_size = 0.16;
    let half = physical_size * 0.5;
    let rig = make_rig();

    // Mild yaw so the square stays in front of both cameras with clear disparity.
    let yaw = 0.18_f64;
    let right = normalize_v(Vector3::new(yaw.cos(), 0.0, -yaw.sin()));
    // Marker +Y (up) ≈ image top = −Y OpenCV, slightly tilted.
    let up = normalize_v(Vector3::new(0.05, -1.0, 0.08));
    // Re-orthogonalize up against right.
    let dot_ru = right.x * up.x + right.y * up.y + right.z * up.z;
    let up = normalize_v(Vector3::new(
        up.x - dot_ru * right.x,
        up.y - dot_ru * right.y,
        up.z - dot_ru * right.z,
    ));
    let center_cv = Vector3::new(0.03, 0.02, 1.7);
    let corners = stereo_square_corners_cv(center_cv, right, up, half);
    let (left_px, right_px) = project_corners_stereo(&corners, &rig);

    let estimate =
        estimate_square_pose_from_stereo_pixels(left_px, right_px, physical_size, &rig).unwrap();

    let gl_true_pos = Vector3::new(center_cv.x, -center_cv.y, -center_cv.z);
    let pos_err = Vector3::new(
        estimate.pose.position.x - gl_true_pos.x,
        estimate.pose.position.y - gl_true_pos.y,
        estimate.pose.position.z - gl_true_pos.z,
    )
    .length();
    assert!(pos_err < 1e-3, "tilted pos err {}", pos_err);
    assert!(
        estimate.confidence > 0.9,
        "confidence {}",
        estimate.confidence
    );
    assert!(
        estimate.normalized_corner_error < 1e-4,
        "corner error {}",
        estimate.normalized_corner_error
    );
}

#[test]
fn stereo_square_pose_rejects_bad_size() {
    let rig = make_rig();
    let px = [Vector2::new(100.0, 100.0); 4];
    assert_eq!(
        estimate_square_pose_from_stereo_pixels(px, px, 0.0, &rig),
        Err(PnpError::SolverFailed)
    );
}
