//! Integration tests for calibrated stereo PnP.

use pnp_core::pose_tools::{self, transform_point};
use pnp_core::rodrigues::rvec_to_rotation_matrix;
use pnp_core::types::rotation_matrix_to_quaternion;
use pnp_core::{
    solve_pnp_stereo, Camera, Landmark, PnpError, Pose, Quaternion, SolvePnpMethod,
    StereoLandmarkObservation, StereoRig, Vector2, Vector3,
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
