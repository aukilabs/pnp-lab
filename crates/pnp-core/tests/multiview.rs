//! Integration tests for multi-view PnP (N ≥ 1).

use pnp_core::pose_tools::{self, transform_point};
use pnp_core::rodrigues::rvec_to_rotation_matrix;
use pnp_core::types::rotation_matrix_to_quaternion;
use pnp_core::{
    solve_pnp, solve_pnp_multiview, Camera, CameraView, Landmark, LandmarkObservation,
    MultiViewObservation, MultiViewRig, Pose, Quaternion, SolvePnpMethod, Vector2, Vector3,
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

fn make_n1_rig(camera: &Camera) -> MultiViewRig {
    MultiViewRig::new(vec![CameraView {
        camera: camera.clone(),
        from_primary: Pose::identity(),
    }])
    .unwrap()
}

fn cv_pose_from_rvec_tvec(rvec: [f64; 3], t: [f64; 3]) -> Pose {
    let rot = rvec_to_rotation_matrix(&rvec);
    let q = rotation_matrix_to_quaternion(&rot);
    Pose::new(Vector3::new(t[0], t[1], t[2]), q)
}

fn transform_object_to_primary(cv_pose: &Pose, pw: Vector3) -> Vector3 {
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

fn project_mono(landmarks: &[Landmark], cam: &Camera, cv_pose: &Pose) -> Vec<LandmarkObservation> {
    landmarks
        .iter()
        .map(|lm| {
            let x = transform_object_to_primary(cv_pose, lm.position);
            LandmarkObservation {
                id: lm.id.clone(),
                position: project_pinhole(cam, x),
            }
        })
        .collect()
}

fn project_multiview(
    landmarks: &[Landmark],
    rig: &MultiViewRig,
    cv_pose: &Pose,
) -> Vec<MultiViewObservation> {
    landmarks
        .iter()
        .map(|lm| {
            let x_primary = transform_object_to_primary(cv_pose, lm.position);
            let mut pixels = Vec::with_capacity(rig.num_views());
            for (c, view) in rig.views.iter().enumerate() {
                let x_c = if c == 0 {
                    x_primary
                } else {
                    transform_point(&view.from_primary, x_primary)
                };
                pixels.push(Some(project_pinhole(&view.camera, x_c)));
            }
            MultiViewObservation {
                id: lm.id.clone(),
                pixels,
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

/// N=1 multiview must match monocular `solve_pnp` on the same synthetic set.
#[test]
fn multiview_n1_matches_solve_pnp_synthetic() {
    let landmarks = square_landmarks();
    let camera = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
    let rig = make_n1_rig(&camera);
    let cv_true = cv_pose_from_rvec_tvec([0.05, -0.15, 0.08], [0.02, -0.01, 1.5]);
    let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);

    let mono_obs = project_mono(&landmarks, &camera, &cv_true);
    let mv_obs = project_multiview(&landmarks, &rig, &cv_true);

    let pose_mono =
        solve_pnp(&landmarks, &mono_obs, &camera, SolvePnpMethod::Iterative).expect("mono");
    let pose_mv = solve_pnp_multiview(&landmarks, &mv_obs, &rig, SolvePnpMethod::Iterative)
        .expect("multiview n1");

    // Both recover ground truth.
    let (mono_pos, mono_rot) = pose_errors(&pose_mono, &gl_true);
    let (mv_pos, mv_rot) = pose_errors(&pose_mv, &gl_true);
    assert!(mono_pos < 1e-3, "mono pos err {}", mono_pos);
    assert!(mono_rot < 1e-3, "mono rot err {}", mono_rot);
    assert!(mv_pos < 1e-3, "mv pos err {}", mv_pos);
    assert!(mv_rot < 1e-3, "mv rot err {}", mv_rot);

    // N=1 ≡ mono (within tight float tol).
    let (pos_err, rot_err) = pose_errors(&pose_mv, &pose_mono);
    assert!(
        pos_err < 1e-6,
        "N=1 vs mono position error {} exceeds 1e-6",
        pos_err
    );
    assert!(
        rot_err < 1e-6,
        "N=1 vs mono rotation error {} rad exceeds 1e-6",
        rot_err
    );
}

#[test]
fn multiview_n1_recovers_known_pose_all_methods() {
    let landmarks = square_landmarks();
    let camera = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
    let rig = make_n1_rig(&camera);
    let cv_true = cv_pose_from_rvec_tvec([0.05, -0.15, 0.08], [0.02, -0.01, 1.5]);
    let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
    let obs = project_multiview(&landmarks, &rig, &cv_true);

    for method in [
        SolvePnpMethod::Iterative,
        SolvePnpMethod::EPnP,
        SolvePnpMethod::SQPnP,
    ] {
        let pose = solve_pnp_multiview(&landmarks, &obs, &rig, method).expect("multiview");
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

