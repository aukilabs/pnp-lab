use pnp_core::types::*;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct ReferenceData {
    landmarks: Vec<[f64; 3]>,
    camera_matrix: [[f64; 3]; 3],
    observation_sets: Vec<Vec<[f64; 2]>>,
    results: Vec<ReferenceResult>,
}

#[derive(Deserialize)]
struct ReferenceResult {
    set_index: usize,
    method: String,
    success: bool,
    rvec: [f64; 3],
    tvec: [f64; 3],
    gl_position: [f64; 3],
    gl_quaternion: [f64; 4],
    camera_position: [f64; 3],
    camera_quaternion: [f64; 4],
}

fn load_reference() -> ReferenceData {
    // cargo test runs from workspace root
    let data = fs::read_to_string("tests/reference_vectors/reference_output.json")
        .or_else(|_| fs::read_to_string("../../tests/reference_vectors/reference_output.json"))
        .expect("reference output file not found");
    serde_json::from_str(&data).expect("failed to parse reference JSON")
}

fn make_landmarks(ref_data: &ReferenceData) -> Vec<Landmark> {
    ref_data
        .landmarks
        .iter()
        .enumerate()
        .map(|(i, p)| Landmark {
            id: i.to_string(),
            position: Vector3::new(p[0], p[1], p[2]),
        })
        .collect()
}

fn make_observations(ref_data: &ReferenceData, set_idx: usize) -> Vec<LandmarkObservation> {
    ref_data.observation_sets[set_idx]
        .iter()
        .enumerate()
        .map(|(i, p)| LandmarkObservation {
            id: i.to_string(),
            position: Vector2::new(p[0], p[1]),
        })
        .collect()
}

fn make_camera_matrix(ref_data: &ReferenceData) -> Matrix3x3 {
    let k = &ref_data.camera_matrix;
    Matrix3x3::camera_matrix(k[0][0], k[1][1], k[0][2], k[1][2])
}

fn to_method(name: &str) -> Option<SolvePnpMethod> {
    match name {
        "epnp" => Some(SolvePnpMethod::EPnP),
        "iterative" => Some(SolvePnpMethod::Iterative),
        "sqpnp" => Some(SolvePnpMethod::SQPnP),
        _ => None,
    }
}

/// Compute rotation angle between two quaternions (handles sign ambiguity).
fn quaternion_angle(q1: &Quaternion, q2: &Quaternion) -> f64 {
    let dot = (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w).abs();
    2.0 * (dot.min(1.0)).acos()
}

fn position_error(p1: &Vector3, p2: &Vector3) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)).sqrt()
}

// ---------- solve_pnp (GL pose) ----------

#[test]
fn test_iterative_set0_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 0 && r.method == "iterative")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 0);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(
        pos_err < 1e-3,
        "set0 iterative GL position error: {}",
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "set0 iterative GL rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_iterative_set1_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 1 && r.method == "iterative")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 1);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(
        pos_err < 1e-3,
        "set1 iterative GL position error: {}",
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "set1 iterative GL rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_iterative_set2_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 2 && r.method == "iterative")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 2);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(
        pos_err < 1e-3,
        "set2 iterative GL position error: {}",
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "set2 iterative GL rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_sqpnp_set0_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 0 && r.method == "sqpnp")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 0);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(pos_err < 1e-3, "set0 sqpnp GL position error: {}", pos_err);
    assert!(
        rot_err < 1e-3,
        "set0 sqpnp GL rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_sqpnp_set1_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 1 && r.method == "sqpnp")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 1);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(pos_err < 1e-3, "set1 sqpnp GL position error: {}", pos_err);
    assert!(
        rot_err < 1e-3,
        "set1 sqpnp GL rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_sqpnp_set2_gl_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 2 && r.method == "sqpnp")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 2);
    let cam = make_camera_matrix(&ref_data);

    let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

    let ref_pos = Vector3::new(r.gl_position[0], r.gl_position[1], r.gl_position[2]);
    let ref_quat = Quaternion::new(
        r.gl_quaternion[0],
        r.gl_quaternion[1],
        r.gl_quaternion[2],
        r.gl_quaternion[3],
    );

    let pos_err = position_error(&pose.position, &ref_pos);
    let rot_err = quaternion_angle(&pose.rotation, &ref_quat);

    assert!(pos_err < 1e-3, "set2 sqpnp GL position error: {}", pos_err);
    assert!(
        rot_err < 1e-3,
        "set2 sqpnp GL rotation error: {} rad",
        rot_err
    );
}

// ---------- solve_pnp_camera_pose ----------

#[test]
fn test_iterative_set0_camera_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 0 && r.method == "iterative")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 0);
    let cam = make_camera_matrix(&ref_data);

    let cam_pose =
        pnp_core::solve_pnp_camera_pose(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();

    let ref_pos = Vector3::new(
        r.camera_position[0],
        r.camera_position[1],
        r.camera_position[2],
    );
    let ref_quat = Quaternion::new(
        r.camera_quaternion[0],
        r.camera_quaternion[1],
        r.camera_quaternion[2],
        r.camera_quaternion[3],
    );

    let pos_err = position_error(&cam_pose.position, &ref_pos);
    let rot_err = quaternion_angle(&cam_pose.rotation, &ref_quat);

    assert!(
        pos_err < 1e-3,
        "set0 iterative camera position error: {}",
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "set0 iterative camera rotation error: {} rad",
        rot_err
    );
}

#[test]
fn test_sqpnp_set0_camera_pose() {
    let ref_data = load_reference();
    let r = ref_data
        .results
        .iter()
        .find(|r| r.set_index == 0 && r.method == "sqpnp")
        .unwrap();

    let landmarks = make_landmarks(&ref_data);
    let obs = make_observations(&ref_data, 0);
    let cam = make_camera_matrix(&ref_data);

    let cam_pose =
        pnp_core::solve_pnp_camera_pose(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap();

    let ref_pos = Vector3::new(
        r.camera_position[0],
        r.camera_position[1],
        r.camera_position[2],
    );
    let ref_quat = Quaternion::new(
        r.camera_quaternion[0],
        r.camera_quaternion[1],
        r.camera_quaternion[2],
        r.camera_quaternion[3],
    );

    let pos_err = position_error(&cam_pose.position, &ref_pos);
    let rot_err = quaternion_angle(&cam_pose.rotation, &ref_quat);

    assert!(
        pos_err < 1e-3,
        "set0 sqpnp camera position error: {}",
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "set0 sqpnp camera rotation error: {} rad",
        rot_err
    );
}

// ---------- Consistency checks ----------

#[test]
fn test_camera_pose_equals_inverted_solve_pnp() {
    let ref_data = load_reference();
    let landmarks = make_landmarks(&ref_data);
    let cam = make_camera_matrix(&ref_data);

    for set_idx in 0..3 {
        let obs = make_observations(&ref_data, set_idx);

        let pose = pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap();
        let cam_pose =
            pnp_core::solve_pnp_camera_pose(&landmarks, &obs, &cam, SolvePnpMethod::Iterative)
                .unwrap();
        let inverted = pnp_core::camera_pose_from_solve_pnp_pose(&pose);

        let pos_err = position_error(&cam_pose.position, &inverted.position);
        let rot_err = quaternion_angle(&cam_pose.rotation, &inverted.rotation);

        assert!(
            pos_err < 1e-6,
            "set{} camera_pose != inverted solve_pnp, pos diff: {}",
            set_idx,
            pos_err
        );
        assert!(
            rot_err < 1e-6,
            "set{} camera_pose != inverted solve_pnp, rot diff: {} rad",
            set_idx,
            rot_err
        );
    }
}

#[test]
fn test_opencv_opengl_roundtrip() {
    use pnp_core::pose_tools::{from_opencv_to_opengl, from_opengl_to_opencv};

    let pose = Pose::new(
        Vector3::new(1.5, -0.3, 2.0),
        Quaternion::new(0.1, 0.2, 0.3, 0.9).normalize(),
    );

    let roundtrip = from_opengl_to_opencv(&from_opencv_to_opengl(&pose));

    let pos_err = position_error(&pose.position, &roundtrip.position);
    let rot_err = quaternion_angle(&pose.rotation, &roundtrip.rotation);

    assert!(pos_err < 1e-10, "roundtrip position error: {}", pos_err);
    assert!(rot_err < 1e-6, "roundtrip rotation error: {} rad", rot_err);
}

// ---------- All methods produce valid results ----------

#[test]
fn test_all_methods_succeed_all_sets() {
    let ref_data = load_reference();
    let landmarks = make_landmarks(&ref_data);
    let cam = make_camera_matrix(&ref_data);

    for set_idx in 0..3 {
        let obs = make_observations(&ref_data, set_idx);
        for method in [
            SolvePnpMethod::EPnP,
            SolvePnpMethod::Iterative,
            SolvePnpMethod::SQPnP,
        ] {
            let result = pnp_core::solve_pnp(&landmarks, &obs, &cam, method);
            assert!(
                result.is_ok(),
                "set{} {:?} failed: {:?}",
                set_idx,
                method,
                result.err()
            );
        }
    }
}

// ---------- Reprojection error (low-level) ----------

#[test]
fn test_iterative_reprojection_error_all_sets() {
    let ref_data = load_reference();
    let landmarks = make_landmarks(&ref_data);
    let cam = make_camera_matrix(&ref_data);
    let na_cam =
        pnp_core::types::Matrix3x3::camera_matrix(815.8511, 815.8511, 960.0, 540.0).to_na();

    for set_idx in 0..3 {
        let obs = make_observations(&ref_data, set_idx);
        let object_points: Vec<_> = landmarks.iter().map(|l| l.position).collect();
        let image_points: Vec<_> = obs.iter().map(|o| o.position).collect();

        let (r, t) = pnp_core::iterative::solve_iterative(
            &object_points,
            &image_points,
            &na_cam,
            None,
            None,
        )
        .unwrap();

        let reproj = pnp_core::epnp::reprojection_error(
            &object_points,
            &image_points,
            &r,
            &t,
            815.8511,
            815.8511,
            960.0,
            540.0,
        );
        assert!(
            reproj < 1.0,
            "set{} iterative reprojection error: {}",
            set_idx,
            reproj
        );
    }
}

#[test]
fn test_sqpnp_reprojection_error_all_sets() {
    let ref_data = load_reference();
    let landmarks = make_landmarks(&ref_data);
    let cam = make_camera_matrix(&ref_data);
    let na_cam =
        pnp_core::types::Matrix3x3::camera_matrix(815.8511, 815.8511, 960.0, 540.0).to_na();

    for set_idx in 0..3 {
        let obs = make_observations(&ref_data, set_idx);
        let object_points: Vec<_> = landmarks.iter().map(|l| l.position).collect();
        let image_points: Vec<_> = obs.iter().map(|o| o.position).collect();

        let (r, t) = pnp_core::sqpnp::solve_sqpnp(&object_points, &image_points, &na_cam).unwrap();

        let reproj = pnp_core::epnp::reprojection_error(
            &object_points,
            &image_points,
            &r,
            &t,
            815.8511,
            815.8511,
            960.0,
            540.0,
        );
        assert!(
            reproj < 1.0,
            "set{} sqpnp reprojection error: {}",
            set_idx,
            reproj
        );
    }
}

// ---------- Minimum point checks ----------

#[test]
fn test_insufficient_points_errors() {
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
    let obs = vec![
        LandmarkObservation {
            id: "0".to_string(),
            position: Vector2::new(0.0, 0.0),
        },
        LandmarkObservation {
            id: "1".to_string(),
            position: Vector2::new(1.0, 0.0),
        },
    ];
    let cam = Matrix3x3::camera_matrix(815.8511, 815.8511, 960.0, 540.0);

    assert_eq!(
        pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::EPnP).unwrap_err(),
        PnpError::InsufficientPoints
    );
    assert_eq!(
        pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::Iterative).unwrap_err(),
        PnpError::InsufficientPoints
    );
    assert_eq!(
        pnp_core::solve_pnp(&landmarks, &obs, &cam, SolvePnpMethod::SQPnP).unwrap_err(),
        PnpError::InsufficientPoints
    );
}
