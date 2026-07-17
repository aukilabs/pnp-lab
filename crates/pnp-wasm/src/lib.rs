#[allow(warnings)]
mod bindings;

use bindings::auki::pnp::types as wit;
use bindings::exports::auki::pnp::solver::Guest;
use pnp_core::types as core;

struct PnpComponent;

impl Guest for PnpComponent {
    fn solve_pnp(
        landmarks: Vec<wit::Landmark>,
        observations: Vec<wit::LandmarkObservation>,
        camera_matrix: wit::Matrix3x3,
        method: wit::SolvePnpMethod,
    ) -> Result<wit::Pose, wit::PnpError> {
        let core_landmarks = landmarks
            .into_iter()
            .map(to_core_landmark)
            .collect::<Vec<_>>();
        let core_obs = observations
            .into_iter()
            .map(to_core_observation)
            .collect::<Vec<_>>();
        let core_cam = to_core_matrix(camera_matrix);
        let core_method = to_core_method(method);

        pnp_core::solve_pnp(&core_landmarks, &core_obs, &core_cam, core_method)
            .map(|p| to_wit_pose(&p))
            .map_err(to_wit_error)
    }

    fn solve_pnp_camera_pose(
        landmarks: Vec<wit::Landmark>,
        observations: Vec<wit::LandmarkObservation>,
        camera_matrix: wit::Matrix3x3,
        method: wit::SolvePnpMethod,
    ) -> Result<wit::Pose, wit::PnpError> {
        let core_landmarks = landmarks
            .into_iter()
            .map(to_core_landmark)
            .collect::<Vec<_>>();
        let core_obs = observations
            .into_iter()
            .map(to_core_observation)
            .collect::<Vec<_>>();
        let core_cam = to_core_matrix(camera_matrix);
        let core_method = to_core_method(method);

        pnp_core::solve_pnp_camera_pose(&core_landmarks, &core_obs, &core_cam, core_method)
            .map(|p| to_wit_pose(&p))
            .map_err(to_wit_error)
    }

    fn camera_pose_from_solve_pnp_pose(pose: wit::Pose) -> wit::Pose {
        let core_pose = to_core_pose(&pose);
        let result = pnp_core::camera_pose_from_solve_pnp_pose(&core_pose);
        to_wit_pose(&result)
    }
}

// --- Type conversions ---

fn to_core_landmark(l: wit::Landmark) -> core::Landmark {
    core::Landmark {
        id: l.id,
        position: core::Vector3::new(l.position.x, l.position.y, l.position.z),
    }
}

fn to_core_observation(o: wit::LandmarkObservation) -> core::LandmarkObservation {
    core::LandmarkObservation {
        id: o.id,
        position: core::Vector2::new(o.position.x, o.position.y),
    }
}

fn to_core_matrix(m: wit::Matrix3x3) -> core::Matrix3x3 {
    // WIT uses M<col><row> naming, pnp-core uses column-major m[col*3+row]
    core::Matrix3x3 {
        m: [
            m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22,
        ],
    }
}

fn to_core_method(m: wit::SolvePnpMethod) -> core::SolvePnpMethod {
    match m {
        wit::SolvePnpMethod::Epnp => core::SolvePnpMethod::EPnP,
        wit::SolvePnpMethod::Iterative => core::SolvePnpMethod::Iterative,
        wit::SolvePnpMethod::Sqpnp => core::SolvePnpMethod::SQPnP,
    }
}

fn to_core_pose(p: &wit::Pose) -> core::Pose {
    core::Pose::new(
        core::Vector3::new(p.position.x, p.position.y, p.position.z),
        core::Quaternion::new(p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w),
    )
}

fn to_wit_pose(p: &core::Pose) -> wit::Pose {
    wit::Pose {
        position: wit::Vector3 {
            x: p.position.x,
            y: p.position.y,
            z: p.position.z,
        },
        rotation: wit::Quaternion {
            x: p.rotation.x,
            y: p.rotation.y,
            z: p.rotation.z,
            w: p.rotation.w,
        },
    }
}

fn to_wit_error(e: core::PnpError) -> wit::PnpError {
    match e {
        core::PnpError::InsufficientPoints => wit::PnpError::InsufficientPoints,
        core::PnpError::SolverFailed => wit::PnpError::SolverFailed,
        core::PnpError::MismatchedCounts => wit::PnpError::MismatchedCounts,
    }
}

bindings::export!(PnpComponent with_types_in bindings);
