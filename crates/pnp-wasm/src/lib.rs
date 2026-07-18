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
        camera: wit::Camera,
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
        let core_cam = to_core_camera(camera)?;
        let core_method = to_core_method(method);

        pnp_core::solve_pnp(&core_landmarks, &core_obs, &core_cam, core_method)
            .map(|p| to_wit_pose(&p))
            .map_err(to_wit_error)
    }

    fn solve_pnp_camera_pose(
        landmarks: Vec<wit::Landmark>,
        observations: Vec<wit::LandmarkObservation>,
        camera: wit::Camera,
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
        let core_cam = to_core_camera(camera)?;
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

    fn estimate_square_pose_from_pixels(
        pixels: Vec<wit::Vector2>,
        physical_size: f64,
        camera: wit::Camera,
    ) -> Result<wit::SquarePoseEstimate, wit::PnpError> {
        if pixels.len() != 4 {
            return Err(wit::PnpError::InsufficientPoints);
        }
        let core_cam = to_core_camera(camera)?;
        let corners = [
            core::Vector2::new(pixels[0].x, pixels[0].y),
            core::Vector2::new(pixels[1].x, pixels[1].y),
            core::Vector2::new(pixels[2].x, pixels[2].y),
            core::Vector2::new(pixels[3].x, pixels[3].y),
        ];
        pnp_core::estimate_square_pose_from_pixels(corners, physical_size, &core_cam)
            .map(|e| to_wit_square_estimate(&e))
            .map_err(to_wit_error)
    }
}

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

fn to_core_camera(c: wit::Camera) -> Result<pnp_core::Camera, wit::PnpError> {
    pnp_core::Camera::new(c.fx, c.fy, c.cx, c.cy, &c.dist).map_err(|_| wit::PnpError::SolverFailed)
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

fn to_wit_square_estimate(e: &core::SquarePoseEstimate) -> wit::SquarePoseEstimate {
    wit::SquarePoseEstimate {
        pose: to_wit_pose(&e.pose),
        confidence: e.confidence,
        normalized_corner_error: e.normalized_corner_error,
        ray_distances: e.ray_distances.to_vec(),
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
