//! Joint stereo Perspective-n-Point: mono seed + dual-view Levenberg–Marquardt.
//!
//! # Pipeline
//! 1. Match landmarks to [`StereoLandmarkObservation`]s by `id`.
//! 2. Seed object pose with monocular [`solve_pnp`] on the **left** view.
//! 3. Refine with joint LM over left + right reprojection residuals (OpenCV frame).
//! 4. Return the refined object pose in **OpenGL** (same convention as mono).
//!
//! # Frames
//! - Seed and public results: OpenGL object pose (primary view = left).
//! - Joint residual math: OpenCV object-in-left pose; `rig.right_from_left`
//!   is treated as OpenCV extrinsics (`X_right = R_rl * X_left + t_rl`).
//! - Distortion: pixels are undistorted first; residuals use ideal pinhole `K`.

use crate::camera::Camera;
use crate::pose_tools::{self, transform_point};
use crate::rodrigues;
use crate::solve::{camera_pose_from_solve_pnp_pose, solve_pnp};
use crate::stereo::{StereoLandmarkObservation, StereoRig};
use crate::types::{
    rotation_matrix_to_quaternion, Landmark, LandmarkObservation, Matrix3x3, PnpError, Pose,
    SolvePnpMethod, Vector2, Vector3,
};
use alloc::vec::Vec;
use nalgebra::{DMatrix, DVector, Vector3 as NaVector3};

/// Minimum number of residual components (2 per image projection).
const MIN_RESIDUALS: usize = 6;
const MAX_LM_ITERS: usize = 100;
const LM_LAMBDA0: f64 = 1e-3;
const LM_LAMBDA_FACTOR: f64 = 10.0;
const LM_CONV: f64 = 1e-8;
const FD_EPS: f64 = 1e-6;
const DEPTH_EPS: f64 = 1e-12;

/// Matched object point with optional left/right **undistorted** pixels.
struct StereoPair {
    object: Vector3,
    left: Option<Vector2>,
    right: Option<Vector2>,
}

/// Estimate the **object** pose from stereo landmark observations.
///
/// # Inputs
/// - `landmarks` / `observations`: matched by string `id` (same length required)
/// - `rig`: calibrated stereo pair; `right_from_left` in **OpenCV** convention
/// - `method`: monocular solver used only for the left-view seed
///
/// Missing left or right pixels are skipped in the joint residual. The seed
/// requires enough **left** observations for the chosen monocular method.
///
/// # Returns
/// Object pose in **OpenGL** (primary view = left camera), same meaning as
/// [`solve_pnp`].
///
/// # Errors
/// - [`PnpError::MismatchedCounts`] — length or id mismatch
/// - [`PnpError::InsufficientPoints`] — too few projections / left seed points
/// - [`PnpError::SolverFailed`] — seed or numerical failure
pub fn solve_pnp_stereo(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pairs = match_stereo_correspondences(landmarks, observations, rig)?;

    let n_proj = count_projections(&pairs);
    if n_proj * 2 < MIN_RESIDUALS {
        return Err(PnpError::InsufficientPoints);
    }

    // --- Seed: monocular PnP on left observations ---
    let (left_landmarks, left_obs) = left_mono_inputs(landmarks, observations);
    if left_obs.is_empty() {
        return Err(PnpError::InsufficientPoints);
    }
    let seed_gl = solve_pnp(&left_landmarks, &left_obs, &rig.left, method)?;
    let seed_cv = pose_tools::from_opengl_to_opencv(&seed_gl);
    let (mut rvec, mut tvec) = pose_to_rvec_tvec(&seed_cv);

    // --- Joint LM (finite-difference Jacobian) ---
    refine_stereo_lm(&pairs, rig, &mut rvec, &mut tvec)?;

    let refined_cv = rvec_tvec_to_pose(&rvec, &tvec);
    Ok(pose_tools::from_opencv_to_opengl(&refined_cv))
}

/// Stereo object pose inverted to camera pose (same convention as mono).
///
/// Equivalent to [`solve_pnp_stereo`] followed by
/// [`camera_pose_from_solve_pnp_pose`].
pub fn solve_pnp_stereo_camera_pose(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pose = solve_pnp_stereo(landmarks, observations, rig, method)?;
    Ok(camera_pose_from_solve_pnp_pose(&pose))
}

/// Match by id; undistort pixels; require equal lengths and present ids.
fn match_stereo_correspondences(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
) -> Result<Vec<StereoPair>, PnpError> {
    if landmarks.len() != observations.len() {
        return Err(PnpError::MismatchedCounts);
    }

    let mut pairs = Vec::with_capacity(observations.len());
    for obs in observations {
        let landmark = landmarks
            .iter()
            .find(|l| l.id == obs.id)
            .ok_or(PnpError::MismatchedCounts)?;

        if obs.left.is_none() && obs.right.is_none() {
            // No measurements for this id — still a valid match, zero residual.
            pairs.push(StereoPair {
                object: landmark.position,
                left: None,
                right: None,
            });
            continue;
        }

        let left = obs.left.map(|p| rig.left.undistort_pixel(p));
        let right = obs.right.map(|p| rig.right.undistort_pixel(p));
        pairs.push(StereoPair {
            object: landmark.position,
            left,
            right,
        });
    }

    Ok(pairs)
}

fn left_mono_inputs(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
) -> (Vec<Landmark>, Vec<LandmarkObservation>) {
    let mut lms = Vec::new();
    let mut obs_out = Vec::new();
    for obs in observations {
        if let Some(px) = obs.left {
            if let Some(lm) = landmarks.iter().find(|l| l.id == obs.id) {
                lms.push(Landmark {
                    id: lm.id.clone(),
                    position: lm.position,
                });
                obs_out.push(LandmarkObservation {
                    id: obs.id.clone(),
                    position: px,
                });
            }
        }
    }
    (lms, obs_out)
}

fn count_projections(pairs: &[StereoPair]) -> usize {
    pairs.iter().fold(0usize, |acc, p| {
        acc + p.left.is_some() as usize + p.right.is_some() as usize
    })
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

/// Ideal pinhole project (no distortion) — residuals compare to undistorted pixels.
fn project_pinhole(cam: &Camera, pc: Vector3) -> Option<Vector2> {
    if !pc.z.is_finite() || pc.z.abs() < DEPTH_EPS {
        return None;
    }
    Some(Vector2::new(
        cam.fx * pc.x / pc.z + cam.cx,
        cam.fy * pc.y / pc.z + cam.cy,
    ))
}

fn transform_object_to_left(rvec: &[f64; 3], tvec: &NaVector3<f64>, pw: Vector3) -> Vector3 {
    let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
    let p = r * NaVector3::new(pw.x, pw.y, pw.z) + tvec;
    Vector3::new(p.x, p.y, p.z)
}

/// Build residual vector of length `2 * n_projections`.
fn compute_residuals(
    pairs: &[StereoPair],
    rig: &StereoRig,
    rvec: &[f64; 3],
    tvec: &NaVector3<f64>,
) -> DVector<f64> {
    let n = count_projections(pairs);
    let mut residuals = DVector::zeros(2 * n);
    let mut row = 0usize;

    for pair in pairs {
        let x_left = transform_object_to_left(rvec, tvec, pair.object);

        if let Some(obs) = pair.left {
            match project_pinhole(&rig.left, x_left) {
                Some(proj) => {
                    residuals[row] = proj.x - obs.x;
                    residuals[row + 1] = proj.y - obs.y;
                }
                None => {
                    residuals[row] = 1e6;
                    residuals[row + 1] = 1e6;
                }
            }
            row += 2;
        }

        if let Some(obs) = pair.right {
            let x_right = transform_point(&rig.right_from_left, x_left);
            match project_pinhole(&rig.right, x_right) {
                Some(proj) => {
                    residuals[row] = proj.x - obs.x;
                    residuals[row + 1] = proj.y - obs.y;
                }
                None => {
                    residuals[row] = 1e6;
                    residuals[row + 1] = 1e6;
                }
            }
            row += 2;
        }
    }

    residuals
}

fn residual_cost(residuals: &DVector<f64>) -> f64 {
    residuals.iter().map(|r| r * r).sum()
}

/// Finite-difference Jacobian of residuals w.r.t. `[rvec; tvec]` (6 params).
fn compute_jacobian_fd(
    pairs: &[StereoPair],
    rig: &StereoRig,
    rvec: &[f64; 3],
    tvec: &NaVector3<f64>,
) -> DMatrix<f64> {
    let n = count_projections(pairs);
    let mut jac = DMatrix::zeros(2 * n, 6);
    let base = [rvec[0], rvec[1], rvec[2], tvec.x, tvec.y, tvec.z];

    for k in 0..6 {
        let mut plus = base;
        let mut minus = base;
        plus[k] += FD_EPS;
        minus[k] -= FD_EPS;

        let rv_p = [plus[0], plus[1], plus[2]];
        let tv_p = NaVector3::new(plus[3], plus[4], plus[5]);
        let rv_m = [minus[0], minus[1], minus[2]];
        let tv_m = NaVector3::new(minus[3], minus[4], minus[5]);

        let res_p = compute_residuals(pairs, rig, &rv_p, &tv_p);
        let res_m = compute_residuals(pairs, rig, &rv_m, &tv_m);

        for i in 0..(2 * n) {
            jac[(i, k)] = (res_p[i] - res_m[i]) / (2.0 * FD_EPS);
        }
    }

    jac
}

fn refine_stereo_lm(
    pairs: &[StereoPair],
    rig: &StereoRig,
    rvec: &mut [f64; 3],
    tvec: &mut NaVector3<f64>,
) -> Result<(), PnpError> {
    let mut lambda = LM_LAMBDA0;
    let mut prev_cost = residual_cost(&compute_residuals(pairs, rig, rvec, tvec));
    if !prev_cost.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    for _ in 0..MAX_LM_ITERS {
        let residuals = compute_residuals(pairs, rig, rvec, tvec);
        let jacobian = compute_jacobian_fd(pairs, rig, rvec, tvec);

        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        let mut a = jtj.clone();
        for i in 0..6 {
            a[(i, i)] += lambda * jtj[(i, i)].max(1e-10);
        }

        let neg_jtr = -&jtr;
        let delta = match a.lu().solve(&neg_jtr) {
            Some(d) => d,
            None => break,
        };

        if delta.norm() < LM_CONV {
            break;
        }

        let new_rvec = [rvec[0] + delta[0], rvec[1] + delta[1], rvec[2] + delta[2]];
        let new_tvec = NaVector3::new(tvec.x + delta[3], tvec.y + delta[4], tvec.z + delta[5]);
        let new_cost = residual_cost(&compute_residuals(pairs, rig, &new_rvec, &new_tvec));

        if new_cost.is_finite() && new_cost < prev_cost {
            *rvec = new_rvec;
            *tvec = new_tvec;
            lambda = (lambda / LM_LAMBDA_FACTOR).max(1e-10);
            if (prev_cost - new_cost) / prev_cost.max(1e-15) < LM_CONV {
                break;
            }
            prev_cost = new_cost;
        } else {
            lambda *= LM_LAMBDA_FACTOR;
            if lambda > 1e16 {
                break;
            }
        }
    }

    // Final sanity: refined parameters must be finite.
    if !rvec[0].is_finite()
        || !rvec[1].is_finite()
        || !rvec[2].is_finite()
        || !tvec.x.is_finite()
        || !tvec.y.is_finite()
        || !tvec.z.is_finite()
    {
        return Err(PnpError::SolverFailed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Quaternion;
    use alloc::string::ToString;

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
                let left = project_pinhole(&rig.left, x_left).expect("left project");
                let x_right = transform_point(&rig.right_from_left, x_left);
                let right = project_pinhole(&rig.right, x_right).expect("right project");
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

    #[test]
    fn residual_zero_at_ground_truth() {
        let landmarks = square_landmarks();
        let rig = make_rig();
        let cv_true = true_cv_pose();
        let obs = project_stereo_obs(&landmarks, &rig, &cv_true);
        let pairs = match_stereo_correspondences(&landmarks, &obs, &rig).unwrap();
        let (rvec, tvec) = pose_to_rvec_tvec(&cv_true);
        let cost = residual_cost(&compute_residuals(&pairs, &rig, &rvec, &tvec));
        assert!(cost < 1e-12, "cost at GT should be ~0, got {}", cost);
    }
}
