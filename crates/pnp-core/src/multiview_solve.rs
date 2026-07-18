//! Joint multi-view Perspective-n-Point: mono seed + N-view Levenberg–Marquardt.
//!
//! # Pipeline
//! 1. Match landmarks to [`MultiViewObservation`]s by `id`.
//! 2. Seed object pose with monocular [`solve_pnp`]: prefer the **primary**
//!    view; if it lacks enough points for `method`, seed from the view with
//!    the most observations (still ≥ method minimum) and transport the pose
//!    into the primary frame.
//! 3. Refine with joint LM over all-view reprojection residuals (OpenCV frame).
//! 4. Return the refined object pose in **OpenGL** (same convention as mono).
//!
//! # Frames
//! - Seed and public results: OpenGL object pose (primary view = `views[0]`).
//! - Joint residual math: OpenCV object-in-primary pose; each view's
//!   `from_primary` is treated as OpenCV extrinsics
//!   (`X_c = R * X_primary + t`).
//! - Distortion: pixels are undistorted first; residuals use ideal pinhole `K`.

use crate::camera::Camera;
use crate::multiview::{MultiViewObservation, MultiViewRig};
use crate::pose_tools::{self, compose_poses, invert_pose, transform_point};
use crate::rodrigues;
use crate::solve::{camera_pose_from_solve_pnp_pose, solve_pnp};
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

/// Matched object point with per-view **undistorted** pixels (`None` = missing).
struct MultiViewPair {
    object: Vector3,
    /// Length equals `rig.num_views()`.
    pixels: Vec<Option<Vector2>>,
}

/// Estimate the **object** pose from multi-view landmark observations.
///
/// # Inputs
/// - `landmarks` / `observations`: matched by string `id` (same length required)
/// - `rig`: calibrated multi-view rig; `from_primary` in **OpenCV** convention
/// - `method`: monocular solver used for the seed (primary preferred; otherwise
///   richest view with enough points)
///
/// Missing pixels in any view are skipped in the joint residual. The seed
/// needs enough observations in **some** view for the chosen monocular method
/// (primary preferred when it meets the minimum).
///
/// # Returns
/// Object pose in **OpenGL** (primary view), same meaning as [`solve_pnp`].
///
/// # Errors
/// - [`PnpError::MismatchedCounts`] — length, pixel-vector length, or id mismatch
/// - [`PnpError::InsufficientPoints`] — too few projections / seed points
/// - [`PnpError::SolverFailed`] — seed or numerical failure
pub fn solve_pnp_multiview(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    rig: &MultiViewRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pairs = match_multiview_correspondences(landmarks, observations, rig)?;

    let n_proj = count_projections(&pairs);
    if n_proj * 2 < MIN_RESIDUALS {
        return Err(PnpError::InsufficientPoints);
    }

    // --- Seed: mono PnP on primary, or richest eligible view if primary sparse ---
    let seed_cv = seed_object_pose_cv(landmarks, observations, rig, method)?;
    let (mut rvec, mut tvec) = pose_to_rvec_tvec(&seed_cv);

    // --- Joint LM (finite-difference Jacobian) ---
    refine_multiview_lm(&pairs, rig, &mut rvec, &mut tvec)?;

    let refined_cv = rvec_tvec_to_pose(&rvec, &tvec);
    Ok(pose_tools::from_opencv_to_opengl(&refined_cv))
}

/// Multi-view object pose inverted to camera pose (same convention as mono).
///
/// Equivalent to [`solve_pnp_multiview`] followed by
/// [`camera_pose_from_solve_pnp_pose`].
pub fn solve_pnp_multiview_camera_pose(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    rig: &MultiViewRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let pose = solve_pnp_multiview(landmarks, observations, rig, method)?;
    Ok(camera_pose_from_solve_pnp_pose(&pose))
}

/// Match by id; undistort present pixels; require equal lengths and present ids.
fn match_multiview_correspondences(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    rig: &MultiViewRig,
) -> Result<Vec<MultiViewPair>, PnpError> {
    if landmarks.len() != observations.len() {
        return Err(PnpError::MismatchedCounts);
    }

    let n_views = rig.num_views();
    let mut pairs = Vec::with_capacity(observations.len());
    for obs in observations {
        if obs.pixels.len() != n_views {
            return Err(PnpError::MismatchedCounts);
        }

        let landmark = landmarks
            .iter()
            .find(|l| l.id == obs.id)
            .ok_or(PnpError::MismatchedCounts)?;

        let mut pixels = Vec::with_capacity(n_views);
        for (c, px) in obs.pixels.iter().enumerate() {
            pixels.push(px.map(|p| rig.views[c].camera.undistort_pixel(p)));
        }

        pairs.push(MultiViewPair {
            object: landmark.position,
            pixels,
        });
    }

    Ok(pairs)
}

/// Minimum monocular correspondences required by [`solve_pnp`] for `method`.
fn min_mono_points(method: SolvePnpMethod) -> usize {
    match method {
        SolvePnpMethod::SQPnP => 3,
        _ => 4,
    }
}

/// Collect monocular landmarks/observations for view index `view_idx`.
fn view_mono_inputs(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    view_idx: usize,
) -> (Vec<Landmark>, Vec<LandmarkObservation>) {
    let mut lms = Vec::new();
    let mut obs_out = Vec::new();
    for obs in observations {
        let Some(px) = obs.pixels.get(view_idx).and_then(|p| *p) else {
            continue;
        };
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
    (lms, obs_out)
}

/// Pick seed view: primary if it has ≥ min points; else richest view ≥ min.
fn select_seed_view(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    n_views: usize,
    method: SolvePnpMethod,
) -> Option<usize> {
    let min_pts = min_mono_points(method);
    let counts: Vec<usize> = (0..n_views)
        .map(|c| view_mono_inputs(landmarks, observations, c).1.len())
        .collect();

    if counts.first().copied().unwrap_or(0) >= min_pts {
        return Some(0);
    }

    counts
        .iter()
        .enumerate()
        .filter(|(_, &n)| n >= min_pts)
        .max_by_key(|(idx, n)| (*n, core::cmp::Reverse(*idx))) // more pts; tie → lower idx
        .map(|(idx, _)| idx)
}

/// Monocular seed in OpenCV **object-in-primary** frame.
///
/// If the seed view is not primary, transport:
/// `T_primary = inv(from_primary_c) ∘ T_view_c` (apply `T_view_c` then inv).
fn seed_object_pose_cv(
    landmarks: &[Landmark],
    observations: &[MultiViewObservation],
    rig: &MultiViewRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let seed_view = select_seed_view(landmarks, observations, rig.num_views(), method)
        .ok_or(PnpError::InsufficientPoints)?;

    let (lms, mono_obs) = view_mono_inputs(landmarks, observations, seed_view);
    let seed_gl = solve_pnp(&lms, &mono_obs, &rig.views[seed_view].camera, method)?;
    let seed_in_view_cv = pose_tools::from_opengl_to_opencv(&seed_gl);

    if seed_view == 0 {
        return Ok(seed_in_view_cv);
    }

    // T_p = inv(T_{c←primary}) * T_c
    let from_primary = &rig.views[seed_view].from_primary;
    Ok(compose_poses(&invert_pose(from_primary), &seed_in_view_cv))
}

fn count_projections(pairs: &[MultiViewPair]) -> usize {
    pairs.iter().fold(0usize, |acc, p| {
        acc + p.pixels.iter().filter(|px| px.is_some()).count()
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

fn transform_object_to_primary(rvec: &[f64; 3], tvec: &NaVector3<f64>, pw: Vector3) -> Vector3 {
    let r = rodrigues::rvec_to_rotation_matrix(rvec).to_na();
    let p = r * NaVector3::new(pw.x, pw.y, pw.z) + tvec;
    Vector3::new(p.x, p.y, p.z)
}

/// Build residual vector of length `2 * n_projections`.
fn compute_residuals(
    pairs: &[MultiViewPair],
    rig: &MultiViewRig,
    rvec: &[f64; 3],
    tvec: &NaVector3<f64>,
) -> DVector<f64> {
    let n = count_projections(pairs);
    let mut residuals = DVector::zeros(2 * n);
    let mut row = 0usize;

    for pair in pairs {
        let x_primary = transform_object_to_primary(rvec, tvec, pair.object);

        for (c, maybe_obs) in pair.pixels.iter().enumerate() {
            let Some(obs) = maybe_obs else {
                continue;
            };

            let x_c = if c == 0 {
                x_primary
            } else {
                transform_point(&rig.views[c].from_primary, x_primary)
            };

            match project_pinhole(&rig.views[c].camera, x_c) {
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
    pairs: &[MultiViewPair],
    rig: &MultiViewRig,
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

fn refine_multiview_lm(
    pairs: &[MultiViewPair],
    rig: &MultiViewRig,
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
    use alloc::vec;

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

    fn make_n1_rig() -> MultiViewRig {
        let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        MultiViewRig::new(vec![crate::multiview::CameraView {
            camera: cam,
            from_primary: Pose::identity(),
        }])
        .unwrap()
    }

    fn true_cv_pose() -> Pose {
        let rvec = [0.05, -0.15, 0.08];
        let t = NaVector3::new(0.02, -0.01, 1.5);
        rvec_tvec_to_pose(&rvec, &t)
    }

    fn project_multiview_obs(
        landmarks: &[Landmark],
        rig: &MultiViewRig,
        cv_pose: &Pose,
    ) -> Vec<MultiViewObservation> {
        let (rvec, tvec) = pose_to_rvec_tvec(cv_pose);
        landmarks
            .iter()
            .map(|lm| {
                let x_primary = transform_object_to_primary(&rvec, &tvec, lm.position);
                let mut pixels = Vec::with_capacity(rig.num_views());
                for (c, view) in rig.views.iter().enumerate() {
                    let x_c = if c == 0 {
                        x_primary
                    } else {
                        transform_point(&view.from_primary, x_primary)
                    };
                    let px = project_pinhole(&view.camera, x_c).expect("project");
                    pixels.push(Some(px));
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
        2.0 * libm::acos(dot)
    }

    #[test]
    fn recovers_known_n1_pose() {
        let landmarks = square_landmarks();
        let rig = make_n1_rig();
        let cv_true = true_cv_pose();
        let gl_true = pose_tools::from_opencv_to_opengl(&cv_true);
        let obs = project_multiview_obs(&landmarks, &rig, &cv_true);

        let pose =
            solve_pnp_multiview(&landmarks, &obs, &rig, SolvePnpMethod::Iterative).unwrap();

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
    fn residual_zero_at_ground_truth() {
        let landmarks = square_landmarks();
        let rig = make_n1_rig();
        let cv_true = true_cv_pose();
        let obs = project_multiview_obs(&landmarks, &rig, &cv_true);
        let pairs = match_multiview_correspondences(&landmarks, &obs, &rig).unwrap();
        let (rvec, tvec) = pose_to_rvec_tvec(&cv_true);
        let cost = residual_cost(&compute_residuals(&pairs, &rig, &rvec, &tvec));
        assert!(cost < 1e-12, "cost at GT should be ~0, got {}", cost);
    }

    #[test]
    fn mismatched_pixel_len_error() {
        let landmarks = square_landmarks();
        let rig = make_n1_rig();
        let obs = vec![MultiViewObservation {
            id: "0".to_string(),
            // Wrong length: 2 pixels for N=1 rig
            pixels: vec![Some(Vector2::new(1.0, 1.0)), Some(Vector2::new(2.0, 2.0))],
        }];
        // Also length mismatch on landmarks vs observations
        let err =
            solve_pnp_multiview(&landmarks, &obs, &rig, SolvePnpMethod::EPnP).unwrap_err();
        assert_eq!(err, PnpError::MismatchedCounts);
    }

    #[test]
    fn camera_pose_is_inverse() {
        let landmarks = square_landmarks();
        let rig = make_n1_rig();
        let cv_true = true_cv_pose();
        let obs = project_multiview_obs(&landmarks, &rig, &cv_true);

        let obj =
            solve_pnp_multiview(&landmarks, &obs, &rig, SolvePnpMethod::SQPnP).unwrap();
        let cam =
            solve_pnp_multiview_camera_pose(&landmarks, &obs, &rig, SolvePnpMethod::SQPnP)
                .unwrap();
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
