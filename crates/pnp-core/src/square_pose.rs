//! Planar square-marker pose from four corner rays or image pixels.
//!
//! Recover a rigid pose of a known square (e.g. a QR code or printed marker)
//! from either:
//!
//! - four world-space (or tracking-space) rays through the corners, or
//! - four image pixels plus a [`Camera`] (undistort + OpenGL unprojection), or
//! - dual-view stereo corner pixels via triangulation + square geometry fit.
//!
//! Corner order is always **top-left, top-right, bottom-right, bottom-left**.

use crate::camera::Camera;
use crate::pose_tools;
use crate::stereo::StereoRig;
use crate::triangulate::triangulate_midpoint;
use crate::types::{
    rotation_matrix_to_quaternion, Matrix3x3, PnpError, Pose, Ray3, SquarePoseEstimate, Vector2,
    Vector3,
};

const DISTANCE_COUNT: usize = 4;
const RESIDUAL_COUNT: usize = 9;
const MIN_DISTANCE: f64 = 1e-4;
const MIN_VECTOR_LENGTH: f64 = 1e-12;
const MAX_ITERATIONS: usize = 80;
const MAX_DISTANCE_MULTIPLIER: f64 = 10_000.0;
const SUCCESS_RESIDUAL_RMS: f64 = 0.05;

/// Estimate square pose from four **image corner pixels** and a camera model.
///
/// # Arguments
///
/// - `corner_pixels`: TL, TR, BR, BL in distorted OpenCV image coordinates
/// - `physical_size`: side length of the square in the same units as ray space
///   (typically meters)
/// - `camera`: used to undistort and form OpenGL-style rays via
///   [`Camera::unproject_opengl_ray`]
///
/// Rays are assumed to originate at the camera origin in camera space.
pub fn estimate_square_pose_from_pixels(
    corner_pixels: [Vector2; 4],
    physical_size: f64,
    camera: &Camera,
) -> Result<SquarePoseEstimate, PnpError> {
    let rays = [
        camera.unproject_opengl_ray(corner_pixels[0]),
        camera.unproject_opengl_ray(corner_pixels[1]),
        camera.unproject_opengl_ray(corner_pixels[2]),
        camera.unproject_opengl_ray(corner_pixels[3]),
    ];
    estimate_square_pose_from_rays(rays, physical_size)
}

/// Estimate square pose from four **world- or tracking-space rays**.
///
/// # Arguments
///
/// - `rays`: TL, TR, BR, BL. Origins and directions may be unnormalized;
///   directions are normalized internally. Units must match `physical_size`.
/// - `physical_size`: square side length (same unit as ray origins/directions)
///
/// # Returns
///
/// [`SquarePoseEstimate`] with pose in the ray coordinate frame, a confidence
/// score in \[0, 1\], residual error, and optimized positive ray distances.
///
/// # Errors
///
/// [`PnpError::SolverFailed`] for non-positive size, degenerate geometry, or
/// residual above the success threshold.
pub fn estimate_square_pose_from_rays(
    rays: [Ray3; 4],
    physical_size: f64,
) -> Result<SquarePoseEstimate, PnpError> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PnpError::SolverFailed);
    }

    let rays = normalize_rays(rays)?;
    let initial_distance = initial_ray_distance(&rays, physical_size);
    let max_distance = max_f64(initial_distance * MAX_DISTANCE_MULTIPLIER, physical_size);
    let mut distances = [initial_distance; DISTANCE_COUNT];
    let residuals = solve_distances(&rays, physical_size, max_distance, &mut distances)?;
    let normalized_corner_error = rms(&residuals);
    if !normalized_corner_error.is_finite() || normalized_corner_error > SUCCESS_RESIDUAL_RMS {
        return Err(PnpError::SolverFailed);
    }

    let points = points_from_distances(&rays, &distances)?;
    let pose = pose_from_points(&points)?;

    let confidence = 1.0 / (1.0 + normalized_corner_error);
    if !confidence.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    Ok(SquarePoseEstimate {
        pose,
        confidence: clamp01(confidence),
        normalized_corner_error,
        ray_distances: distances,
    })
}

/// Estimate square pose from **stereo corner pixels** (both eyes).
///
/// # Strategy (v1)
///
/// 1. Triangulate each of the four TL→TR→BR→BL correspondences with
///    [`triangulate_midpoint`] → 3D corners in the **left OpenCV** frame.
/// 2. Fit square pose from those points via the same axis construction as the
///    monocular ray path (`pose_from_points`).
/// 3. Score residual square constraints (edge/diagonal/orthogonality) for
///    confidence; convert the rigid pose to **OpenGL** (left camera) for the
///    public result, matching mono / stereo PnP conventions.
///
/// # Arguments
///
/// - `left_corners` / `right_corners`: TL, TR, BR, BL in each eye (OpenCV pixels)
/// - `physical_size`: square side length (same units as triangulation / baseline)
/// - `rig`: calibrated stereo pair (`right_from_left` in left OpenCV frame)
///
/// # Returns
///
/// [`SquarePoseEstimate`] with:
/// - `pose` in the **left OpenGL** camera frame
/// - `ray_distances` = distance from the left camera origin to each triangulated
///   corner (not monocular LM distances)
///
/// # Errors
///
/// [`PnpError::SolverFailed`] for non-positive size, failed triangulation,
/// degenerate geometry, or residual above the success threshold.
pub fn estimate_square_pose_from_stereo_pixels(
    left_corners: [Vector2; 4],
    right_corners: [Vector2; 4],
    physical_size: f64,
    rig: &StereoRig,
) -> Result<SquarePoseEstimate, PnpError> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PnpError::SolverFailed);
    }

    let mut points = [Vector3::new(0.0, 0.0, 0.0); 4];
    let mut ray_distances = [0.0; 4];
    for i in 0..DISTANCE_COUNT {
        let p = triangulate_midpoint(rig, left_corners[i], right_corners[i])?;
        let d = length(p);
        if !d.is_finite() || d <= 0.0 {
            return Err(PnpError::SolverFailed);
        }
        points[i] = p;
        ray_distances[i] = d;
    }

    let residuals = square_residuals_from_points(&points, physical_size)?;
    let normalized_corner_error = rms(&residuals);
    if !normalized_corner_error.is_finite() || normalized_corner_error > SUCCESS_RESIDUAL_RMS {
        return Err(PnpError::SolverFailed);
    }

    let pose_cv = pose_from_points(&points)?;
    let pose = pose_tools::from_opencv_to_opengl(&pose_cv);

    let confidence = 1.0 / (1.0 + normalized_corner_error);
    if !confidence.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    Ok(SquarePoseEstimate {
        pose,
        confidence: clamp01(confidence),
        normalized_corner_error,
        ray_distances,
    })
}

fn normalize_rays(mut rays: [Ray3; 4]) -> Result<[Ray3; 4], PnpError> {
    for ray in &mut rays {
        if !is_vector_finite(&ray.origin) || !is_vector_finite(&ray.direction) {
            return Err(PnpError::SolverFailed);
        }
        ray.direction = normalize(ray.direction).ok_or(PnpError::SolverFailed)?;
    }
    Ok(rays)
}

fn initial_ray_distance(rays: &[Ray3; 4], physical_size: f64) -> f64 {
    let fallback = max_f64(physical_size * 2.0, MIN_DISTANCE * 10.0);
    match (
        initial_guess_between(rays[1].direction, rays[3].direction, physical_size),
        initial_guess_between(rays[0].direction, rays[2].direction, physical_size),
    ) {
        (Some(a), Some(b)) => max_f64(min_f64(a, b), MIN_DISTANCE),
        (Some(a), None) => max_f64(a, MIN_DISTANCE),
        (None, Some(b)) => max_f64(b, MIN_DISTANCE),
        (None, None) => fallback,
    }
}

fn initial_guess_between(a: Vector3, b: Vector3, physical_size: f64) -> Option<f64> {
    let dot_value = clamp(dot(a, b), -1.0, 1.0);
    let angle = libm::acos(dot_value);
    if !angle.is_finite() || angle <= 1e-7 {
        return None;
    }

    let half_diagonal = physical_size * libm::sqrt(2.0) * 0.5;
    let tan_half_angle = libm::tan(angle * 0.5);
    if !tan_half_angle.is_finite() || tan_half_angle.abs() <= 1e-12 {
        return None;
    }

    let middle_distance = half_diagonal / tan_half_angle;
    let guess = libm::sqrt(middle_distance * middle_distance + half_diagonal * half_diagonal);
    if guess.is_finite() && guess > MIN_DISTANCE {
        Some(guess)
    } else {
        None
    }
}

fn solve_distances(
    rays: &[Ray3; 4],
    physical_size: f64,
    max_distance: f64,
    distances: &mut [f64; 4],
) -> Result<[f64; RESIDUAL_COUNT], PnpError> {
    if !max_distance.is_finite() || max_distance <= MIN_DISTANCE {
        return Err(PnpError::SolverFailed);
    }

    let mut residuals = residuals_for(rays, distances, physical_size)?;
    let mut current_objective = objective(&residuals);
    if !current_objective.is_finite() {
        return Err(PnpError::SolverFailed);
    }

    let mut damping = 1e-3;

    for _ in 0..MAX_ITERATIONS {
        if current_objective < 1e-24 {
            break;
        }

        let jacobian = numeric_jacobian(rays, distances, physical_size, &residuals)?;
        let (jtj, jtr) = normal_equations(&jacobian, &residuals);
        let mut accepted = false;

        for _ in 0..12 {
            let mut damped = jtj;
            for i in 0..DISTANCE_COUNT {
                damped[i][i] += damping * (jtj[i][i].abs() + 1.0);
            }

            let delta = match solve_linear4(damped, [-jtr[0], -jtr[1], -jtr[2], -jtr[3]]) {
                Some(delta) => delta,
                None => {
                    damping = min_f64(damping * 10.0, 1e12);
                    continue;
                }
            };

            if !is_array4_finite(&delta) {
                damping = min_f64(damping * 10.0, 1e12);
                continue;
            }

            let mut candidate = *distances;
            for i in 0..DISTANCE_COUNT {
                candidate[i] = clamp(candidate[i] + delta[i], MIN_DISTANCE, max_distance);
            }

            let candidate_residuals = match residuals_for(rays, &candidate, physical_size) {
                Ok(candidate_residuals) => candidate_residuals,
                Err(PnpError::SolverFailed) => {
                    damping = min_f64(damping * 10.0, 1e12);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let candidate_objective = objective(&candidate_residuals);
            if candidate_objective.is_finite() && candidate_objective < current_objective {
                let improvement = current_objective - candidate_objective;
                let step_norm = array4_norm(&delta);
                *distances = candidate;
                residuals = candidate_residuals;
                current_objective = candidate_objective;
                damping = max_f64(damping * 0.3, 1e-12);
                accepted = true;

                if (improvement < 1e-18 || step_norm < 1e-10)
                    && rms(&residuals) <= SUCCESS_RESIDUAL_RMS
                {
                    return Ok(residuals);
                }

                break;
            }

            damping = min_f64(damping * 10.0, 1e12);
        }

        if !accepted {
            if current_objective < 1e-16 {
                break;
            }
            return Err(PnpError::SolverFailed);
        }
    }

    Ok(residuals)
}

fn numeric_jacobian(
    rays: &[Ray3; 4],
    distances: &[f64; 4],
    physical_size: f64,
    base_residuals: &[f64; RESIDUAL_COUNT],
) -> Result<[[f64; DISTANCE_COUNT]; RESIDUAL_COUNT], PnpError> {
    let mut jacobian = [[0.0; DISTANCE_COUNT]; RESIDUAL_COUNT];

    for col in 0..DISTANCE_COUNT {
        let step = 1e-6 * max_f64(distances[col].abs(), 1.0);
        let mut plus = *distances;
        plus[col] += step;
        let plus_residuals = residuals_for(rays, &plus, physical_size)?;

        if distances[col] - step > MIN_DISTANCE {
            let mut minus = *distances;
            minus[col] -= step;
            let minus_residuals = residuals_for(rays, &minus, physical_size)?;
            let denom = 2.0 * step;
            for row in 0..RESIDUAL_COUNT {
                jacobian[row][col] = (plus_residuals[row] - minus_residuals[row]) / denom;
            }
        } else {
            for row in 0..RESIDUAL_COUNT {
                jacobian[row][col] = (plus_residuals[row] - base_residuals[row]) / step;
            }
        }
    }

    Ok(jacobian)
}

fn normal_equations(
    jacobian: &[[f64; DISTANCE_COUNT]; RESIDUAL_COUNT],
    residuals: &[f64; RESIDUAL_COUNT],
) -> (
    [[f64; DISTANCE_COUNT]; DISTANCE_COUNT],
    [f64; DISTANCE_COUNT],
) {
    let mut jtj = [[0.0; DISTANCE_COUNT]; DISTANCE_COUNT];
    let mut jtr = [0.0; DISTANCE_COUNT];

    for row in 0..RESIDUAL_COUNT {
        for col in 0..DISTANCE_COUNT {
            jtr[col] += jacobian[row][col] * residuals[row];
            for col2 in 0..DISTANCE_COUNT {
                jtj[col][col2] += jacobian[row][col] * jacobian[row][col2];
            }
        }
    }

    (jtj, jtr)
}

fn residuals_for(
    rays: &[Ray3; 4],
    distances: &[f64; 4],
    physical_size: f64,
) -> Result<[f64; RESIDUAL_COUNT], PnpError> {
    let points = points_from_distances(rays, distances)?;
    square_residuals_from_points(&points, physical_size)
}

/// Normalized square-geometry residuals for four ordered corners (TL,TR,BR,BL).
fn square_residuals_from_points(
    points: &[Vector3; 4],
    physical_size: f64,
) -> Result<[f64; RESIDUAL_COUNT], PnpError> {
    let diagonal = physical_size * libm::sqrt(2.0);
    let size2 = physical_size * physical_size;
    let size3 = size2 * physical_size;

    let edge01 = sub(points[1], points[0]);
    let edge12 = sub(points[2], points[1]);
    let edge32 = sub(points[3], points[2]);
    let edge03 = sub(points[0], points[3]);
    let diagonal02 = sub(points[2], points[0]);
    let diagonal13 = sub(points[3], points[1]);
    let left_edge_down = sub(points[3], points[0]);

    let residuals = [
        (length(edge01) - physical_size) / physical_size,
        (length(edge12) - physical_size) / physical_size,
        (length(edge32) - physical_size) / physical_size,
        (length(edge03) - physical_size) / physical_size,
        (length(diagonal02) - diagonal) / physical_size,
        (length(diagonal13) - diagonal) / physical_size,
        dot(edge01, left_edge_down) / size2,
        dot(edge12, edge32) / size2,
        dot(cross(edge01, left_edge_down), diagonal02) / size3,
    ];

    if residuals.iter().all(|value| value.is_finite()) {
        Ok(residuals)
    } else {
        Err(PnpError::SolverFailed)
    }
}

fn points_from_distances(rays: &[Ray3; 4], distances: &[f64; 4]) -> Result<[Vector3; 4], PnpError> {
    let mut points = [Vector3::new(0.0, 0.0, 0.0); 4];
    for i in 0..DISTANCE_COUNT {
        let distance = distances[i];
        if !distance.is_finite() || distance <= 0.0 {
            return Err(PnpError::SolverFailed);
        }
        points[i] = add(rays[i].origin, scale(rays[i].direction, distance));
        if !is_vector_finite(&points[i]) {
            return Err(PnpError::SolverFailed);
        }
    }
    Ok(points)
}

fn pose_from_points(points: &[Vector3; 4]) -> Result<Pose, PnpError> {
    let right = normalize(sub(points[1], points[0])).ok_or(PnpError::SolverFailed)?;
    let up_guess =
        normalize(scale(sub(points[3], points[0]), -1.0)).ok_or(PnpError::SolverFailed)?;
    let forward = normalize(cross(right, up_guess)).ok_or(PnpError::SolverFailed)?;
    let up = normalize(cross(forward, right)).ok_or(PnpError::SolverFailed)?;
    let forward = normalize(cross(right, up)).ok_or(PnpError::SolverFailed)?;
    let center = scale(add(points[1], points[3]), 0.5);

    let basis = Matrix3x3::new([
        right.x, right.y, right.z, up.x, up.y, up.z, forward.x, forward.y, forward.z,
    ]);
    let rotation = rotation_matrix_to_quaternion(&basis).normalize();
    if !is_vector_finite(&center)
        || !rotation.x.is_finite()
        || !rotation.y.is_finite()
        || !rotation.z.is_finite()
        || !rotation.w.is_finite()
    {
        return Err(PnpError::SolverFailed);
    }

    Ok(Pose::new(center, rotation))
}

fn solve_linear4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for pivot in 0..DISTANCE_COUNT {
        let mut pivot_row = pivot;
        let mut pivot_abs = a[pivot][pivot].abs();
        for (row, values) in a.iter().enumerate().skip(pivot + 1) {
            let candidate_abs = values[pivot].abs();
            if candidate_abs > pivot_abs {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if !pivot_abs.is_finite() || pivot_abs < 1e-12 {
            return None;
        }

        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }

        let pivot_value = a[pivot][pivot];
        for row in (pivot + 1)..DISTANCE_COUNT {
            let factor = a[row][pivot] / pivot_value;
            a[row][pivot] = 0.0;
            for col in (pivot + 1)..DISTANCE_COUNT {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    let mut x = [0.0; DISTANCE_COUNT];
    for row in (0..DISTANCE_COUNT).rev() {
        let mut sum = b[row];
        for (col, value) in x.iter().enumerate().skip(row + 1) {
            sum -= a[row][col] * value;
        }
        let diagonal = a[row][row];
        if !diagonal.is_finite() || diagonal.abs() < 1e-12 {
            return None;
        }
        x[row] = sum / diagonal;
        if !x[row].is_finite() {
            return None;
        }
    }

    Some(x)
}

fn objective(residuals: &[f64; RESIDUAL_COUNT]) -> f64 {
    residuals.iter().map(|value| value * value).sum()
}

fn rms(residuals: &[f64; RESIDUAL_COUNT]) -> f64 {
    libm::sqrt(objective(residuals) / RESIDUAL_COUNT as f64)
}

fn array4_norm(values: &[f64; 4]) -> f64 {
    libm::sqrt(values.iter().map(|value| value * value).sum())
}

fn is_array4_finite(values: &[f64; 4]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn is_vector_finite(v: &Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn normalize(v: Vector3) -> Option<Vector3> {
    let len = length(v);
    if !len.is_finite() || len <= MIN_VECTOR_LENGTH {
        None
    } else {
        Some(scale(v, 1.0 / len))
    }
}

fn add(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn scale(v: Vector3, s: f64) -> Vector3 {
    Vector3::new(v.x * s, v.y * s, v.z * s)
}

fn dot(a: Vector3, b: Vector3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn length(v: Vector3) -> f64 {
    libm::sqrt(dot(v, v))
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn clamp01(value: f64) -> f64 {
    clamp(value, 0.0, 1.0)
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}
