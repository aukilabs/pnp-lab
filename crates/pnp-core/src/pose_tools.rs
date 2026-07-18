//! Rigid-pose helpers: OpenCV ↔ OpenGL frame conversion and inversion.

use crate::types::{Pose, Quaternion, Vector3};
#[allow(unused_imports)]
use nalgebra;
use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion, Vector3 as NaVector3}; // for Unit::new_normalize

/// Convert a pose from the **OpenCV** camera frame to **OpenGL**.
///
/// | Frame | Axes |
/// |-------|------|
/// | OpenCV | Y-down, Z-forward |
/// | OpenGL | Y-up, Z-backward |
///
/// Position: negate Y and Z. Rotation: compose with a 180° rotation about X
/// (`flipYZ * R_cv`).
pub fn from_opencv_to_opengl(pose: &Pose) -> Pose {
    let position = Vector3::new(pose.position.x, -pose.position.y, -pose.position.z);

    // 180° rotation around X-axis: quaternion = (sin(π/2), 0, 0, cos(π/2)) = (1, 0, 0, 0)
    let flip_yz = UnitQuaternion::from_axis_angle(
        &nalgebra::Unit::new_normalize(NaVector3::new(1.0, 0.0, 0.0)),
        core::f64::consts::PI,
    );

    // Compose: flipYZ * cvRotation
    let cv_quat = pose.rotation.to_na_unit();
    let gl_rotation = flip_yz * cv_quat;

    let q = gl_rotation.quaternion();
    let rotation = Quaternion::new(q.i, q.j, q.k, q.w);

    Pose::new(position, rotation)
}

/// Convert a pose from **OpenGL** to **OpenCV**.
///
/// Self-inverse of [`from_opencv_to_opengl`].
pub fn from_opengl_to_opencv(pose: &Pose) -> Pose {
    from_opencv_to_opengl(pose)
}

/// Invert a rigid-body pose `T = (R, t)` as `T⁻¹ = (Rᵀ, −Rᵀ t)`.
///
/// Converts object-in-camera to camera-in-world (and vice versa).
pub fn invert_pose(pose: &Pose) -> Pose {
    let pos = NaVector3::new(pose.position.x, pose.position.y, pose.position.z);

    let rq = UnitQuaternion::from_quaternion(NaQuaternion::new(
        pose.rotation.w,
        pose.rotation.x,
        pose.rotation.y,
        pose.rotation.z,
    ));
    let r_mat = rq.to_rotation_matrix();

    let r_mat_inv = r_mat.transpose();
    let pos_inv = -(r_mat_inv * pos);

    let rot_inv = UnitQuaternion::from_rotation_matrix(&r_mat_inv);
    let q = rot_inv.quaternion();

    Pose::new(
        Vector3::new(pos_inv.x, pos_inv.y, pos_inv.z),
        Quaternion::new(q.i, q.j, q.k, q.w),
    )
}

/// Apply rigid transform: `R * p + t` using pose rotation as a unit quaternion.
///
/// OpenCV/OpenGL-agnostic math on the pose components as stored. Stereo
/// internal math stores `right_from_left` in the **OpenCV** camera frame.
pub fn transform_point(pose: &Pose, p: Vector3) -> Vector3 {
    let uq = pose.rotation.normalize().to_na_unit();
    let rotated = uq * nalgebra::Vector3::new(p.x, p.y, p.z);
    Vector3::new(
        rotated.x + pose.position.x,
        rotated.y + pose.position.y,
        rotated.z + pose.position.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn quaternion_angle_between(a: &Quaternion, b: &Quaternion) -> f64 {
        let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        2.0 * libm::acos(libm::fabs(dot).min(1.0))
    }

    fn poses_close(a: &Pose, b: &Pose, pos_tol: f64, rot_tol: f64) -> bool {
        let dp = Vector3::new(
            a.position.x - b.position.x,
            a.position.y - b.position.y,
            a.position.z - b.position.z,
        );
        if dp.length() > pos_tol {
            return false;
        }
        quaternion_angle_between(&a.rotation, &b.rotation) < rot_tol
    }

    #[test]
    fn test_opencv_to_opengl_identity() {
        let pose = Pose::identity();
        let result = from_opencv_to_opengl(&pose);
        // Position should remain at origin
        assert!(result.position.length() < EPS);
        // Rotation: flipYZ * identity = 180° about X
        // Expected quaternion: (1, 0, 0, 0) — 180° about X
        let angle =
            quaternion_angle_between(&result.rotation, &Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert!(
            angle < EPS,
            "expected 180° X rotation, angle diff = {}",
            angle
        );
    }

    #[test]
    fn test_opencv_to_opengl_self_inverse() {
        let pose = Pose::new(
            Vector3::new(1.0, 2.0, 3.0),
            Quaternion::new(0.1, 0.2, 0.3, 0.9).normalize(),
        );
        let once = from_opencv_to_opengl(&pose);
        let twice = from_opencv_to_opengl(&once);
        assert!(
            poses_close(&pose, &twice, 1e-12, 1e-12),
            "applying twice should return to original"
        );
    }

    #[test]
    fn test_opengl_to_opencv_is_same() {
        let pose = Pose::new(
            Vector3::new(-0.5, 1.2, -3.4),
            Quaternion::new(0.3, -0.4, 0.1, 0.85).normalize(),
        );
        let a = from_opencv_to_opengl(&pose);
        let b = from_opengl_to_opencv(&pose);
        assert!(poses_close(&a, &b, 1e-12, 1e-12));
    }

    #[test]
    fn test_invert_pose_identity() {
        let pose = Pose::identity();
        let inv = invert_pose(&pose);
        assert!(poses_close(&inv, &Pose::identity(), EPS, EPS));
    }

    #[test]
    fn test_invert_pose_roundtrip() {
        let pose = Pose::new(
            Vector3::new(1.0, -2.0, 3.0),
            Quaternion::new(0.2, -0.3, 0.5, 0.78).normalize(),
        );
        let inv = invert_pose(&pose);
        let back = invert_pose(&inv);
        assert!(
            poses_close(&pose, &back, 1e-12, 1e-12),
            "invert(invert(p)) should == p"
        );
    }

    #[test]
    fn test_invert_pose_known() {
        // Position = (1, 2, 3), rotation = 90° about Y
        let half = core::f64::consts::FRAC_PI_4;
        let s = libm::sin(half);
        let c = libm::cos(half);
        let pose = Pose::new(Vector3::new(1.0, 2.0, 3.0), Quaternion::new(0.0, s, 0.0, c));
        let inv = invert_pose(&pose);

        // R(90°Y) = [[0,0,1],[0,1,0],[-1,0,0]]
        // R^T = [[0,0,-1],[0,1,0],[1,0,0]]
        // pos_inv = -R^T * [1,2,3] = -[0*1+0*2+(-1)*3, 0*1+1*2+0*3, 1*1+0*2+0*3]
        //         = -[-3, 2, 1] = [3, -2, -1]
        assert!((inv.position.x - 3.0).abs() < 1e-10);
        assert!((inv.position.y - (-2.0)).abs() < 1e-10);
        assert!((inv.position.z - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_against_reference_pipeline() {
        // Test the full pipeline: rvec,tvec → Pose → from_opencv_to_opengl
        // Using a synthetic known case
        use crate::rodrigues::rvec_to_rotation_matrix;
        use crate::types::rotation_matrix_to_quaternion;

        let rvec = [0.1, -0.2, 0.3];
        let tvec = [0.5, -0.3, 1.0];

        // Step 1: rvec → rotation matrix
        let rot_mat = rvec_to_rotation_matrix(&rvec);

        // Step 2: create pose
        let q = rotation_matrix_to_quaternion(&rot_mat);
        let pose = Pose::new(Vector3::new(tvec[0], tvec[1], tvec[2]), q);

        // Step 3: convert to OpenGL
        let gl_pose = from_opencv_to_opengl(&pose);

        // Verify position conversion
        assert!((gl_pose.position.x - tvec[0]).abs() < EPS);
        assert!((gl_pose.position.y - (-tvec[1])).abs() < EPS);
        assert!((gl_pose.position.z - (-tvec[2])).abs() < EPS);

        // Verify rotation is valid quaternion
        let q_norm = gl_pose.rotation.norm();
        assert!((q_norm - 1.0).abs() < 1e-10);

        // Verify self-inverse property (quaternion composition introduces ~3e-8 rad fp error)
        let back = from_opencv_to_opengl(&gl_pose);
        assert!(
            poses_close(&pose, &back, 1e-10, 1e-6),
            "position diff: {:?} vs {:?}, angle: {}",
            pose.position,
            back.position,
            quaternion_angle_between(&pose.rotation, &back.rotation),
        );
    }

    #[test]
    fn transform_point_translates() {
        let pose = Pose::new(Vector3::new(1.0, 2.0, 3.0), Quaternion::identity());
        let out = transform_point(&pose, Vector3::new(0.0, 0.0, 0.0));
        assert!((out.x - 1.0).abs() < 1e-12);
    }
}
