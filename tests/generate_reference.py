#!/usr/bin/env python3
"""Generate reference test vectors using OpenCV's solvePnP.

Produces JSON files with ground-truth rvec, tvec, rotation matrices,
quaternions, and final poses for cross-validating the Rust implementation.
"""

import json
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python>=4.8")
    sys.exit(1)

# ─── Test vectors from reference-code solve-pnp-test/index.html ───

LANDMARKS_3D = np.array([
    [-0.15, -0.15, 0.0],
    [ 0.15, -0.15, 0.0],
    [ 0.15,  0.15, 0.0],
    [-0.15,  0.15, 0.0],
], dtype=np.float64)

CAMERA_MATRIX = np.array([
    [815.8511, 0.0,      960.0],
    [0.0,      815.8511, 540.0],
    [0.0,      0.0,      1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

OBSERVATION_SETS = [
    np.array([
        [849.3577, 461.7641],
        [1070.642, 461.7641],
        [1096.898, 636.8014],
        [823.1021, 636.8014],
    ], dtype=np.float64),
    np.array([
        [1324.333, 208.0732],
        [1604.393, 129.3065],
        [1604.393, 403.1022],
        [1324.333, 429.3577],
    ], dtype=np.float64),
    np.array([
        [379.0064, 743.9628],
        [552.0745, 570.8946],
        [725.1426, 743.9628],
        [552.0745, 917.0309],
    ], dtype=np.float64),
]

# Methods to test (those we implement in Rust)
METHODS = {
    "epnp": cv2.SOLVEPNP_EPNP,
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "sqpnp": cv2.SOLVEPNP_SQPNP,
    "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
}


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w).

    Uses the same algorithm as nalgebra/GLM (Shepperd's method).
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    return [x/norm, y/norm, z/norm, w/norm]


def from_opencv_to_opengl(position, quaternion):
    """Convert pose from OpenCV to OpenGL coordinate system.

    Position: negate Y, Z
    Rotation: flipYZ (180° X-axis) * rotation
    """
    gl_position = [position[0], -position[1], -position[2]]

    # flipYZ quaternion: 180° about X = (1, 0, 0, 0) as (x, y, z, w)
    # sin(π/2) = 1, cos(π/2) = 0
    flip_x, flip_y, flip_z, flip_w = 1.0, 0.0, 0.0, 0.0

    # Hamilton product: flip * q
    qx, qy, qz, qw = quaternion
    gl_qx = flip_w * qx + flip_x * qw + flip_y * qz - flip_z * qy
    gl_qy = flip_w * qy - flip_x * qz + flip_y * qw + flip_z * qx
    gl_qz = flip_w * qz + flip_x * qy - flip_y * qx + flip_z * qw
    gl_qw = flip_w * qw - flip_x * qx - flip_y * qy - flip_z * qz

    return gl_position, [gl_qx, gl_qy, gl_qz, gl_qw]


def invert_pose(position, quaternion):
    """Invert a rigid-body pose: T_inv = (R^T, -R^T * t)."""
    # Quaternion to rotation matrix
    qx, qy, qz, qw = quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])

    R_inv = R.T
    pos = np.array(position)
    pos_inv = -R_inv @ pos

    q_inv = rotation_matrix_to_quaternion(R_inv)

    return pos_inv.tolist(), q_inv


def compute_reprojection_error(obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs):
    """Compute mean reprojection error."""
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - img_pts) ** 2, axis=1))
    return float(np.mean(errors))


def solve_for_set(set_idx, observations, method_name, method_flag):
    """Solve PnP for one observation set with one method."""
    obj_pts = LANDMARKS_3D.copy()
    img_pts = observations.copy()

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS,
        flags=method_flag
    )

    if not success:
        return None

    rvec = rvec.astype(np.float64).flatten()
    tvec = tvec.astype(np.float64).flatten()

    # Rodrigues: rvec → rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation_matrix = rotation_matrix.astype(np.float64)

    # Rotation matrix → quaternion
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    # Build OpenCV pose
    cv_position = tvec.tolist()
    cv_quaternion = quaternion

    # Convert to OpenGL
    gl_position, gl_quaternion = from_opencv_to_opengl(cv_position, cv_quaternion)

    # Invert to get camera pose
    cam_position, cam_quaternion = invert_pose(gl_position, gl_quaternion)

    # Reprojection error
    reproj_error = compute_reprojection_error(
        obj_pts, img_pts, rvec.reshape(3, 1), tvec.reshape(3, 1),
        CAMERA_MATRIX, DIST_COEFFS
    )

    return {
        "set_index": set_idx,
        "method": method_name,
        "success": True,
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "cv_quaternion": cv_quaternion,
        "cv_position": cv_position,
        "gl_position": gl_position,
        "gl_quaternion": gl_quaternion,
        "camera_position": cam_position,
        "camera_quaternion": cam_quaternion,
        "reprojection_error_px": reproj_error,
    }


def generate_edge_cases():
    """Generate additional edge-case test vectors."""
    edge_cases = []

    # 1. Non-coplanar: 6 cube corners
    cube_pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Known pose: slight rotation + translation
    rvec_true = np.array([0.1, -0.2, 0.15], dtype=np.float64)
    tvec_true = np.array([0.5, 0.3, 3.0], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        cube_pts, rvec_true.reshape(3, 1), tvec_true.reshape(3, 1),
        CAMERA_MATRIX, DIST_COEFFS
    )
    img_pts = projected.reshape(-1, 2)

    for method_name, method_flag in [("epnp", cv2.SOLVEPNP_EPNP), ("iterative", cv2.SOLVEPNP_ITERATIVE), ("sqpnp", cv2.SOLVEPNP_SQPNP)]:
        success, rvec, tvec = cv2.solvePnP(
            cube_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS, flags=method_flag
        )
        if success:
            rvec = rvec.astype(np.float64).flatten()
            tvec = tvec.astype(np.float64).flatten()
            reproj = compute_reprojection_error(
                cube_pts, img_pts, rvec.reshape(3, 1), tvec.reshape(3, 1),
                CAMERA_MATRIX, DIST_COEFFS
            )
            R, _ = cv2.Rodrigues(rvec)
            edge_cases.append({
                "name": f"noncoplanar_6pt_{method_name}",
                "object_points": cube_pts.tolist(),
                "image_points": img_pts.tolist(),
                "method": method_name,
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "rotation_matrix": R.tolist(),
                "true_rvec": rvec_true.tolist(),
                "true_tvec": tvec_true.tolist(),
                "reprojection_error_px": reproj,
            })

    # 2. Near-identity: camera at (0, 0, 1) looking at origin
    rvec_near_id = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tvec_near_id = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        LANDMARKS_3D, rvec_near_id.reshape(3, 1), tvec_near_id.reshape(3, 1),
        CAMERA_MATRIX, DIST_COEFFS
    )
    img_near_id = projected.reshape(-1, 2)

    for method_name, method_flag in [("epnp", cv2.SOLVEPNP_EPNP), ("iterative", cv2.SOLVEPNP_ITERATIVE), ("sqpnp", cv2.SOLVEPNP_SQPNP)]:
        success, rvec, tvec = cv2.solvePnP(
            LANDMARKS_3D, img_near_id, CAMERA_MATRIX, DIST_COEFFS, flags=method_flag
        )
        if success:
            rvec = rvec.astype(np.float64).flatten()
            tvec = tvec.astype(np.float64).flatten()
            reproj = compute_reprojection_error(
                LANDMARKS_3D, img_near_id, rvec.reshape(3, 1), tvec.reshape(3, 1),
                CAMERA_MATRIX, DIST_COEFFS
            )
            R, _ = cv2.Rodrigues(rvec)
            edge_cases.append({
                "name": f"near_identity_{method_name}",
                "object_points": LANDMARKS_3D.tolist(),
                "image_points": img_near_id.tolist(),
                "method": method_name,
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "rotation_matrix": R.tolist(),
                "true_rvec": rvec_near_id.tolist(),
                "true_tvec": tvec_near_id.tolist(),
                "reprojection_error_px": reproj,
            })

    # 3. Large rotation (170°)
    angle = np.radians(170)
    axis = np.array([0.577, 0.577, 0.577])  # ~(1,1,1)/sqrt(3)
    axis = axis / np.linalg.norm(axis)
    rvec_large = (axis * angle).astype(np.float64)
    tvec_large = np.array([0.0, 0.0, 2.0], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        LANDMARKS_3D, rvec_large.reshape(3, 1), tvec_large.reshape(3, 1),
        CAMERA_MATRIX, DIST_COEFFS
    )
    img_large = projected.reshape(-1, 2)

    for method_name, method_flag in [("epnp", cv2.SOLVEPNP_EPNP), ("iterative", cv2.SOLVEPNP_ITERATIVE), ("sqpnp", cv2.SOLVEPNP_SQPNP)]:
        try:
            success, rvec, tvec = cv2.solvePnP(
                LANDMARKS_3D, img_large, CAMERA_MATRIX, DIST_COEFFS, flags=method_flag
            )
            if success:
                rvec = rvec.astype(np.float64).flatten()
                tvec = tvec.astype(np.float64).flatten()
                reproj = compute_reprojection_error(
                    LANDMARKS_3D, img_large, rvec.reshape(3, 1), tvec.reshape(3, 1),
                    CAMERA_MATRIX, DIST_COEFFS
                )
                R, _ = cv2.Rodrigues(rvec)
                edge_cases.append({
                    "name": f"large_rotation_170deg_{method_name}",
                    "object_points": LANDMARKS_3D.tolist(),
                    "image_points": img_large.tolist(),
                    "method": method_name,
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                    "rotation_matrix": R.tolist(),
                    "true_rvec": rvec_large.tolist(),
                    "true_tvec": tvec_large.tolist(),
                    "reprojection_error_px": reproj,
                })
        except cv2.error:
            pass

    # 4. Large point count (50 random 3D points)
    np.random.seed(42)
    pts_50 = np.random.uniform(-1, 1, (50, 3)).astype(np.float64)
    pts_50[:, 2] = np.abs(pts_50[:, 2]) * 0.5  # keep z positive but varied

    rvec_50 = np.array([0.3, -0.4, 0.2], dtype=np.float64)
    tvec_50 = np.array([0.1, -0.2, 3.0], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        pts_50, rvec_50.reshape(3, 1), tvec_50.reshape(3, 1),
        CAMERA_MATRIX, DIST_COEFFS
    )
    img_50 = projected.reshape(-1, 2)

    for method_name, method_flag in [("epnp", cv2.SOLVEPNP_EPNP), ("iterative", cv2.SOLVEPNP_ITERATIVE), ("sqpnp", cv2.SOLVEPNP_SQPNP)]:
        success, rvec, tvec = cv2.solvePnP(
            pts_50, img_50, CAMERA_MATRIX, DIST_COEFFS, flags=method_flag
        )
        if success:
            rvec = rvec.astype(np.float64).flatten()
            tvec = tvec.astype(np.float64).flatten()
            reproj = compute_reprojection_error(
                pts_50, img_50, rvec.reshape(3, 1), tvec.reshape(3, 1),
                CAMERA_MATRIX, DIST_COEFFS
            )
            R, _ = cv2.Rodrigues(rvec)
            edge_cases.append({
                "name": f"large_50pt_{method_name}",
                "object_points": pts_50.tolist(),
                "image_points": img_50.tolist(),
                "method": method_name,
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "rotation_matrix": R.tolist(),
                "true_rvec": rvec_50.tolist(),
                "true_tvec": tvec_50.tolist(),
                "reprojection_error_px": reproj,
            })

    return edge_cases


def main():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

    # ─── Main reference output ───
    results = []
    for set_idx, observations in enumerate(OBSERVATION_SETS):
        for method_name, method_flag in METHODS.items():
            result = solve_for_set(set_idx, observations, method_name, method_flag)
            if result:
                results.append(result)
                print(f"  Set {set_idx} / {method_name}: reproj={result['reprojection_error_px']:.6f}px")
            else:
                print(f"  Set {set_idx} / {method_name}: FAILED")

    output = {
        "metadata": {
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "precision": "float64",
        },
        "landmarks": LANDMARKS_3D.tolist(),
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "observation_sets": [obs.tolist() for obs in OBSERVATION_SETS],
        "results": results,
    }

    with open("tests/reference_vectors/reference_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(results)} results to tests/reference_vectors/reference_output.json")

    # ─── Edge cases ───
    edge_cases = generate_edge_cases()

    edge_output = {
        "metadata": {
            "opencv_version": cv2.__version__,
            "precision": "float64",
        },
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "cases": edge_cases,
    }

    with open("tests/reference_vectors/edge_cases.json", "w") as f:
        json.dump(edge_output, f, indent=2)
    print(f"Wrote {len(edge_cases)} edge cases to tests/reference_vectors/edge_cases.json")

    # Verify reprojection errors
    for r in results:
        if r["reprojection_error_px"] > 1.0:
            print(f"WARNING: High reprojection error for {r['method']} set {r['set_index']}: {r['reprojection_error_px']:.4f}px")

    for e in edge_cases:
        if e["reprojection_error_px"] > 1.0:
            print(f"WARNING: High reprojection error for edge case {e['name']}: {e['reprojection_error_px']:.4f}px")


if __name__ == "__main__":
    main()
