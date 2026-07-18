"""Stereo PnP and triangulation tests for auki_pnplab."""

from __future__ import annotations

import math

import numpy as np
import pytest

import auki_pnplab


def _identity_quat() -> dict[str, float]:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _rig(baseline: float = 0.12, fx: float = 800.0, cx: float = 320.0, cy: float = 240.0) -> dict:
    cam = {"fx": fx, "fy": fx, "cx": cx, "cy": cy, "dist": []}
    return {
        "left": cam,
        "right": dict(cam),
        "right_from_left": {
            "position": {"x": baseline, "y": 0.0, "z": 0.0},
            "rotation": _identity_quat(),
        },
    }


def _square_landmarks() -> list[dict]:
    corners = [(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, 0.1, 0.0)]
    return [
        {"id": str(i), "position": {"x": c[0], "y": c[1], "z": c[2]}}
        for i, c in enumerate(corners)
    ]


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle → 3×3 rotation matrix (OpenCV convention)."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-15:
        return np.eye(3)
    k = rvec / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _rot_to_quat(R: np.ndarray) -> dict[str, float]:
    """Rotation matrix → xyzw quaternion."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


def _cv_pose(rvec: list[float], t: list[float]) -> tuple[np.ndarray, np.ndarray, dict]:
    R = _rodrigues(np.asarray(rvec, dtype=np.float64))
    tvec = np.asarray(t, dtype=np.float64)
    return R, tvec, {
        "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "rotation": _rot_to_quat(R),
    }


def _quat_multiply(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Hamilton product a*b with xyzw dicts."""
    ax, ay, az, aw = a["x"], a["y"], a["z"], a["w"]
    bx, by, bz, bw = b["x"], b["y"], b["z"], b["w"]
    return {
        "x": aw * bx + ax * bw + ay * bz - az * by,
        "y": aw * by - ax * bz + ay * bw + az * bx,
        "z": aw * bz + ax * by - ay * bx + az * bw,
        "w": aw * bw - ax * bx - ay * by - az * bz,
    }


def _from_opencv_to_opengl(R: np.ndarray, t: np.ndarray) -> dict:
    """Mirror pose_tools::from_opencv_to_opengl (position Y/Z flip + 180° X quat)."""
    cv_q = _rot_to_quat(R)
    # 180° about X: (x, y, z, w) = (1, 0, 0, 0)
    flip_yz = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
    return {
        "position": {"x": float(t[0]), "y": float(-t[1]), "z": float(-t[2])},
        "rotation": _quat_multiply(flip_yz, cv_q),
    }


def _project_stereo(
    landmarks: list[dict],
    rig: dict,
    R: np.ndarray,
    t: np.ndarray,
    *,
    left_only: bool = False,
) -> list[dict]:
    """Project using the same convention as core stereo_solve tests.

    ``X_right = R_rl * X_left + t_rl`` via ``transform_point(right_from_left, ...)``.
    """
    fx = rig["left"]["fx"]
    cx = rig["left"]["cx"]
    cy = rig["left"]["cy"]
    t_rl = np.array(
        [
            rig["right_from_left"]["position"]["x"],
            rig["right_from_left"]["position"]["y"],
            rig["right_from_left"]["position"]["z"],
        ],
        dtype=np.float64,
    )
    obs = []
    for lm in landmarks:
        p = np.array(
            [lm["position"]["x"], lm["position"]["y"], lm["position"]["z"]],
            dtype=np.float64,
        )
        x_left = R @ p + t
        ul = fx * x_left[0] / x_left[2] + cx
        vl = fx * x_left[1] / x_left[2] + cy
        # Identity rotation on right_from_left → X_right = X_left + t_rl
        x_right = x_left + t_rl
        ur = fx * x_right[0] / x_right[2] + cx
        vr = fx * x_right[1] / x_right[2] + cy
        entry: dict = {"id": lm["id"], "left": [float(ul), float(vl)]}
        if not left_only:
            entry["right"] = [float(ur), float(vr)]
        else:
            entry["right"] = None
        obs.append(entry)
    return obs


def _position_error(p1: dict[str, float], p2: dict[str, float]) -> float:
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2
    )


def _quaternion_angle(q1: dict[str, float], q2: dict[str, float]) -> float:
    dot = abs(
        q1["x"] * q2["x"] + q1["y"] * q2["y"] + q1["z"] * q2["z"] + q1["w"] * q2["w"]
    )
    return 2.0 * math.acos(min(dot, 1.0))


def test_triangulate_known_point() -> None:
    # Same fixture as core triangulate_known_point_in_front.
    rig = _rig(baseline=0.1, fx=500.0, cx=320.0, cy=240.0)
    left = [320.0, 240.0]
    right = [500.0 * (-0.1) / 2.0 + 320.0, 240.0]
    pt = auki_pnplab.triangulate(left, right, rig)
    assert abs(pt["x"] - 0.0) < 1e-6
    assert abs(pt["y"] - 0.0) < 1e-6
    assert abs(pt["z"] - 2.0) < 1e-6


def test_triangulate_rejects_parallel_rays() -> None:
    rig = _rig(baseline=0.1, fx=500.0)
    with pytest.raises(RuntimeError, match="solver failed"):
        auki_pnplab.triangulate([320.0, 240.0], [320.0, 240.0], rig)


def test_solve_pnp_stereo_recovers_known_pose() -> None:
    landmarks = _square_landmarks()
    rig = _rig()
    R, t, _ = _cv_pose([0.05, -0.15, 0.08], [0.02, -0.01, 1.5])
    gl_true = _from_opencv_to_opengl(R, t)
    obs = _project_stereo(landmarks, rig, R, t)

    pose = auki_pnplab.solve_pnp_stereo(landmarks, obs, rig, method="iterative")
    assert _position_error(pose["position"], gl_true["position"]) < 1e-3
    assert _quaternion_angle(pose["rotation"], gl_true["rotation"]) < 1e-3


def test_solve_pnp_stereo_left_only() -> None:
    landmarks = _square_landmarks()
    rig = _rig()
    R, t, _ = _cv_pose([0.0, 0.2, 0.0], [0.0, 0.0, 1.2])
    gl_true = _from_opencv_to_opengl(R, t)
    obs = _project_stereo(landmarks, rig, R, t, left_only=True)

    pose = auki_pnplab.solve_pnp_stereo(landmarks, obs, rig, method="iterative")
    assert _position_error(pose["position"], gl_true["position"]) < 1e-3
    assert _quaternion_angle(pose["rotation"], gl_true["rotation"]) < 1e-3


def test_solve_pnp_stereo_camera_pose_is_inverse() -> None:
    landmarks = _square_landmarks()
    rig = _rig()
    R, t, _ = _cv_pose([0.05, -0.15, 0.08], [0.02, -0.01, 1.5])
    obs = _project_stereo(landmarks, rig, R, t)

    obj = auki_pnplab.solve_pnp_stereo(landmarks, obs, rig, method="sqpnp")
    cam = auki_pnplab.solve_pnp_stereo_camera_pose(landmarks, obs, rig, method="sqpnp")
    back = auki_pnplab.camera_pose_from_solve_pnp_pose(cam)
    assert _position_error(obj["position"], back["position"]) < 1e-6
    assert _quaternion_angle(obj["rotation"], back["rotation"]) < 1e-6


def test_solve_pnp_stereo_mismatched_ids() -> None:
    landmarks = _square_landmarks()
    rig = _rig()
    obs = [
        {"id": "0", "left": [100.0, 100.0], "right": [90.0, 100.0]},
        {"id": "1", "left": [200.0, 100.0], "right": None},
        {"id": "2", "left": [200.0, 200.0], "right": None},
        {"id": "99", "left": [100.0, 200.0], "right": None},
    ]
    with pytest.raises(ValueError, match="do not match"):
        auki_pnplab.solve_pnp_stereo(landmarks, obs, rig, method="epnp")


def test_solve_pnp_stereo_invalid_rig() -> None:
    landmarks = _square_landmarks()
    obs = [
        {"id": "0", "left": [100.0, 100.0], "right": [90.0, 100.0]},
        {"id": "1", "left": [200.0, 100.0], "right": [190.0, 100.0]},
        {"id": "2", "left": [200.0, 200.0], "right": [190.0, 200.0]},
        {"id": "3", "left": [100.0, 200.0], "right": [90.0, 200.0]},
    ]
    bad_rig = {
        "left": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
        "right": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
        "right_from_left": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": _identity_quat(),
        },
    }
    with pytest.raises(RuntimeError, match="solver failed"):
        auki_pnplab.solve_pnp_stereo(landmarks, obs, bad_rig)
