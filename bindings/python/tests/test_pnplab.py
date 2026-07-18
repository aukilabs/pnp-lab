from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import auki_pnplab

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = json.loads(
    (ROOT / "tests/reference_vectors/reference_output.json").read_text()
)


def _quaternion_angle(q1: dict[str, float], q2: list[float] | tuple[float, ...]) -> float:
    dot = abs(
        q1["x"] * q2[0] + q1["y"] * q2[1] + q1["z"] * q2[2] + q1["w"] * q2[3]
    )
    return 2.0 * math.acos(min(dot, 1.0))


def _position_error(p1: dict[str, float], p2: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(
        (p1["x"] - p2[0]) ** 2 + (p1["y"] - p2[1]) ** 2 + (p1["z"] - p2[2]) ** 2
    )


def _landmarks() -> np.ndarray:
    return np.asarray(REFERENCE["landmarks"], dtype=np.float64)


def _observations(set_index: int) -> np.ndarray:
    return np.asarray(REFERENCE["observation_sets"][set_index], dtype=np.float64)


def _camera() -> dict[str, float | list[float]]:
    k = REFERENCE["camera_matrix"]
    return {
        "fx": k[0][0],
        "fy": k[1][1],
        "cx": k[0][2],
        "cy": k[1][2],
        "dist": [],
    }


def _camera_matrix() -> np.ndarray:
    """Pinhole OpenCV K matrix (accepted as a convenience camera form)."""
    return np.asarray(REFERENCE["camera_matrix"], dtype=np.float64)


def _reference_result(set_index: int, method: str) -> dict:
    return next(
        r
        for r in REFERENCE["results"]
        if r["set_index"] == set_index and r["method"] == method
    )


def test_version_is_exposed() -> None:
    assert isinstance(auki_pnplab.__version__, str)
    assert auki_pnplab.__version__


def test_solve_pnp_matches_reference_iterative_set0() -> None:
    ref = _reference_result(0, "iterative")
    pose = auki_pnplab.solve_pnp(
        _landmarks(),
        _observations(0),
        _camera(),
        method="iterative",
    )

    assert _position_error(pose["position"], ref["gl_position"]) < 1e-3
    assert _quaternion_angle(pose["rotation"], ref["gl_quaternion"]) < 1e-3


def test_solve_pnp_camera_pose_matches_reference() -> None:
    ref = _reference_result(0, "iterative")
    camera_pose = auki_pnplab.solve_pnp_camera_pose(
        _landmarks(),
        _observations(0),
        _camera(),
        method="iterative",
    )

    assert _position_error(camera_pose["position"], ref["camera_position"]) < 1e-3
    assert (
        _quaternion_angle(camera_pose["rotation"], ref["camera_quaternion"]) < 1e-3
    )


def test_dict_landmarks_and_camera_forms() -> None:
    landmarks = [
        {"id": str(i), "position": {"x": p[0], "y": p[1], "z": p[2]}}
        for i, p in enumerate(REFERENCE["landmarks"])
    ]
    observations = [
        {"id": str(i), "position": {"x": p[0], "y": p[1]}}
        for i, p in enumerate(REFERENCE["observation_sets"][0])
    ]
    ref = _reference_result(0, "iterative")

    for camera in (
        _camera(),
        _camera_matrix(),
        {"m": [815.8511, 0.0, 0.0, 0.0, 815.8511, 0.0, 960.0, 540.0, 1.0]},
    ):
        pose = auki_pnplab.solve_pnp(landmarks, observations, camera, method="iterative")
        assert _position_error(pose["position"], ref["gl_position"]) < 1e-3


def test_solve_pnp_with_distortion_coeffs() -> None:
    # Mild distortion should still solve after internal undistort.
    camera = {
        **_camera(),
        "dist": [0.05, -0.02, 0.0, 0.0, 0.0],
    }
    pose = auki_pnplab.solve_pnp(
        _landmarks(),
        _observations(0),
        camera,
        method="iterative",
    )
    assert math.isfinite(pose["position"]["x"])
    assert math.isfinite(pose["rotation"]["w"])


def test_camera_pose_from_solve_pnp_pose_inverts_translation() -> None:
    pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    camera = auki_pnplab.camera_pose_from_solve_pnp_pose(pose)
    assert camera["position"]["x"] == pytest.approx(-1.0)
    assert camera["position"]["y"] == pytest.approx(-2.0)
    assert camera["position"]["z"] == pytest.approx(-3.0)
    assert camera["rotation"]["w"] == pytest.approx(1.0)


def test_estimate_square_pose_from_rays_synthetic() -> None:
    physical_size = 0.8
    half = physical_size * 0.5
    center = np.array([0.35, 0.2, 2.6], dtype=np.float64)
    camera_origin = np.array([-0.25, -0.15, -1.2], dtype=np.float64)

    yaw = 0.37
    pitch = -0.24
    right = np.array([math.cos(yaw), math.sin(pitch), -math.sin(yaw)], dtype=np.float64)
    right /= np.linalg.norm(right)
    forward_seed = np.array([math.sin(yaw), 0.35, math.cos(yaw)], dtype=np.float64)
    forward_seed /= np.linalg.norm(forward_seed)
    up = np.cross(forward_seed, right)
    up /= np.linalg.norm(up)

    corners = [
        center + (-half) * right + half * up,
        center + half * right + half * up,
        center + half * right + (-half) * up,
        center + (-half) * right + (-half) * up,
    ]
    rays = [
        {
            "origin": {"x": float(camera_origin[0]), "y": float(camera_origin[1]), "z": float(camera_origin[2])},
            "direction": {
                "x": float(corner[0] - camera_origin[0]),
                "y": float(corner[1] - camera_origin[1]),
                "z": float(corner[2] - camera_origin[2]),
            },
        }
        for corner in corners
    ]

    estimate = auki_pnplab.estimate_square_pose_from_rays(rays, physical_size)
    assert estimate["confidence"] > 0.95
    assert abs(estimate["pose"]["position"]["x"] - center[0]) < 1e-4
    assert abs(estimate["pose"]["position"]["y"] - center[1]) < 1e-4
    assert abs(estimate["pose"]["position"]["z"] - center[2]) < 1e-4
    assert len(estimate["ray_distances"]) == 4
    assert all(d > 0.0 for d in estimate["ray_distances"])


def test_invalid_inputs_raise_python_exceptions() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        auki_pnplab.solve_pnp(
            _landmarks(),
            _observations(0),
            _camera(),
            method="unknown",
        )
    with pytest.raises(ValueError, match="insufficient points"):
        auki_pnplab.solve_pnp(
            _landmarks()[:2],
            _observations(0)[:2],
            _camera(),
            method="iterative",
        )
    with pytest.raises(ValueError, match="physical_size"):
        auki_pnplab.estimate_square_pose_from_rays(
            [
                {"origin": [0, 0, 0], "direction": [0, 0, 1]},
                {"origin": [0, 0, 0], "direction": [0, 0, 1]},
                {"origin": [0, 0, 0], "direction": [0, 0, 1]},
                {"origin": [0, 0, 0], "direction": [0, 0, 1]},
            ],
            0.0,
        )
    with pytest.raises(ValueError, match="exactly 4 rays"):
        auki_pnplab.estimate_square_pose_from_rays(
            [{"origin": [0, 0, 0], "direction": [0, 0, 1]}],
            0.8,
        )
