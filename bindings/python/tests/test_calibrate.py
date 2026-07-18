"""Multi-view monocular camera calibration tests for auki_pnpkit."""

from __future__ import annotations

import math

import numpy as np
import pytest

import auki_pnpkit


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle → 3×3 rotation matrix (OpenCV convention)."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-15:
        return np.eye(3)
    k = rvec / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _square_object_points(physical_size: float) -> np.ndarray:
    """TL→TR→BR→BL centered square on Z=0 (matches pnp-core)."""
    h = physical_size / 2.0
    return np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float64,
    )


def _project_pinhole(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project OpenCV-frame object points: Xc = R X + t."""
    out = np.zeros((points.shape[0], 2), dtype=np.float64)
    for i, p in enumerate(points):
        pc = R @ p + t
        assert pc[2] > 1e-9, "point behind camera"
        out[i, 0] = fx * pc[0] / pc[2] + cx
        out[i, 1] = fy * pc[1] / pc[2] + cy
    return out


# Diverse OpenCV rvec/tvec poses matching core calibrate_from_square_views test.
_TRUE_POSES: list[tuple[list[float], list[float]]] = [
    ([0.15, -0.10, 0.05], [0.02, -0.01, 0.55]),
    ([-0.20, 0.18, -0.08], [-0.03, 0.02, 0.62]),
    ([0.10, 0.25, 0.12], [0.01, 0.0, 0.48]),
    ([0.30, -0.05, -0.15], [-0.02, 0.03, 0.70]),
    ([-0.12, -0.22, 0.08], [0.04, -0.02, 0.58]),
    ([0.05, 0.12, -0.20], [0.0, 0.01, 0.52]),
    ([0.28, 0.15, 0.10], [-0.01, 0.02, 0.60]),
    ([-0.25, -0.15, -0.05], [0.03, -0.01, 0.65]),
    ([0.18, -0.28, 0.15], [-0.02, 0.0, 0.50]),
    ([-0.08, 0.30, -0.12], [0.01, 0.03, 0.57]),
]


def _synthetic_square_corners(
    *,
    fx: float = 800.0,
    fy: float = 800.0,
    cx: float = 320.0,
    cy: float = 240.0,
    physical_size: float = 0.2,
) -> np.ndarray:
    object_pts = _square_object_points(physical_size)
    corners = []
    for rvec, tvec in _TRUE_POSES:
        R = _rodrigues(np.asarray(rvec, dtype=np.float64))
        t = np.asarray(tvec, dtype=np.float64)
        corners.append(_project_pinhole(object_pts, R, t, fx, fy, cx, cy))
    return np.stack(corners, axis=0)  # (N, 4, 2)


def test_calibrate_from_square_views_noise_free():
    fx_true, fy_true, cx_true, cy_true = 800.0, 800.0, 320.0, 240.0
    physical_size = 0.2
    corners = _synthetic_square_corners(
        fx=fx_true, fy=fy_true, cx=cx_true, cy=cy_true, physical_size=physical_size
    )

    result = auki_pnpkit.calibrate_from_square_views(
        corners,
        physical_size=physical_size,
        image_size=(640, 480),
        fix_aspect_ratio=True,
        dist_len=0,
        min_views=3,
        function_tolerance=1e-12,
    )

    n = len(_TRUE_POSES)
    assert result["views_used"] == n
    assert len(result["object_poses"]) == n
    assert len(result["per_view_rms"]) == n
    assert result["rms_reprojection_error"] < 1e-2
    for i, r in enumerate(result["per_view_rms"]):
        assert r < 1e-2, f"view {i} RMS={r}"

    cam = result["camera"]
    assert abs(cam["fx"] - fx_true) / fx_true < 1e-3
    assert abs(cam["fy"] - fy_true) / fy_true < 1e-3
    assert abs(cam["cx"] - cx_true) < 0.5
    assert abs(cam["cy"] - cy_true) < 0.5
    assert abs(cam["fy"] - cam["fx"]) < 1e-9
    assert list(cam["dist"]) == []

    pose0 = result["object_poses"][0]
    assert set(pose0.keys()) == {"position", "rotation"}
    assert set(pose0["position"].keys()) == {"x", "y", "z"}
    assert set(pose0["rotation"].keys()) == {"x", "y", "z", "w"}


def test_calibrate_from_square_views_image_width_height():
    corners = _synthetic_square_corners()
    result = auki_pnpkit.calibrate_from_square_views(
        corners,
        physical_size=0.2,
        image_width=640,
        image_height=480,
        dist_len=0,
        function_tolerance=1e-12,
    )
    assert result["views_used"] == len(_TRUE_POSES)
    assert abs(result["camera"]["fx"] - 800.0) / 800.0 < 1e-3


def test_calibrate_from_square_views_sequence_input():
    corners_arr = _synthetic_square_corners()
    corners_list = [
        [[float(p[0]), float(p[1])] for p in view] for view in corners_arr
    ]
    result = auki_pnpkit.calibrate_from_square_views(
        corners_list,
        physical_size=0.2,
        image_size=(640, 480),
        dist_len=0,
        function_tolerance=1e-12,
    )
    assert result["rms_reprojection_error"] < 1e-2


def test_calibrate_camera_planar_square():
    physical_size = 0.2
    object_pts = _square_object_points(physical_size)
    corners = _synthetic_square_corners(physical_size=physical_size)
    result = auki_pnpkit.calibrate_camera(
        object_pts,
        corners,  # (V, N, 2)
        image_size=(640, 480),
        dist_len=0,
        function_tolerance=1e-12,
    )
    assert result["views_used"] == len(_TRUE_POSES)
    assert abs(result["camera"]["fx"] - 800.0) / 800.0 < 1e-3


def test_calibrate_rejects_bad_physical_size():
    corners = _synthetic_square_corners()[:3]
    for size in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            auki_pnpkit.calibrate_from_square_views(
                corners,
                physical_size=size,
                image_size=(640, 480),
                dist_len=0,
            )


def test_calibrate_rejects_too_few_views():
    corners = _synthetic_square_corners()[:2]
    with pytest.raises(ValueError):
        auki_pnpkit.calibrate_from_square_views(
            corners,
            physical_size=0.2,
            image_size=(640, 480),
            dist_len=0,
            min_views=3,
        )


def test_calibrate_requires_image_size():
    corners = _synthetic_square_corners()
    with pytest.raises(ValueError, match="image_size"):
        auki_pnpkit.calibrate_from_square_views(
            corners,
            physical_size=0.2,
            dist_len=0,
        )
