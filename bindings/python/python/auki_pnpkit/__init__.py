"""NumPy-friendly Python bindings for PnPKit."""

from __future__ import annotations

from typing import Any

from . import _native

__version__ = _native.__version__


def solve_pnp(
    landmarks: Any,
    observations: Any,
    camera_matrix: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate object pose from 3D–2D correspondences (OpenGL coordinates).

    ``landmarks`` may be a sequence of ``{"id", "position"}`` mappings or an
    ``(N, 3)`` float array of object points (ids default to ``"0".."N-1"``).
    ``observations`` may be a sequence of ``{"id", "position"}`` mappings or an
    ``(N, 2)`` float array of image points.

    ``camera_matrix`` accepts a standard OpenCV ``(3, 3)`` array
    ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, a length-9 column-major sequence,
    a mapping with key ``m``, or a mapping with ``fx``/``fy``/``cx``/``cy``.

    ``method`` is one of ``"epnp"``, ``"iterative"``, or ``"sqpnp"``.
    """
    return _native.solve_pnp(landmarks, observations, camera_matrix, method)


def solve_pnp_camera_pose(
    landmarks: Any,
    observations: Any,
    camera_matrix: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate camera pose (inverse of the object pose returned by :func:`solve_pnp`)."""
    return _native.solve_pnp_camera_pose(
        landmarks, observations, camera_matrix, method
    )


def camera_pose_from_solve_pnp_pose(pose: Any) -> dict[str, Any]:
    """Invert a solvePnP object pose to obtain the camera pose."""
    return _native.camera_pose_from_solve_pnp_pose(pose)


def estimate_square_pose_from_rays(
    rays: Any,
    physical_size: float,
) -> dict[str, Any]:
    """Estimate an Ark square-marker pose from four corner rays.

    ``rays`` must be ordered top-left, top-right, bottom-right, bottom-left.
    Each ray is a mapping with ``origin`` and ``direction`` vectors, or an
    ``(4, 6)`` array of ``[ox, oy, oz, dx, dy, dz]`` rows. Directions may be
    unnormalized. ``physical_size`` uses the same units as the ray frame.
    """
    return _native.estimate_square_pose_from_rays(rays, physical_size)


__all__ = [
    "__version__",
    "camera_pose_from_solve_pnp_pose",
    "estimate_square_pose_from_rays",
    "solve_pnp",
    "solve_pnp_camera_pose",
]
