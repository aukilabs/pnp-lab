"""NumPy-friendly Python bindings for PnPKit."""

from __future__ import annotations

from typing import Any

from . import _native

__version__ = _native.__version__


def solve_pnp(
    landmarks: Any,
    observations: Any,
    camera: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate object pose from 3D–2D correspondences (OpenGL coordinates).

    ``camera`` is a calibrated monocular model:
    ``{"fx", "fy", "cx", "cy", "dist"?}`` with OpenCV distortion coeffs, or a
    pinhole ``(3, 3)`` OpenCV camera matrix.

    Observations are **distorted pixels**. Distortion is undistorted inside the
    solver before EPnP / iterative / SQPnP run on an ideal pinhole model.
    """
    return _native.solve_pnp(landmarks, observations, camera, method)


def solve_pnp_camera_pose(
    landmarks: Any,
    observations: Any,
    camera: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate camera pose (inverse of the object pose returned by :func:`solve_pnp`)."""
    return _native.solve_pnp_camera_pose(landmarks, observations, camera, method)


def camera_pose_from_solve_pnp_pose(pose: Any) -> dict[str, Any]:
    """Invert a solvePnP object pose to obtain the camera pose."""
    return _native.camera_pose_from_solve_pnp_pose(pose)


def estimate_square_pose_from_rays(
    rays: Any,
    physical_size: float,
) -> dict[str, Any]:
    """Estimate a planar square-marker pose from four corner rays.

    Rays must already be unprojected (distortion handled by the caller).
    Prefer :func:`estimate_square_pose_from_pixels` when you have image pixels.
    """
    return _native.estimate_square_pose_from_rays(rays, physical_size)


def estimate_square_pose_from_pixels(
    pixels: Any,
    physical_size: float,
    camera: Any,
) -> dict[str, Any]:
    """Estimate a planar square-marker pose from four corner pixels and a camera.

    Corner order is top-left, top-right, bottom-right, bottom-left.
    """
    return _native.estimate_square_pose_from_pixels(pixels, physical_size, camera)


def solve_pnp_stereo(
    landmarks: Any,
    observations: Any,
    rig: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate object pose from stereo landmark observations (OpenGL, left primary).

    ``rig`` is a calibrated stereo pair::

        {
            "left": {"fx", "fy", "cx", "cy", "dist"?},
            "right": {...},
            "right_from_left": {"position": {...}, "rotation": {...}},
        }

    ``right_from_left`` is the pose of the right camera in the **left** camera
    frame in **OpenCV** convention. Observations are matched to landmarks by
    string ``id``; each observation may omit ``left`` or ``right`` (or pass
    ``None``).
    """
    return _native.solve_pnp_stereo(landmarks, observations, rig, method)


def solve_pnp_stereo_camera_pose(
    landmarks: Any,
    observations: Any,
    rig: Any,
    method: str = "iterative",
) -> dict[str, Any]:
    """Estimate camera pose from stereo observations (inverse of object pose)."""
    return _native.solve_pnp_stereo_camera_pose(landmarks, observations, rig, method)


def triangulate(
    left_pixel: Any,
    right_pixel: Any,
    rig: Any,
) -> dict[str, Any]:
    """Midpoint-triangulate a stereo correspondence into the left OpenCV frame.

    Returns ``{"x", "y", "z"}`` in the **left camera OpenCV** coordinate system
    (+Z forward). Input pixels may be distorted; they are undistorted via the
    rig cameras.
    """
    return _native.triangulate(left_pixel, right_pixel, rig)


def calibrate_from_square_views(
    corners: Any,
    physical_size: float,
    image_size: Any = None,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    fix_aspect_ratio: bool = True,
    fix_principal_point: bool = False,
    dist_len: int = 5,
    min_views: int = 3,
    max_iterations: int = 100,
    function_tolerance: float = 1e-10,
    rms_success_threshold: float | None = None,
) -> dict[str, Any]:
    """Calibrate monocular intrinsics from multi-view square-marker corners.

    ``corners`` is a sequence of 4-point views (TL→TR→BR→BL) or an ``(N, 4, 2)``
    array. Image size is ``image_size=(width, height)`` or
    ``image_width`` / ``image_height``.

    Returns a dict with ``camera`` (``fx``, ``fy``, ``cx``, ``cy``, ``dist``),
    ``rms_reprojection_error``, ``per_view_rms``, ``object_poses`` (OpenGL),
    and ``views_used``.
    """
    return _native.calibrate_from_square_views(
        corners,
        physical_size,
        image_size,
        image_width=image_width,
        image_height=image_height,
        fix_aspect_ratio=fix_aspect_ratio,
        fix_principal_point=fix_principal_point,
        dist_len=dist_len,
        min_views=min_views,
        max_iterations=max_iterations,
        function_tolerance=function_tolerance,
        rms_success_threshold=rms_success_threshold,
    )


def calibrate_camera(
    object_points: Any,
    views: Any,
    image_size: Any = None,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    fix_aspect_ratio: bool = True,
    fix_principal_point: bool = False,
    dist_len: int = 5,
    min_views: int = 3,
    max_iterations: int = 100,
    function_tolerance: float = 1e-10,
    rms_success_threshold: float | None = None,
) -> dict[str, Any]:
    """Multi-view monocular calibration from shared 3D object points.

    ``object_points`` is ``(N, 3)`` (or a sequence of vector3). ``views`` is a
    sequence of ``(N, 2)`` image-point lists (same order every view), or a
    ``(V, N, 2)`` array. Image size as for :func:`calibrate_from_square_views`.

    Returned pose convention matches :func:`solve_pnp` (OpenGL object poses).
    """
    return _native.calibrate_camera(
        object_points,
        views,
        image_size,
        image_width=image_width,
        image_height=image_height,
        fix_aspect_ratio=fix_aspect_ratio,
        fix_principal_point=fix_principal_point,
        dist_len=dist_len,
        min_views=min_views,
        max_iterations=max_iterations,
        function_tolerance=function_tolerance,
        rms_success_threshold=rms_success_threshold,
    )


__all__ = [
    "__version__",
    "calibrate_camera",
    "calibrate_from_square_views",
    "camera_pose_from_solve_pnp_pose",
    "estimate_square_pose_from_pixels",
    "estimate_square_pose_from_rays",
    "solve_pnp",
    "solve_pnp_camera_pose",
    "solve_pnp_stereo",
    "solve_pnp_stereo_camera_pose",
    "triangulate",
]
