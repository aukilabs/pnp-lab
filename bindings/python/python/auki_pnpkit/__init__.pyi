from typing import Any, Literal, Mapping, Sequence, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

Method = Literal["epnp", "iterative", "sqpnp"]

class Vector2(TypedDict):
    x: float
    y: float

class Vector3(TypedDict):
    x: float
    y: float
    z: float

class Quaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float

class Pose(TypedDict):
    position: Vector3
    rotation: Quaternion

class Camera(TypedDict, total=False):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Sequence[float] | NDArray[np.float64]
    m: Sequence[float]

class Landmark(TypedDict, total=False):
    id: str
    position: Vector3 | Sequence[float] | NDArray[np.float64]

class LandmarkObservation(TypedDict, total=False):
    id: str
    position: Vector2 | Sequence[float] | NDArray[np.float64]

class StereoLandmarkObservation(TypedDict, total=False):
    id: str
    left: Vector2 | Sequence[float] | NDArray[np.float64] | None
    right: Vector2 | Sequence[float] | NDArray[np.float64] | None

class StereoRig(TypedDict):
    left: Camera | ArrayLike | Mapping[str, Any]
    right: Camera | ArrayLike | Mapping[str, Any]
    right_from_left: Pose | Mapping[str, Any]

class Ray(TypedDict):
    origin: Vector3 | Sequence[float] | NDArray[np.float64]
    direction: Vector3 | Sequence[float] | NDArray[np.float64]

class SquarePoseEstimate(TypedDict):
    pose: Pose
    confidence: float
    normalized_corner_error: float
    ray_distances: list[float]

class CalibrationCamera(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float]

class CalibrationResult(TypedDict):
    camera: CalibrationCamera
    rms_reprojection_error: float
    per_view_rms: list[float]
    object_poses: list[Pose]
    views_used: int

__version__: str

def solve_pnp(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[LandmarkObservation] | ArrayLike,
    camera: Camera | ArrayLike | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def solve_pnp_camera_pose(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[LandmarkObservation] | ArrayLike,
    camera: Camera | ArrayLike | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def camera_pose_from_solve_pnp_pose(pose: Pose | Mapping[str, Any]) -> Pose: ...

def estimate_square_pose_from_rays(
    rays: Sequence[Ray] | ArrayLike,
    physical_size: float,
) -> SquarePoseEstimate: ...

def estimate_square_pose_from_pixels(
    pixels: Sequence[Vector2 | Sequence[float]] | ArrayLike,
    physical_size: float,
    camera: Camera | ArrayLike | Mapping[str, Any],
) -> SquarePoseEstimate: ...

def solve_pnp_stereo(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[StereoLandmarkObservation],
    rig: StereoRig | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def solve_pnp_stereo_camera_pose(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[StereoLandmarkObservation],
    rig: StereoRig | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def triangulate(
    left_pixel: Vector2 | Sequence[float] | ArrayLike,
    right_pixel: Vector2 | Sequence[float] | ArrayLike,
    rig: StereoRig | Mapping[str, Any],
) -> Vector3: ...

def calibrate_from_square_views(
    corners: Sequence[Sequence[Vector2 | Sequence[float]] | ArrayLike] | ArrayLike,
    physical_size: float,
    image_size: Sequence[int] | tuple[int, int] | None = ...,
    *,
    image_width: int | None = ...,
    image_height: int | None = ...,
    fix_aspect_ratio: bool = ...,
    fix_principal_point: bool = ...,
    dist_len: int = ...,
    min_views: int = ...,
    max_iterations: int = ...,
    function_tolerance: float = ...,
    rms_success_threshold: float | None = ...,
) -> CalibrationResult: ...

def calibrate_camera(
    object_points: Sequence[Vector3 | Sequence[float]] | ArrayLike,
    views: Sequence[Sequence[Vector2 | Sequence[float]] | ArrayLike | Mapping[str, Any]] | ArrayLike,
    image_size: Sequence[int] | tuple[int, int] | None = ...,
    *,
    image_width: int | None = ...,
    image_height: int | None = ...,
    fix_aspect_ratio: bool = ...,
    fix_principal_point: bool = ...,
    dist_len: int = ...,
    min_views: int = ...,
    max_iterations: int = ...,
    function_tolerance: float = ...,
    rms_success_threshold: float | None = ...,
) -> CalibrationResult: ...
