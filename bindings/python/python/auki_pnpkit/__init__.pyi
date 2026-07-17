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

class Landmark(TypedDict, total=False):
    id: str
    position: Vector3 | Sequence[float] | NDArray[np.float64]

class LandmarkObservation(TypedDict, total=False):
    id: str
    position: Vector2 | Sequence[float] | NDArray[np.float64]

class Ray(TypedDict):
    origin: Vector3 | Sequence[float] | NDArray[np.float64]
    direction: Vector3 | Sequence[float] | NDArray[np.float64]

class SquarePoseEstimate(TypedDict):
    pose: Pose
    confidence: float
    normalized_corner_error: float
    ray_distances: list[float]

__version__: str

def solve_pnp(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[LandmarkObservation] | ArrayLike,
    camera_matrix: ArrayLike | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def solve_pnp_camera_pose(
    landmarks: Sequence[Landmark] | ArrayLike,
    observations: Sequence[LandmarkObservation] | ArrayLike,
    camera_matrix: ArrayLike | Mapping[str, Any],
    method: Method = ...,
) -> Pose: ...

def camera_pose_from_solve_pnp_pose(pose: Pose | Mapping[str, Any]) -> Pose: ...

def estimate_square_pose_from_rays(
    rays: Sequence[Ray] | ArrayLike,
    physical_size: float,
) -> SquarePoseEstimate: ...
