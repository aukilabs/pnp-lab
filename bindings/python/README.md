# Auki PnPKit for Python

NumPy-friendly Python bindings for PnPKit’s Perspective-n-Point solvers and
square-marker pose estimation.

| | |
|---|---|
| **Distribution** | `aukilabs-pnpkit` |
| **Import** | `auki_pnpkit` |
| **License** | [MIT](LICENSE) |
| **Python** | 3.9+ |

> Not yet published to PyPI. Build from this repository (see below). After the
> first release:
>
> ```bash
> pip install aukilabs-pnpkit
> ```

## Quick start

```python
import numpy as np
import auki_pnpkit

object_points = np.array(
    [
        [-0.15, -0.15, 0.0],
        [0.15, -0.15, 0.0],
        [0.15, 0.15, 0.0],
        [-0.15, 0.15, 0.0],
    ],
    dtype=np.float64,
)
image_points = np.array(
    [
        [849.3577, 461.7641],
        [1070.642, 461.7641],
        [1096.898, 636.8014],
        [823.1021, 636.8014],
    ],
    dtype=np.float64,
)

# Preferred: explicit monocular camera (+ optional OpenCV distortion).
camera = {
    "fx": 815.8511,
    "fy": 815.8511,
    "cx": 960.0,
    "cy": 540.0,
    "dist": [],  # or [k1, k2, p1, p2, k3, ...]
}

pose = auki_pnpkit.solve_pnp(
    object_points,
    image_points,
    camera,
    method="iterative",
)
# pose["position"] / pose["rotation"] — object pose in OpenGL coordinates

camera_pose = auki_pnpkit.solve_pnp_camera_pose(
    object_points,
    image_points,
    camera,
    method="iterative",
)
```

A pinhole `(3, 3)` OpenCV camera matrix is also accepted instead of the
`camera` dict.

## API overview

| Function | Description |
|----------|-------------|
| `solve_pnp(...)` | Object pose (OpenGL) from 3D–2D correspondences |
| `solve_pnp_camera_pose(...)` | Camera pose (inverse of object pose) |
| `camera_pose_from_solve_pnp_pose(pose)` | Invert a pose |
| `estimate_square_pose_from_rays(rays, size)` | Square pose from four rays (TL→TR→BR→BL) |
| `estimate_square_pose_from_pixels(pixels, size, camera)` | Square pose from four pixels + camera |
| `solve_pnp_stereo(...)` | Object pose from stereo observations (OpenGL, left primary) |
| `solve_pnp_stereo_camera_pose(...)` | Camera pose from stereo observations |
| `triangulate(left_px, right_px, rig)` | Midpoint triangulation → left OpenCV 3D point |

Image points are **distorted pixels**. When `dist` is set, they are undistorted
inside the solver before EPnP / iterative / SQPnP run on an ideal pinhole model.

Methods: `"epnp"`, `"iterative"`, `"sqpnp"`.

## Stereo

Calibrated stereo uses a **left-primary** rig. Extrinsics `right_from_left` are
the pose of the right camera in the left camera frame, in **OpenCV** convention
(+Z forward). Returned object poses from `solve_pnp_stereo` are **OpenGL**
(same as mono). Triangulated points are in the **left OpenCV** frame.

```python
import auki_pnpkit

rig = {
    "left": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "dist": []},
    "right": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "dist": []},
    "right_from_left": {
        "position": {"x": 0.12, "y": 0.0, "z": 0.0},  # 12 cm baseline
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    },
}

landmarks = [
    {"id": "0", "position": {"x": -0.1, "y": -0.1, "z": 0.0}},
    {"id": "1", "position": {"x": 0.1, "y": -0.1, "z": 0.0}},
    {"id": "2", "position": {"x": 0.1, "y": 0.1, "z": 0.0}},
    {"id": "3", "position": {"x": -0.1, "y": 0.1, "z": 0.0}},
]

# Matched by id. Either eye may be None / omitted.
observations = [
    {"id": "0", "left": [220.0, 140.0], "right": [200.0, 140.0]},
    {"id": "1", "left": [420.0, 140.0], "right": [400.0, 140.0]},
    {"id": "2", "left": [420.0, 340.0], "right": [400.0, 340.0]},
    {"id": "3", "left": [220.0, 340.0], "right": [200.0, 340.0]},
]

pose = auki_pnpkit.solve_pnp_stereo(
    landmarks, observations, rig, method="iterative"
)

# Single correspondence → 3D in left OpenCV frame.
point = auki_pnpkit.triangulate([320.0, 240.0], [295.0, 240.0], rig)
# point["x"], point["y"], point["z"]
```

## Build and test (from monorepo root)

```bash
just python-build    # wheel → bindings/python/dist/
just python-test     # isolated wheel + pytest
```

From this directory:

```bash
maturin build --release --out dist
maturin sdist --out dist
```

Native work runs without holding the GIL; inputs are copied into Rust-owned
storage before detaching.

## License

MIT — see [LICENSE](LICENSE). Parent project: [PnPKit](https://github.com/aukilabs/pnpkit).
