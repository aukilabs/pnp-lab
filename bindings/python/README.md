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

Image points are **distorted pixels**. When `dist` is set, they are undistorted
inside the solver before EPnP / iterative / SQPnP run on an ideal pinhole model.

Methods: `"epnp"`, `"iterative"`, `"sqpnp"`.

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
