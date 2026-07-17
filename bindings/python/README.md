# Auki PnPKit for Python

NumPy-friendly Python bindings for PnPKit's Perspective-n-Point solvers and
Ark square-marker pose estimation. The distribution is named
`aukilabs-pnpkit` and imports as `auki_pnpkit`.

> The package is not yet published to PyPI. After its first release, it will
> be installable with:

```bash
pip install aukilabs-pnpkit
```

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
camera_matrix = np.array(
    [
        [815.8511, 0.0, 960.0],
        [0.0, 815.8511, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

pose = auki_pnpkit.solve_pnp(
    object_points,
    image_points,
    camera_matrix,
    method="iterative",
)
camera_pose = auki_pnpkit.solve_pnp_camera_pose(
    object_points,
    image_points,
    camera_matrix,
    method="iterative",
)
```

`solve_pnp` returns the object pose in OpenGL coordinates. Use
`solve_pnp_camera_pose` (or `camera_pose_from_solve_pnp_pose`) when you need the
camera pose instead.

For square-marker calibration, pass four corner rays ordered top-left,
top-right, bottom-right, bottom-left:

```python
estimate = auki_pnpkit.estimate_square_pose_from_rays(rays, physical_size=0.8)
print(estimate["pose"], estimate["confidence"])
```

Native solver work runs without holding Python's GIL. Inputs are copied into
Rust-owned storage before detaching so another thread cannot race the Python
objects.

Build a local wheel from the repository root with `just python-build`, or run
the Python integration suite with `just python-test`.

To inspect the artifacts intended for PyPI, run the following from this
directory:

```bash
maturin build --release --out dist
maturin sdist --out dist
```

The Python distribution is licensed under the [MIT License](LICENSE).
