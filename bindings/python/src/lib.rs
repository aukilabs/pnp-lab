//! Native Python/NumPy bindings for PnPKit.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pnp_core::{
    calibrate_camera as core_calibrate_camera,
    calibrate_from_square_views as core_calibrate_from_square_views,
    camera_pose_from_solve_pnp_pose as core_camera_pose_from_solve_pnp_pose,
    estimate_square_pose_from_pixels as core_estimate_square_pose_from_pixels,
    estimate_square_pose_from_rays as core_estimate_square_pose_from_rays,
    solve_pnp as core_solve_pnp, solve_pnp_camera_pose as core_solve_pnp_camera_pose,
    solve_pnp_stereo as core_solve_pnp_stereo,
    solve_pnp_stereo_camera_pose as core_solve_pnp_stereo_camera_pose,
    triangulate_midpoint as core_triangulate_midpoint, CalibrateOptions, CalibrationResult,
    CalibrationView, Camera, Landmark, LandmarkObservation, Matrix3x3, PnpError, Pose, Quaternion,
    Ray3, SolvePnpMethod, SquarePoseEstimate, StereoLandmarkObservation, StereoRig, Vector2,
    Vector3,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PySequence, PyString};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn pnp_error(error: PnpError) -> PyErr {
    match error {
        PnpError::InsufficientPoints => PyValueError::new_err("insufficient points for solver"),
        PnpError::SolverFailed => PyRuntimeError::new_err("solver failed to converge"),
        PnpError::MismatchedCounts => {
            PyValueError::new_err("landmark and observation counts do not match")
        }
    }
}

fn parse_method(method: &str) -> PyResult<SolvePnpMethod> {
    match method {
        "epnp" => Ok(SolvePnpMethod::EPnP),
        "iterative" => Ok(SolvePnpMethod::Iterative),
        "sqpnp" => Ok(SolvePnpMethod::SQPnP),
        _ => Err(PyValueError::new_err(format!(
            "unknown method {method:?}; expected 'epnp', 'iterative', or 'sqpnp'"
        ))),
    }
}

fn as_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(v);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v as f64);
    }
    Err(PyTypeError::new_err(format!(
        "expected a number, got {}",
        value.get_type().name()?
    )))
}

fn mapping_get<'py>(value: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    if value.hasattr(key)? {
        let item = value.getattr(key)?;
        if item.is_none() {
            return Ok(None);
        }
        return Ok(Some(item));
    }
    if let Ok(mapping) = value.cast::<pyo3::types::PyDict>() {
        return Ok(mapping.get_item(key)?);
    }
    // Support Mapping protocol (e.g. typed dicts / custom objects).
    if value.hasattr("get")? {
        let item = value.call_method1("get", (key,))?;
        if item.is_none() {
            return Ok(None);
        }
        return Ok(Some(item));
    }
    Ok(None)
}

fn parse_vector2(value: &Bound<'_, PyAny>) -> PyResult<Vector2> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 1 || shape[0] != 2 {
            return Err(PyValueError::new_err(format!(
                "expected a length-2 vector, got shape {:?}",
                shape
            )));
        }
        let view = array.as_array();
        return Ok(Vector2::new(view[0], view[1]));
    }

    if let Some(x) = mapping_get(value, "x")? {
        let y =
            mapping_get(value, "y")?.ok_or_else(|| PyValueError::new_err("vector2 missing 'y'"))?;
        return Ok(Vector2::new(as_f64(&x)?, as_f64(&y)?));
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("expected vector2 as sequence, mapping, or length-2 ndarray")
    })?;
    if seq.len()? != 2 {
        return Err(PyValueError::new_err(format!(
            "expected a length-2 vector, got length {}",
            seq.len()?
        )));
    }
    Ok(Vector2::new(
        as_f64(&seq.get_item(0)?)?,
        as_f64(&seq.get_item(1)?)?,
    ))
}

fn parse_vector3(value: &Bound<'_, PyAny>) -> PyResult<Vector3> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 1 || shape[0] != 3 {
            return Err(PyValueError::new_err(format!(
                "expected a length-3 vector, got shape {:?}",
                shape
            )));
        }
        let view = array.as_array();
        return Ok(Vector3::new(view[0], view[1], view[2]));
    }

    if let Some(x) = mapping_get(value, "x")? {
        let y =
            mapping_get(value, "y")?.ok_or_else(|| PyValueError::new_err("vector3 missing 'y'"))?;
        let z =
            mapping_get(value, "z")?.ok_or_else(|| PyValueError::new_err("vector3 missing 'z'"))?;
        return Ok(Vector3::new(as_f64(&x)?, as_f64(&y)?, as_f64(&z)?));
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("expected vector3 as sequence, mapping, or length-3 ndarray")
    })?;
    if seq.len()? != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a length-3 vector, got length {}",
            seq.len()?
        )));
    }
    Ok(Vector3::new(
        as_f64(&seq.get_item(0)?)?,
        as_f64(&seq.get_item(1)?)?,
        as_f64(&seq.get_item(2)?)?,
    ))
}

fn parse_quaternion(value: &Bound<'_, PyAny>) -> PyResult<Quaternion> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 1 || shape[0] != 4 {
            return Err(PyValueError::new_err(format!(
                "expected a length-4 quaternion [x, y, z, w], got shape {:?}",
                shape
            )));
        }
        let view = array.as_array();
        return Ok(Quaternion::new(view[0], view[1], view[2], view[3]));
    }

    if let Some(x) = mapping_get(value, "x")? {
        let y = mapping_get(value, "y")?
            .ok_or_else(|| PyValueError::new_err("quaternion missing 'y'"))?;
        let z = mapping_get(value, "z")?
            .ok_or_else(|| PyValueError::new_err("quaternion missing 'z'"))?;
        let w = mapping_get(value, "w")?
            .ok_or_else(|| PyValueError::new_err("quaternion missing 'w'"))?;
        return Ok(Quaternion::new(
            as_f64(&x)?,
            as_f64(&y)?,
            as_f64(&z)?,
            as_f64(&w)?,
        ));
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "expected quaternion as sequence [x,y,z,w], mapping, or length-4 ndarray",
        )
    })?;
    if seq.len()? != 4 {
        return Err(PyValueError::new_err(format!(
            "expected a length-4 quaternion, got length {}",
            seq.len()?
        )));
    }
    Ok(Quaternion::new(
        as_f64(&seq.get_item(0)?)?,
        as_f64(&seq.get_item(1)?)?,
        as_f64(&seq.get_item(2)?)?,
        as_f64(&seq.get_item(3)?)?,
    ))
}

fn parse_pose(value: &Bound<'_, PyAny>) -> PyResult<Pose> {
    let position = mapping_get(value, "position")?
        .ok_or_else(|| PyValueError::new_err("pose missing 'position'"))?;
    let rotation = mapping_get(value, "rotation")?
        .ok_or_else(|| PyValueError::new_err("pose missing 'rotation'"))?;
    Ok(Pose::new(
        parse_vector3(&position)?,
        parse_quaternion(&rotation)?,
    ))
}

fn parse_id(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = value.extract::<String>() {
        return Ok(s);
    }
    if let Ok(s) = value.cast::<PyString>() {
        return Ok(s.to_str()?.to_owned());
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(i.to_string());
    }
    Ok(value.str()?.to_str()?.to_owned())
}

fn parse_landmarks(value: &Bound<'_, PyAny>) -> PyResult<Vec<Landmark>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "object_points must have shape (N, 3), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut landmarks = Vec::with_capacity(shape[0]);
        for (i, row) in view.outer_iter().enumerate() {
            landmarks.push(Landmark {
                id: i.to_string(),
                position: Vector3::new(row[0], row[1], row[2]),
            });
        }
        return Ok(landmarks);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("landmarks must be a sequence of dicts or an (N, 3) ndarray")
    })?;
    let mut landmarks = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let id = match mapping_get(&item, "id")? {
            Some(id) => parse_id(&id)?,
            None => i.to_string(),
        };
        let position = mapping_get(&item, "position")?
            .ok_or_else(|| PyValueError::new_err(format!("landmark {i} missing 'position'")))?;
        landmarks.push(Landmark {
            id,
            position: parse_vector3(&position)?,
        });
    }
    Ok(landmarks)
}

fn parse_observations(value: &Bound<'_, PyAny>) -> PyResult<Vec<LandmarkObservation>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "image_points must have shape (N, 2), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut observations = Vec::with_capacity(shape[0]);
        for (i, row) in view.outer_iter().enumerate() {
            observations.push(LandmarkObservation {
                id: i.to_string(),
                position: Vector2::new(row[0], row[1]),
            });
        }
        return Ok(observations);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("observations must be a sequence of dicts or an (N, 2) ndarray")
    })?;
    let mut observations = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let id = match mapping_get(&item, "id")? {
            Some(id) => parse_id(&id)?,
            None => i.to_string(),
        };
        let position = mapping_get(&item, "position")?
            .ok_or_else(|| PyValueError::new_err(format!("observation {i} missing 'position'")))?;
        observations.push(LandmarkObservation {
            id,
            position: parse_vector2(&position)?,
        });
    }
    Ok(observations)
}

fn parse_dist_coeffs(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if value.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    let seq = value
        .cast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("dist must be a sequence of floats"))?;
    let mut out = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        out.push(as_f64(&seq.get_item(i)?)?);
    }
    Ok(out)
}

fn camera_from_intrinsics(fx: f64, fy: f64, cx: f64, cy: f64, dist: &[f64]) -> PyResult<Camera> {
    Camera::new(fx, fy, cx, cy, dist).map_err(pnp_error)
}

/// Parse a [`Camera`] from:
/// - mapping `{fx, fy, cx, cy, dist?}`
/// - OpenCV-style `(3, 3)` matrix (pinhole, no distortion)
/// - mapping with column-major `m` (pinhole)
fn parse_camera(value: &Bound<'_, PyAny>) -> PyResult<Camera> {
    if let Ok(array2) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array2.shape();
        if shape != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "camera as matrix must have shape (3, 3), got {:?}",
                shape
            )));
        }
        let view = array2.as_array();
        return camera_from_intrinsics(view[[0, 0]], view[[1, 1]], view[[0, 2]], view[[1, 2]], &[]);
    }

    if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (
        mapping_get(value, "fx")?,
        mapping_get(value, "fy")?,
        mapping_get(value, "cx")?,
        mapping_get(value, "cy")?,
    ) {
        let dist = match mapping_get(value, "dist")? {
            Some(d) => parse_dist_coeffs(&d)?,
            None => Vec::new(),
        };
        return camera_from_intrinsics(
            as_f64(&fx)?,
            as_f64(&fy)?,
            as_f64(&cx)?,
            as_f64(&cy)?,
            &dist,
        );
    }

    if let Some(m_value) = mapping_get(value, "m")? {
        let seq = m_value.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("camera['m'] must be a length-9 sequence (column-major)")
        })?;
        if seq.len()? != 9 {
            return Err(PyValueError::new_err(format!(
                "camera['m'] must have length 9, got {}",
                seq.len()?
            )));
        }
        let mut m = [0.0; 9];
        for i in 0..9 {
            m[i] = as_f64(&seq.get_item(i)?)?;
        }
        let matrix = Matrix3x3::new(m);
        let dist = match mapping_get(value, "dist")? {
            Some(d) => parse_dist_coeffs(&d)?,
            None => Vec::new(),
        };
        return Camera::from_matrix(&matrix, &dist).map_err(pnp_error);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "camera must be a mapping with fx/fy/cx/cy[/dist], a (3, 3) matrix, or nested 3x3",
        )
    })?;

    if seq.len()? == 3 {
        let mut rows = [[0.0; 3]; 3];
        for r in 0..3 {
            let row = seq.get_item(r)?;
            let row_seq = row
                .cast::<PySequence>()
                .map_err(|_| PyTypeError::new_err("camera rows must be length-3 sequences"))?;
            if row_seq.len()? != 3 {
                return Err(PyValueError::new_err("camera rows must each have length 3"));
            }
            for c in 0..3 {
                rows[r][c] = as_f64(&row_seq.get_item(c)?)?;
            }
        }
        return camera_from_intrinsics(rows[0][0], rows[1][1], rows[0][2], rows[1][2], &[]);
    }

    Err(PyValueError::new_err(
        "unable to parse camera; expected {fx,fy,cx,cy,dist?} or (3,3) matrix",
    ))
}

/// Parse a [`StereoRig`] from:
/// `{left: camera, right: camera, right_from_left: pose}`
///
/// `right_from_left` is the pose of the right camera in the **left** camera
/// frame, in **OpenCV** convention (same as core).
fn parse_stereo_rig(value: &Bound<'_, PyAny>) -> PyResult<StereoRig> {
    let left = mapping_get(value, "left")?
        .ok_or_else(|| PyValueError::new_err("stereo rig missing 'left'"))?;
    let right = mapping_get(value, "right")?
        .ok_or_else(|| PyValueError::new_err("stereo rig missing 'right'"))?;
    let right_from_left = mapping_get(value, "right_from_left")?
        .ok_or_else(|| PyValueError::new_err("stereo rig missing 'right_from_left'"))?;
    StereoRig::new(
        parse_camera(&left)?,
        parse_camera(&right)?,
        parse_pose(&right_from_left)?,
    )
    .map_err(pnp_error)
}

/// Parse stereo observations:
/// `[{"id"?, "left": [u,v]|None, "right": [u,v]|None}, ...]`
///
/// Either eye may be missing/`None`; at least one projection is needed later
/// by the solver (enforced in core).
fn parse_stereo_observations(value: &Bound<'_, PyAny>) -> PyResult<Vec<StereoLandmarkObservation>> {
    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "stereo observations must be a sequence of dicts with optional left/right pixels",
        )
    })?;
    let mut observations = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let id = match mapping_get(&item, "id")? {
            Some(id) => parse_id(&id)?,
            None => i.to_string(),
        };
        // mapping_get on dicts returns Some(PyNone) when the key is present with
        // a None value; treat that the same as a missing eye.
        let left = match mapping_get(&item, "left")? {
            Some(v) if !v.is_none() => Some(parse_vector2(&v)?),
            _ => None,
        };
        let right = match mapping_get(&item, "right")? {
            Some(v) if !v.is_none() => Some(parse_vector2(&v)?),
            _ => None,
        };
        observations.push(StereoLandmarkObservation { id, left, right });
    }
    Ok(observations)
}

fn parse_pixels4(value: &Bound<'_, PyAny>) -> PyResult<[Vector2; 4]> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape != [4, 2] {
            return Err(PyValueError::new_err(format!(
                "pixels must have shape (4, 2), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        return Ok([
            Vector2::new(view[[0, 0]], view[[0, 1]]),
            Vector2::new(view[[1, 0]], view[[1, 1]]),
            Vector2::new(view[[2, 0]], view[[2, 1]]),
            Vector2::new(view[[3, 0]], view[[3, 1]]),
        ]);
    }
    let seq = value
        .cast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("pixels must be a sequence of 4 points"))?;
    if seq.len()? != 4 {
        return Err(PyValueError::new_err(format!(
            "expected exactly 4 pixels (TL, TR, BR, BL), got {}",
            seq.len()?
        )));
    }
    Ok([
        parse_vector2(&seq.get_item(0)?)?,
        parse_vector2(&seq.get_item(1)?)?,
        parse_vector2(&seq.get_item(2)?)?,
        parse_vector2(&seq.get_item(3)?)?,
    ])
}

fn parse_ray(value: &Bound<'_, PyAny>) -> PyResult<Ray3> {
    let origin = mapping_get(value, "origin")?
        .ok_or_else(|| PyValueError::new_err("ray missing 'origin'"))?;
    let direction = mapping_get(value, "direction")?
        .ok_or_else(|| PyValueError::new_err("ray missing 'direction'"))?;
    Ok(Ray3 {
        origin: parse_vector3(&origin)?,
        direction: parse_vector3(&direction)?,
    })
}

fn parse_rays(value: &Bound<'_, PyAny>) -> PyResult<[Ray3; 4]> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape != [4, 6] {
            return Err(PyValueError::new_err(format!(
                "rays ndarray must have shape (4, 6) as [ox,oy,oz,dx,dy,dz], got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut rays = [Ray3 {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(0.0, 0.0, 1.0),
        }; 4];
        for i in 0..4 {
            rays[i] = Ray3 {
                origin: Vector3::new(view[[i, 0]], view[[i, 1]], view[[i, 2]]),
                direction: Vector3::new(view[[i, 3]], view[[i, 4]], view[[i, 5]]),
            };
        }
        return Ok(rays);
    }

    let seq = value
        .cast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("rays must be a sequence of 4 ray mappings"))?;
    if seq.len()? != 4 {
        return Err(PyValueError::new_err(format!(
            "expected exactly 4 rays (TL, TR, BR, BL), got {}",
            seq.len()?
        )));
    }
    Ok([
        parse_ray(&seq.get_item(0)?)?,
        parse_ray(&seq.get_item(1)?)?,
        parse_ray(&seq.get_item(2)?)?,
        parse_ray(&seq.get_item(3)?)?,
    ])
}

fn vector3_to_py<'py>(py: Python<'py>, v: &Vector3) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", v.x)?;
    dict.set_item("y", v.y)?;
    dict.set_item("z", v.z)?;
    Ok(dict)
}

fn quaternion_to_py<'py>(py: Python<'py>, q: &Quaternion) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", q.x)?;
    dict.set_item("y", q.y)?;
    dict.set_item("z", q.z)?;
    dict.set_item("w", q.w)?;
    Ok(dict)
}

fn pose_to_py<'py>(py: Python<'py>, pose: &Pose) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("position", vector3_to_py(py, &pose.position)?)?;
    dict.set_item("rotation", quaternion_to_py(py, &pose.rotation)?)?;
    Ok(dict)
}

fn estimate_to_py<'py>(
    py: Python<'py>,
    estimate: &SquarePoseEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("pose", pose_to_py(py, &estimate.pose)?)?;
    dict.set_item("confidence", estimate.confidence)?;
    dict.set_item("normalized_corner_error", estimate.normalized_corner_error)?;
    let distances = PyList::new(
        py,
        estimate
            .ray_distances
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    dict.set_item("ray_distances", distances)?;
    Ok(dict)
}

fn camera_to_py<'py>(py: Python<'py>, camera: &Camera) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("fx", camera.fx)?;
    dict.set_item("fy", camera.fy)?;
    dict.set_item("cx", camera.cx)?;
    dict.set_item("cy", camera.cy)?;
    let dist = PyList::new(py, camera.dist.as_slice())?;
    dict.set_item("dist", dist)?;
    Ok(dict)
}

fn calibration_result_to_py<'py>(
    py: Python<'py>,
    result: &CalibrationResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("camera", camera_to_py(py, &result.camera)?)?;
    dict.set_item("rms_reprojection_error", result.rms_reprojection_error)?;
    let per_view = PyList::new(py, result.per_view_rms.as_slice())?;
    dict.set_item("per_view_rms", per_view)?;
    let poses = PyList::empty(py);
    for pose in &result.object_poses {
        poses.append(pose_to_py(py, pose)?)?;
    }
    dict.set_item("object_poses", poses)?;
    dict.set_item("views_used", result.views_used)?;
    Ok(dict)
}

/// Resolve image dimensions from either `image_size=(w,h)` or explicit
/// `image_width` / `image_height`. Prefer `image_size` when both are given.
fn resolve_image_size(
    image_size: Option<&Bound<'_, PyAny>>,
    image_width: Option<u32>,
    image_height: Option<u32>,
) -> PyResult<(u32, u32)> {
    if let Some(size) = image_size {
        if size.is_none() {
            // fall through
        } else {
            let seq = size.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err("image_size must be a length-2 sequence (width, height)")
            })?;
            if seq.len()? != 2 {
                return Err(PyValueError::new_err(format!(
                    "image_size must have length 2, got {}",
                    seq.len()?
                )));
            }
            let w = as_f64(&seq.get_item(0)?)?;
            let h = as_f64(&seq.get_item(1)?)?;
            if !w.is_finite() || !h.is_finite() || w < 2.0 || h < 2.0 {
                return Err(PyValueError::new_err(
                    "image_size width/height must be finite and >= 2",
                ));
            }
            return Ok((w as u32, h as u32));
        }
    }
    match (image_width, image_height) {
        (Some(w), Some(h)) => {
            if w < 2 || h < 2 {
                return Err(PyValueError::new_err(
                    "image_width and image_height must be >= 2",
                ));
            }
            Ok((w, h))
        }
        _ => Err(PyValueError::new_err(
            "provide image_size=(width, height) or both image_width and image_height",
        )),
    }
}

fn build_calibrate_options(
    fix_aspect_ratio: bool,
    fix_principal_point: bool,
    dist_len: usize,
    min_views: usize,
    max_iterations: usize,
    function_tolerance: f64,
    rms_success_threshold: Option<f64>,
) -> PyResult<CalibrateOptions> {
    if !function_tolerance.is_finite() || function_tolerance <= 0.0 {
        return Err(PyValueError::new_err(
            "function_tolerance must be a finite positive number",
        ));
    }
    if let Some(t) = rms_success_threshold {
        if !t.is_finite() || t < 0.0 {
            return Err(PyValueError::new_err(
                "rms_success_threshold must be a finite non-negative number",
            ));
        }
    }
    Ok(CalibrateOptions {
        min_views,
        fix_aspect_ratio,
        fix_principal_point,
        dist_len,
        max_iterations,
        function_tolerance,
        rms_success_threshold,
    })
}

/// Parse multi-view square corners: `(N, 4, 2)` ndarray or sequence of 4-point views.
fn parse_square_corners_views(value: &Bound<'_, PyAny>) -> PyResult<Vec<[Vector2; 4]>> {
    if let Ok(array) = value.extract::<PyReadonlyArray3<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 3 || shape[1] != 4 || shape[2] != 2 {
            return Err(PyValueError::new_err(format!(
                "corners ndarray must have shape (N, 4, 2), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut out = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            out.push([
                Vector2::new(view[[i, 0, 0]], view[[i, 0, 1]]),
                Vector2::new(view[[i, 1, 0]], view[[i, 1, 1]]),
                Vector2::new(view[[i, 2, 0]], view[[i, 2, 1]]),
                Vector2::new(view[[i, 3, 0]], view[[i, 3, 1]]),
            ]);
        }
        return Ok(out);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("corners must be a sequence of 4-point views or an (N, 4, 2) ndarray")
    })?;
    let mut out = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        out.push(parse_pixels4(&seq.get_item(i)?)?);
    }
    Ok(out)
}

/// Parse object points as `(N, 3)` ndarray or sequence of vector3.
fn parse_object_points(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vector3>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "object_points must have shape (N, 3), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut pts = Vec::with_capacity(shape[0]);
        for row in view.outer_iter() {
            pts.push(Vector3::new(row[0], row[1], row[2]));
        }
        return Ok(pts);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("object_points must be a sequence of vector3 or an (N, 3) ndarray")
    })?;
    let mut pts = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        pts.push(parse_vector3(&seq.get_item(i)?)?);
    }
    Ok(pts)
}

/// Parse one calibration view's image points: `(N, 2)` or sequence of vector2,
/// or a mapping with `image_points`.
fn parse_image_points(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vector2>> {
    if let Some(inner) = mapping_get(value, "image_points")? {
        return parse_image_points(&inner);
    }

    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "view image_points must have shape (N, 2), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut pts = Vec::with_capacity(shape[0]);
        for row in view.outer_iter() {
            pts.push(Vector2::new(row[0], row[1]));
        }
        return Ok(pts);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "each calibration view must be a sequence of image points, an (N, 2) ndarray, or a mapping with image_points",
        )
    })?;
    let mut pts = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        pts.push(parse_vector2(&seq.get_item(i)?)?);
    }
    Ok(pts)
}

/// Parse multi-view calibration views: sequence of image-point lists, or `(V, N, 2)` ndarray.
fn parse_calibration_views(value: &Bound<'_, PyAny>) -> PyResult<Vec<CalibrationView>> {
    if let Ok(array) = value.extract::<PyReadonlyArray3<'_, f64>>() {
        let shape = array.shape();
        if shape.len() != 3 || shape[2] != 2 {
            return Err(PyValueError::new_err(format!(
                "views ndarray must have shape (V, N, 2), got {:?}",
                shape
            )));
        }
        let view = array.as_array();
        let mut out = Vec::with_capacity(shape[0]);
        for v in 0..shape[0] {
            let mut image_points = Vec::with_capacity(shape[1]);
            for n in 0..shape[1] {
                image_points.push(Vector2::new(view[[v, n, 0]], view[[v, n, 1]]));
            }
            out.push(CalibrationView { image_points });
        }
        return Ok(out);
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("views must be a sequence of image-point lists or a (V, N, 2) ndarray")
    })?;
    let mut out = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        out.push(CalibrationView {
            image_points: parse_image_points(&seq.get_item(i)?)?,
        });
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (landmarks, observations, camera, method = "iterative"))]
fn solve_pnp<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    camera: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_observations(observations)?;
    let camera = parse_camera(camera)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || core_solve_pnp(&landmarks, &observations, &camera, method))
        .map_err(pnp_error)?;
    pose_to_py(py, &pose)
}

#[pyfunction]
#[pyo3(signature = (landmarks, observations, camera, method = "iterative"))]
fn solve_pnp_camera_pose<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    camera: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_observations(observations)?;
    let camera = parse_camera(camera)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || core_solve_pnp_camera_pose(&landmarks, &observations, &camera, method))
        .map_err(pnp_error)?;
    pose_to_py(py, &pose)
}

#[pyfunction]
fn camera_pose_from_solve_pnp_pose<'py>(
    py: Python<'py>,
    pose: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pose = parse_pose(pose)?;
    let inverted = py.detach(move || core_camera_pose_from_solve_pnp_pose(&pose));
    pose_to_py(py, &inverted)
}

#[pyfunction]
fn estimate_square_pose_from_rays<'py>(
    py: Python<'py>,
    rays: &Bound<'py, PyAny>,
    physical_size: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PyValueError::new_err(
            "physical_size must be a finite positive number",
        ));
    }
    let rays = parse_rays(rays)?;
    let estimate = py
        .detach(move || core_estimate_square_pose_from_rays(rays, physical_size))
        .map_err(pnp_error)?;
    estimate_to_py(py, &estimate)
}

#[pyfunction]
fn estimate_square_pose_from_pixels<'py>(
    py: Python<'py>,
    pixels: &Bound<'py, PyAny>,
    physical_size: f64,
    camera: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PyValueError::new_err(
            "physical_size must be a finite positive number",
        ));
    }
    let pixels = parse_pixels4(pixels)?;
    let camera = parse_camera(camera)?;
    let estimate = py
        .detach(move || core_estimate_square_pose_from_pixels(pixels, physical_size, &camera))
        .map_err(pnp_error)?;
    estimate_to_py(py, &estimate)
}

#[pyfunction]
#[pyo3(signature = (landmarks, observations, rig, method = "iterative"))]
fn solve_pnp_stereo<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    rig: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_stereo_observations(observations)?;
    let rig = parse_stereo_rig(rig)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || core_solve_pnp_stereo(&landmarks, &observations, &rig, method))
        .map_err(pnp_error)?;
    pose_to_py(py, &pose)
}

#[pyfunction]
#[pyo3(signature = (landmarks, observations, rig, method = "iterative"))]
fn solve_pnp_stereo_camera_pose<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    rig: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_stereo_observations(observations)?;
    let rig = parse_stereo_rig(rig)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || core_solve_pnp_stereo_camera_pose(&landmarks, &observations, &rig, method))
        .map_err(pnp_error)?;
    pose_to_py(py, &pose)
}

/// Midpoint triangulation → 3D point in the **left camera OpenCV** frame.
#[pyfunction]
fn triangulate<'py>(
    py: Python<'py>,
    left_pixel: &Bound<'py, PyAny>,
    right_pixel: &Bound<'py, PyAny>,
    rig: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let left_px = parse_vector2(left_pixel)?;
    let right_px = parse_vector2(right_pixel)?;
    let rig = parse_stereo_rig(rig)?;
    let point = py
        .detach(move || core_triangulate_midpoint(&rig, left_px, right_px))
        .map_err(pnp_error)?;
    vector3_to_py(py, &point)
}

/// Multi-view monocular calibration from arbitrary shared object points.
#[pyfunction]
#[pyo3(signature = (
    object_points,
    views,
    image_size = None,
    *,
    image_width = None,
    image_height = None,
    fix_aspect_ratio = true,
    fix_principal_point = false,
    dist_len = 5,
    min_views = 3,
    max_iterations = 100,
    function_tolerance = 1e-10,
    rms_success_threshold = None,
))]
fn calibrate_camera<'py>(
    py: Python<'py>,
    object_points: &Bound<'py, PyAny>,
    views: &Bound<'py, PyAny>,
    image_size: Option<&Bound<'py, PyAny>>,
    image_width: Option<u32>,
    image_height: Option<u32>,
    fix_aspect_ratio: bool,
    fix_principal_point: bool,
    dist_len: usize,
    min_views: usize,
    max_iterations: usize,
    function_tolerance: f64,
    rms_success_threshold: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (w, h) = resolve_image_size(image_size, image_width, image_height)?;
    let options = build_calibrate_options(
        fix_aspect_ratio,
        fix_principal_point,
        dist_len,
        min_views,
        max_iterations,
        function_tolerance,
        rms_success_threshold,
    )?;
    let object_points = parse_object_points(object_points)?;
    let views = parse_calibration_views(views)?;
    let result = py
        .detach(move || core_calibrate_camera(&object_points, &views, w, h, &options))
        .map_err(pnp_error)?;
    calibration_result_to_py(py, &result)
}

/// Calibrate from multiple views of a square marker (QR / planar quad corners).
///
/// `corners` is a sequence of 4-point views (TL→TR→BR→BL) or an `(N, 4, 2)`
/// ndarray. Image size via `image_size=(w, h)` or `image_width`/`image_height`.
#[pyfunction]
#[pyo3(signature = (
    corners,
    physical_size,
    image_size = None,
    *,
    image_width = None,
    image_height = None,
    fix_aspect_ratio = true,
    fix_principal_point = false,
    dist_len = 5,
    min_views = 3,
    max_iterations = 100,
    function_tolerance = 1e-10,
    rms_success_threshold = None,
))]
fn calibrate_from_square_views<'py>(
    py: Python<'py>,
    corners: &Bound<'py, PyAny>,
    physical_size: f64,
    image_size: Option<&Bound<'py, PyAny>>,
    image_width: Option<u32>,
    image_height: Option<u32>,
    fix_aspect_ratio: bool,
    fix_principal_point: bool,
    dist_len: usize,
    min_views: usize,
    max_iterations: usize,
    function_tolerance: f64,
    rms_success_threshold: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PyValueError::new_err(
            "physical_size must be a finite positive number",
        ));
    }
    let (w, h) = resolve_image_size(image_size, image_width, image_height)?;
    let options = build_calibrate_options(
        fix_aspect_ratio,
        fix_principal_point,
        dist_len,
        min_views,
        max_iterations,
        function_tolerance,
        rms_success_threshold,
    )?;
    let corners = parse_square_corners_views(corners)?;
    let result = py
        .detach(move || core_calibrate_from_square_views(&corners, physical_size, w, h, &options))
        .map_err(pnp_error)?;
    calibration_result_to_py(py, &result)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    module.add_function(wrap_pyfunction!(solve_pnp, module)?)?;
    module.add_function(wrap_pyfunction!(solve_pnp_camera_pose, module)?)?;
    module.add_function(wrap_pyfunction!(camera_pose_from_solve_pnp_pose, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_square_pose_from_rays, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_square_pose_from_pixels, module)?)?;
    module.add_function(wrap_pyfunction!(solve_pnp_stereo, module)?)?;
    module.add_function(wrap_pyfunction!(solve_pnp_stereo_camera_pose, module)?)?;
    module.add_function(wrap_pyfunction!(triangulate, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_camera, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_from_square_views, module)?)?;
    Ok(())
}
