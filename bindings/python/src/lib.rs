//! Native Python/NumPy bindings for PnPKit.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pnp_core::{
    camera_pose_from_solve_pnp_pose as core_camera_pose_from_solve_pnp_pose,
    estimate_square_pose_from_rays as core_estimate_square_pose_from_rays, solve_pnp as core_solve_pnp,
    solve_pnp_camera_pose as core_solve_pnp_camera_pose, Landmark, LandmarkObservation, Matrix3x3,
    PnpError, Pose, Quaternion, Ray3, SolvePnpMethod, SquarePoseEstimate, Vector2, Vector3,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PySequence, PyString};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn pnp_error(error: PnpError) -> PyErr {
    match error {
        PnpError::InsufficientPoints => {
            PyValueError::new_err("insufficient points for solver")
        }
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

fn mapping_get<'py>(
    value: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
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
        let y = mapping_get(value, "y")?
            .ok_or_else(|| PyValueError::new_err("vector2 missing 'y'"))?;
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
        let y = mapping_get(value, "y")?
            .ok_or_else(|| PyValueError::new_err("vector3 missing 'y'"))?;
        let z = mapping_get(value, "z")?
            .ok_or_else(|| PyValueError::new_err("vector3 missing 'z'"))?;
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

fn parse_camera_matrix(value: &Bound<'_, PyAny>) -> PyResult<Matrix3x3> {
    if let Ok(array2) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array2.shape();
        if shape != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "camera_matrix must have shape (3, 3), got {:?}",
                shape
            )));
        }
        let view = array2.as_array();
        // Accept standard OpenCV row-major K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
        return Ok(Matrix3x3::camera_matrix(
            view[[0, 0]],
            view[[1, 1]],
            view[[0, 2]],
            view[[1, 2]],
        ));
    }

    if let Ok(array1) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        let shape = array1.shape();
        if shape != [9] {
            return Err(PyValueError::new_err(format!(
                "flat camera_matrix must have length 9 (column-major), got shape {:?}",
                shape
            )));
        }
        let view = array1.as_array();
        let mut m = [0.0; 9];
        for (i, value) in view.iter().enumerate() {
            m[i] = *value;
        }
        return Ok(Matrix3x3::new(m));
    }

    if let Some(m_value) = mapping_get(value, "m")? {
        let seq = m_value.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("camera_matrix['m'] must be a length-9 sequence (column-major)")
        })?;
        if seq.len()? != 9 {
            return Err(PyValueError::new_err(format!(
                "camera_matrix['m'] must have length 9, got {}",
                seq.len()?
            )));
        }
        let mut m = [0.0; 9];
        for i in 0..9 {
            m[i] = as_f64(&seq.get_item(i)?)?;
        }
        return Ok(Matrix3x3::new(m));
    }

    if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (
        mapping_get(value, "fx")?,
        mapping_get(value, "fy")?,
        mapping_get(value, "cx")?,
        mapping_get(value, "cy")?,
    ) {
        return Ok(Matrix3x3::camera_matrix(
            as_f64(&fx)?,
            as_f64(&fy)?,
            as_f64(&cx)?,
            as_f64(&cy)?,
        ));
    }

    let seq = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "camera_matrix must be a (3, 3) ndarray, length-9 column-major sequence, \
             dict with 'm', or dict with fx/fy/cx/cy",
        )
    })?;

    // Nested 3x3 sequence (row-major OpenCV K).
    if seq.len()? == 3 {
        let mut rows = [[0.0; 3]; 3];
        for r in 0..3 {
            let row = seq.get_item(r)?;
            let row_seq = row.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err("camera_matrix rows must be length-3 sequences")
            })?;
            if row_seq.len()? != 3 {
                return Err(PyValueError::new_err(
                    "camera_matrix rows must each have length 3",
                ));
            }
            for c in 0..3 {
                rows[r][c] = as_f64(&row_seq.get_item(c)?)?;
            }
        }
        return Ok(Matrix3x3::camera_matrix(
            rows[0][0], rows[1][1], rows[0][2], rows[1][2],
        ));
    }

    if seq.len()? == 9 {
        let mut m = [0.0; 9];
        for i in 0..9 {
            m[i] = as_f64(&seq.get_item(i)?)?;
        }
        return Ok(Matrix3x3::new(m));
    }

    Err(PyValueError::new_err(format!(
        "camera_matrix sequence must be nested 3x3 or flat length 9, got length {}",
        seq.len()?
    )))
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

#[pyfunction]
#[pyo3(signature = (landmarks, observations, camera_matrix, method = "iterative"))]
fn solve_pnp<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    camera_matrix: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_observations(observations)?;
    let camera_matrix = parse_camera_matrix(camera_matrix)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || core_solve_pnp(&landmarks, &observations, &camera_matrix, method))
        .map_err(pnp_error)?;
    pose_to_py(py, &pose)
}

#[pyfunction]
#[pyo3(signature = (landmarks, observations, camera_matrix, method = "iterative"))]
fn solve_pnp_camera_pose<'py>(
    py: Python<'py>,
    landmarks: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyAny>,
    camera_matrix: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let landmarks = parse_landmarks(landmarks)?;
    let observations = parse_observations(observations)?;
    let camera_matrix = parse_camera_matrix(camera_matrix)?;
    let method = parse_method(method)?;
    let pose = py
        .detach(move || {
            core_solve_pnp_camera_pose(&landmarks, &observations, &camera_matrix, method)
        })
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    module.add_function(wrap_pyfunction!(solve_pnp, module)?)?;
    module.add_function(wrap_pyfunction!(solve_pnp_camera_pose, module)?)?;
    module.add_function(wrap_pyfunction!(camera_pose_from_solve_pnp_pose, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_square_pose_from_rays, module)?)?;
    Ok(())
}
