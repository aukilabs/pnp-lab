# Stereo PnP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PnPLab with calibrated stereo: a `StereoRig` model, triangulation, joint left+right landmark PnP, and stereo square-marker pose — without breaking monocular APIs.

**Architecture:** Keep algebraic mono solvers (EPnP / SQPnP / iterative) unchanged. Add a stereo rig type (left/right `Camera` + extrinsics). Seed object pose with mono PnP on the primary (left) view, then refine with a joint Levenberg–Marquardt residual over both eyes. Triangulation is a separate primitive used for tests, square corners, and optional 3D–3D checks. Express all new public poses in the same OpenGL convention as mono after converting from OpenCV internals.

**Tech Stack:** Rust `pnp-core` (`no_std` + `alloc`, nalgebra, libm); existing `Camera`, `Pose`, `solve_pnp`, iterative LM patterns; C FFI (`cbindgen`); Python Maturin/PyO3; GitHub Actions CI.

## Global Constraints

- Mono APIs (`solve_pnp`, `Camera`, square mono entry points) stay source-compatible.
- `pnp-core` remains `no_std` + `alloc` (use `libm`, not `std` math).
- Image pixels: OpenCV (top-left, +Y down). Distortion: OpenCV coeff order.
- **Primary frame = left camera.** Extrinsics: `right_from_left` = pose of the right camera expressed in the left camera frame (OpenCV-style stereo).
- Algebraic solvers run in OpenCV camera frame; public `solve_*` results convert to OpenGL via `pose_tools::from_opencv_to_opengl`.
- Distortion handling v1: **undistort pixels first**, then ideal pinhole project in residuals (same as mono today).
- No stereo matching, disparity maps, or rectification pipelines in scope.
- New code needs unit tests; synthetic stereo fixtures preferred over OpenCV stereo deps.
- License MIT; update CHANGELOG under Unreleased; public items need rustdoc.
- CI: `cargo test --workspace --exclude pnplab-python` must stay green; extend Python job when Python APIs land.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `crates/pnp-core/src/stereo.rs` | `StereoRig`, stereo observation types, validation |
| `crates/pnp-core/src/triangulate.rs` | Midpoint / linear triangulation in left frame |
| `crates/pnp-core/src/absolute_orientation.rs` | Umeyama 3D–3D rigid fit (optional seed / tests) |
| `crates/pnp-core/src/stereo_solve.rs` | `solve_pnp_stereo`, joint LM residual |
| `crates/pnp-core/src/square_pose.rs` | Add stereo square entry points (or `square_pose_stereo.rs` if file grows large) |
| `crates/pnp-core/src/lib.rs` | Module exports |
| `crates/pnp-core/src/types.rs` | Shared types only if needed (prefer stereo-local types first) |
| `crates/pnp-core/tests/stereo.rs` | Integration-style synthetic stereo tests |
| `crates/pnp-ffi/src/lib.rs` + `include/pnp.h` | C ABI for rig + stereo solve |
| `bindings/python/...` | Python surface after core is solid |
| `README.md`, `CHANGELOG.md` | User-facing docs |

**Out of scope files:** Expo dual-camera UI, WIT stereo (add only if core + Python + FFI done and needed).

---

### Task 1: StereoRig types and frame conventions

**Files:**
- Create: `crates/pnp-core/src/stereo.rs`
- Modify: `crates/pnp-core/src/lib.rs`
- Test: unit tests inside `stereo.rs`

**Interfaces:**
- Consumes: `Camera`, `Pose`, `PnpError`, `Vector2` from existing core
- Produces:
  - `StereoRig { left: Camera, right: Camera, right_from_left: Pose }`
  - `StereoRig::new(...) -> Result<Self, PnpError>`
  - `StereoRig::baseline_length(&self) -> f64`
  - `StereoLandmarkObservation { id: String, left: Option<Vector2>, right: Option<Vector2> }`

- [ ] **Step 1: Write the failing tests**

Add `crates/pnp-core/src/stereo.rs` with tests first (module can compile with stubs):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    #[test]
    fn stereo_rig_rejects_near_zero_baseline() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let right_from_left = Pose::identity(); // baseline 0
        assert!(StereoRig::new(left, right, right_from_left).is_err());
    }

    #[test]
    fn stereo_rig_accepts_horizontal_baseline() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let right_from_left = Pose::new(
            Vector3::new(0.12, 0.0, 0.0), // 12 cm baseline along +X in left frame
            Quaternion::identity(),
        );
        let rig = StereoRig::new(left, right, right_from_left).unwrap();
        assert!((rig.baseline_length() - 0.12).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p pnp-core stereo --locked
```

Expected: compile error (`StereoRig` not found) or FAIL.

- [ ] **Step 3: Implement types**

```rust
//! Calibrated stereo rig: two monocular cameras + fixed extrinsics.
//!
//! # Frames
//! - **Left camera** is the primary frame for triangulation and stereo PnP seeds.
//! - `right_from_left` is the pose of the **right** camera expressed in the
//!   **left** camera frame: a point in left coords maps to right as
//!   `X_right = R * X_left + t` where `(R,t)` come from `right_from_left`
//!   in OpenCV convention when used inside solvers.
//!
//! Callers building a rig from device calibration must convert into this
//! convention before constructing [`StereoRig`].

use crate::camera::Camera;
use crate::types::{PnpError, Pose, Quaternion, Vector2, Vector3};
use alloc::string::String;

const MIN_BASELINE: f64 = 1e-6;

/// Calibrated stereo pair.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoRig {
    pub left: Camera,
    pub right: Camera,
    /// Pose of the right camera in the left camera frame.
    pub right_from_left: Pose,
}

/// Per-landmark observations in one or both eyes (pixel coords, possibly distorted).
#[derive(Debug, Clone, PartialEq)]
pub struct StereoLandmarkObservation {
    pub id: String,
    pub left: Option<Vector2>,
    pub right: Option<Vector2>,
}

impl StereoRig {
    pub fn new(
        left: Camera,
        right: Camera,
        right_from_left: Pose,
    ) -> Result<Self, PnpError> {
        let baseline = right_from_left.position.length();
        if !baseline.is_finite() || baseline < MIN_BASELINE {
            return Err(PnpError::SolverFailed);
        }
        // Rotation should be finite unit-ish; normalize if needed later.
        if !right_from_left.rotation.norm().is_finite() {
            return Err(PnpError::SolverFailed);
        }
        Ok(Self {
            left,
            right,
            right_from_left,
        })
    }

    /// Euclidean length of the left→right camera translation.
    pub fn baseline_length(&self) -> f64 {
        self.right_from_left.position.length()
    }
}
```

Wire in `lib.rs`:

```rust
pub mod stereo;
pub use stereo::{StereoLandmarkObservation, StereoRig};
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p pnp-core stereo --locked
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/pnp-core/src/stereo.rs crates/pnp-core/src/lib.rs
git commit -m "feat(core): add StereoRig and stereo observation types"
```

---

### Task 2: Pixel → ray and transform helpers for stereo

**Files:**
- Modify: `crates/pnp-core/src/stereo.rs` (or small helpers in `camera.rs` if cleaner)
- Modify: `crates/pnp-core/src/pose_tools.rs` if transform helpers belong there

**Interfaces:**
- Consumes: `Camera::undistort_pixel`, `Pose`, `Ray3`
- Produces:
  - `Camera::unproject_opencv_ray(pixel) -> Ray3` — **OpenCV** camera frame (+Z forward), origin at camera, direction unnormalized `(x, y, 1)` style after undistort:
    - `x = (u-cx)/fx`, `y = (v-cy)/fy`, `z = 1`
  - `transform_point(pose: &Pose, p: Vector3) -> Vector3` applying `R*p + t` with Hamilton quat (OpenCV/OpenGL-agnostic math on the pose components as stored)
  - Document that stereo internal math uses OpenCV frame poses for `right_from_left`

**Important:** Existing `unproject_opengl_ray` flips Y and uses `z = -1`. Stereo triangulation must **not** use that; add an OpenCV-frame unproject used only by stereo/triangulate.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn opencv_ray_principal_point_is_forward_z() {
    let cam = Camera::pinhole(100.0, 100.0, 50.0, 40.0).unwrap();
    let ray = cam.unproject_opencv_ray(Vector2::new(50.0, 40.0));
    assert!((ray.direction.x).abs() < 1e-12);
    assert!((ray.direction.y).abs() < 1e-12);
    assert!((ray.direction.z - 1.0).abs() < 1e-12);
}

#[test]
fn transform_point_translates() {
    let pose = Pose::new(Vector3::new(1.0, 2.0, 3.0), Quaternion::identity());
    let out = transform_point(&pose, Vector3::new(0.0, 0.0, 0.0));
    assert!((out.x - 1.0).abs() < 1e-12);
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test -p pnp-core opencv_ray --locked
```

- [ ] **Step 3: Implement**

In `camera.rs`:

```rust
/// Pixel → ray in **OpenCV** camera frame (look +Z, Y down in image).
/// After undistort: direction `( (u-cx)/fx, (v-cy)/fy, 1 )`, origin zero.
pub fn unproject_opencv_ray(&self, pixel: Vector2) -> Ray3 {
    let p = self.undistort_pixel(pixel);
    Ray3 {
        origin: Vector3::new(0.0, 0.0, 0.0),
        direction: Vector3::new(
            (p.x - self.cx) / self.fx,
            (p.y - self.cy) / self.fy,
            1.0,
        ),
    }
}
```

In `pose_tools.rs` (or `stereo.rs`):

```rust
/// Apply rigid transform: `R * p + t` using pose rotation as unit quaternion.
pub fn transform_point(pose: &Pose, p: Vector3) -> Vector3 {
    let uq = pose.rotation.normalize().to_na_unit();
    let rotated = uq * nalgebra::Vector3::new(p.x, p.y, p.z);
    Vector3::new(
        rotated.x + pose.position.x,
        rotated.y + pose.position.y,
        rotated.z + pose.position.z,
    )
}
```

- [ ] **Step 4: Tests PASS**

```bash
cargo test -p pnp-core --lib --locked
```

- [ ] **Step 5: Commit**

```bash
git add crates/pnp-core/src/camera.rs crates/pnp-core/src/pose_tools.rs crates/pnp-core/src/stereo.rs
git commit -m "feat(core): OpenCV-frame unproject and rigid transform helpers"
```

---

### Task 3: Midpoint triangulation

**Files:**
- Create: `crates/pnp-core/src/triangulate.rs`
- Modify: `crates/pnp-core/src/lib.rs`
- Test: unit tests in `triangulate.rs`

**Interfaces:**
- Consumes: `StereoRig`, `Vector2`, `Ray3`, OpenCV unproject, `transform_point`, `invert_pose`
- Produces:
  - `triangulate_midpoint(rig, left_px, right_px) -> Result<Vector3, PnpError>`
  - Point returned in **left camera OpenCV coordinates**

**Algorithm (midpoint of common perpendicular between two skew rays):**

1. `ray_l = left.unproject_opencv_ray(left_px)` → origin `O_l = 0`, dir `D_l` normalized  
2. `ray_r_cam = right.unproject_opencv_ray(right_px)` → dir in right frame  
3. Express right ray in left frame:  
   - `O_r = right_from_left.position`  
   - `D_r = R_right_from_left * d_right` (rotate direction only)  
4. Midpoint formula for skew lines; reject if nearly parallel (`|D_l × D_r|` small) or negative depth.

- [ ] **Step 1: Failing synthetic test**

```rust
#[test]
fn triangulate_known_point_in_front() {
    // Left at origin; right translated +0.1 m on X; identical K.
    // Point at (0, 0, 2) in left frame projects to both principal-ish rays with disparity.
    let left = Camera::pinhole(500.0, 500.0, 320.0, 240.0).unwrap();
    let right = left.clone();
    let rig = StereoRig::new(
        left,
        right,
        Pose::new(Vector3::new(0.1, 0.0, 0.0), Quaternion::identity()),
    )
    .unwrap();

    let p = Vector3::new(0.0, 0.0, 2.0);
    // Project with pinhole OpenCV: u = fx*X/Z+cx, v = fy*Y/Z+cy
    let ul = Vector2::new(320.0, 240.0); // principal point for (0,0,2)
    // In right camera: X_r = X_l - 0.1 => (-0.1, 0, 2)
    let ur = Vector2::new(500.0 * (-0.1) / 2.0 + 320.0, 240.0);

    let est = triangulate_midpoint(&rig, ul, ur).unwrap();
    assert!((est.x - 0.0).abs() < 1e-6);
    assert!((est.y - 0.0).abs() < 1e-6);
    assert!((est.z - 2.0).abs() < 1e-6);
}
```

- [ ] **Step 2: Run — FAIL**

```bash
cargo test -p pnp-core triangulate --locked
```

- [ ] **Step 3: Implement `triangulate_midpoint`**

Use standard skew-line midpoint; normalize directions with a small epsilon. Return `PnpError::SolverFailed` on parallel rays or non-finite result.

- [ ] **Step 4: Tests PASS + edge test**

```rust
#[test]
fn triangulate_rejects_parallel_rays() {
    // Same pixel in both views with pure baseline → may still work;
    // force parallel by identical directions after transform with zero disparity case
    // or call internal with parallel dirs — assert SolverFailed when cross ~ 0.
}
```

- [ ] **Step 5: Commit**

```bash
git add crates/pnp-core/src/triangulate.rs crates/pnp-core/src/lib.rs
git commit -m "feat(core): midpoint triangulation for stereo pairs"
```

---

### Task 4: Absolute orientation (Umeyama) — optional seed / validation

**Files:**
- Create: `crates/pnp-core/src/absolute_orientation.rs`
- Modify: `lib.rs`

**Interfaces:**
- Produces: `absolute_orientation(src: &[Vector3], dst: &[Vector3]) -> Result<Pose, PnpError>`
  - Finds `R, t` minimizing `|R * src_i + t - dst_i|` (Umeyama with scale fixed to 1)
  - Need ≥ 3 non-collinear points

- [ ] **Step 1: Failing test** — known rotation/translation recovery  
- [ ] **Step 2: FAIL**  
- [ ] **Step 3: Implement Umeyama (scale=1)** using nalgebra SVD  
- [ ] **Step 4: PASS**  
- [ ] **Step 5: Commit** `feat(core): Umeyama absolute orientation for 3D-3D pose`

This task supports testing and an alternate stereo seed (triangulate all → 3D–3D). Joint LM (Task 5) does not hard-depend on it, but keep it for robustness paths.

---

### Task 5: Joint stereo PnP (`solve_pnp_stereo`)

**Files:**
- Create: `crates/pnp-core/src/stereo_solve.rs`
- Modify: `crates/pnp-core/src/lib.rs`
- Modify: `crates/pnp-core/src/solve.rs` only if reusing match helpers
- Test: `crates/pnp-core/tests/stereo.rs` + unit tests in `stereo_solve.rs`

**Interfaces:**
- Consumes: `StereoRig`, `Landmark`, `StereoLandmarkObservation`, mono `solve_pnp`, `Camera` project/undistort, rodrigues, pose_tools
- Produces:
  - `solve_pnp_stereo(landmarks, observations, rig, method) -> Result<Pose, PnpError>`
  - `solve_pnp_stereo_camera_pose(...)` (invert, same as mono)
  - Returned pose: **object pose in OpenGL**, primary view = left (same meaning as mono `solve_pnp`)

**Algorithm:**

1. Build left-only mono correspondences from observations with `left.is_some()`.  
2. Seed: `solve_pnp(landmarks, left_obs, &rig.left, method)` → OpenGL pose → convert to OpenCV for residual math via `from_opengl_to_opencv`.  
3. Joint LM over rvec/tvec (OpenCV object-in-left-camera pose):
   - For each observation with left pixel: residual vs `project(left, R*X+t)`  
   - For each with right pixel: `X_left = R*X+t`, `X_right = transform_point(right_from_left, X_left)`, residual vs `project(right, X_right)`  
   - Skip missing sides; require total residual count ≥ 6 (3 points × 2) or at least mono minimum if only one side.  
4. Convert refined OpenCV pose back to OpenGL for return.

**Jacobian:** Prefer finite differences on the 6 parameters first (simpler, robust for v1). Match mono iterative convergence tolerances order-of-magnitude (`1e-8` cost delta, ~100 iters). Optionally later analytic Jacobian.

- [ ] **Step 1: Failing integration test** (`tests/stereo.rs`)

Synthetic:

- Known object square / 3D points  
- Known left camera pose  
- Known baseline  
- Project with both cameras (pinhole, no noise)  
- `solve_pnp_stereo` recovers pose within `1e-3` position / `1e-3` rad  

Also:

- Stereo beats mono on noisy depth direction (optional, looser)  
- Landmarks only in left still succeed (degrades to mono)  
- Mismatched ids → `MismatchedCounts`

- [ ] **Step 2: FAIL**

```bash
cargo test -p pnp-core --test stereo --locked
```

- [ ] **Step 3: Implement joint LM**

Structure:

```rust
pub fn solve_pnp_stereo(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> { ... }
```

Match by id: for each observation, find landmark by id (same as mono). Collect residuals as `Vec<f64>` length `2 * n_projections`.

- [ ] **Step 4: PASS all stereo + existing mono tests**

```bash
cargo test -p pnp-core --locked
cargo check -p pnp-core --no-default-features --locked
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(core): joint stereo PnP with mono seed and LM refine"
```

---

### Task 6: Stereo square-marker pose

**Files:**
- Modify: `crates/pnp-core/src/square_pose.rs` **or** Create: `crates/pnp-core/src/square_pose_stereo.rs`
- Modify: `lib.rs` exports
- Test: extend `tests/square_pose.rs` or `tests/stereo.rs`

**Interfaces:**
- Produces: `estimate_square_pose_from_stereo_pixels(left_corners, right_corners, physical_size, rig) -> Result<SquarePoseEstimate, PnpError>`
  - Corners: `[Vector2; 4]` TL,TR,BR,BL each eye  
  - Strategy v1 (simple, accurate with good calib):
    1. Triangulate each of 4 corners → 3D in left frame  
    2. Fit square pose from 3D corners (reuse geometry from existing `pose_from_points` / side-length constraints in `square_pose.rs` — extract shared helpers if needed)  
  - Strategy v2 (optional follow-up): joint ray residual using both cameras  

Prefer **triangulate + existing square geometry** for v1 to avoid duplicating LM.

- [ ] **Step 1: Failing synthetic stereo square test**  
- [ ] **Step 2: FAIL**  
- [ ] **Step 3: Implement triangulate-4 + pose_from_points path; confidence from corner error**  
- [ ] **Step 4: PASS**  
- [ ] **Step 5: Commit** `feat(core): stereo square pose from dual-view corners`

---

### Task 7: C FFI for stereo

**Files:**
- Modify: `crates/pnp-ffi/src/lib.rs`
- Regenerated: `crates/pnp-ffi/include/pnp.h`
- Copy headers into `bindings/expo-pnp/**` when regenerating
- Test: FFI unit tests in `pnp-ffi`

**Interfaces (C):**

```c
typedef struct pnp_stereo_rig_t {
  pnp_camera_t left;
  pnp_camera_t right;
  pnp_pose_t right_from_left;
} pnp_stereo_rig_t;

typedef struct pnp_stereo_observation_t {
  const char *id;
  // 1 = present; 0 = missing. When missing, position ignored.
  int has_left;
  pnp_vector2_t left;
  int has_right;
  pnp_vector2_t right;
} pnp_stereo_observation_t;

pnp_result_t pnp_solve_stereo(
  const pnp_landmark_t *landmarks, uintptr_t n_lm,
  const pnp_stereo_observation_t *obs, uintptr_t n_obs,
  const pnp_stereo_rig_t *rig,
  pnp_method_t method);

pnp_result_t pnp_triangulate(
  pnp_vector2_t left, pnp_vector2_t right,
  const pnp_stereo_rig_t *rig,
  pnp_vector3_t *out_point); // or return struct with error
```

- [ ] **Step 1: Failing FFI test** (synthetic, mirror core)  
- [ ] **Step 2: Implement conversion + calls; `# Safety` docs**  
- [ ] **Step 3: `cargo test -p pnp-ffi --locked`; regenerate header**  
- [ ] **Step 4: Commit** `feat(ffi): stereo rig, triangulate, and solve_stereo C API`

---

### Task 8: Python bindings

**Files:**
- Modify: `bindings/python/src/lib.rs`
- Modify: `bindings/python/python/auki_pnplab/__init__.py`, `__init__.pyi`
- Modify: `bindings/python/tests/test_pnplab.py` or `test_stereo.py`
- Modify: `bindings/python/README.md`

**Interfaces:**

```python
def solve_pnp_stereo(landmarks, observations, rig, method="iterative") -> Pose: ...
def triangulate(left_pixel, right_pixel, rig) -> dict: ...  # {x,y,z}

# rig = {
#   "left": {"fx", "fy", "cx", "cy", "dist"?},
#   "right": {...},
#   "right_from_left": {"position": {...}, "rotation": {...}},
# }
# observations = [{"id", "left": [u,v]|None, "right": [u,v]|None}, ...]
```

- [ ] **Step 1: Failing pytest**  
- [ ] **Step 2: Parse rig/observations; call core**  
- [ ] **Step 3: `./scripts/check-python.sh` PASS**  
- [ ] **Step 4: Commit** `feat(python): stereo PnP and triangulation bindings`

---

### Task 9: Docs, changelog, CI smoke

**Files:**
- Modify: `README.md` (stereo section + conventions table)
- Modify: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md` if module list changes
- Modify: `.github/workflows/ci.yml` only if new jobs needed (usually not)

- [ ] **Step 1: Document frames, `right_from_left`, partial observations, OpenGL return**  
- [ ] **Step 2: CHANGELOG Unreleased bullets**  
- [ ] **Step 3: Full verification**

```bash
cargo fmt --all -- --check
cargo test --workspace --locked --exclude pnplab-python
cargo check -p pnp-core --no-default-features --locked
./scripts/check-python.sh
```

- [ ] **Step 4: Commit** `docs: stereo PnP usage and changelog`

---

## Implementation notes (read before Task 5)

### Pose composition for right residual

Object point `X_o` in object frame. Object-in-left OpenCV pose `(R, t)`:

```text
X_left = R * X_o + t
X_right = R_rl * X_left + t_rl
```

where `(R_rl, t_rl)` come from `rig.right_from_left` **stored and interpreted in OpenCV**.  
When users pass OpenGL poses into the rig by mistake, results will be wrong — document conversion helpers if product poses are OpenGL (`from_opengl_to_opencv` on extrinsics).

**Recommendation:** Store `right_from_left` in OpenCV convention in core; provide `StereoRig::from_opengl_extrinsics(...)` helper if needed later.

### Seeding when left view is sparse

If left has &lt; min points but right has enough: seed with mono PnP on right, then convert pose into left frame via extrinsics before joint refine. Implement if tests need it; otherwise require left seed and return `InsufficientPoints`.

### Error mapping

| Condition | Error |
|-----------|--------|
| Baseline too small | `SolverFailed` |
| &lt; min total projections | `InsufficientPoints` |
| Id mismatch | `MismatchedCounts` |
| LM / triangulate numeric fail | `SolverFailed` |

---

## Self-review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| StereoRig + conventions | 1 |
| OpenCV rays / transforms | 2 |
| Triangulate | 3 |
| 3D–3D absolute orientation | 4 |
| Joint stereo landmark PnP | 5 |
| Stereo square | 6 |
| C FFI | 7 |
| Python | 8 |
| Docs / CI verify | 9 |
| Mono unbroken | Global + Task 5/9 full test suite |

**Placeholders:** None intentional; finite-diff Jacobian called out as explicit v1 choice.

**Type consistency:** `StereoRig.right_from_left: Pose`, observations use `Option<Vector2>`, public stereo solves return OpenGL `Pose` like mono.

---

## Out of scope (do not implement in this plan)

- Stereo rectification (`stereoRectify`)  
- Disparity / block matching  
- Analytic stereo Jacobian (unless finite-diff is too slow in practice)  
- Expo dual-camera product wiring  
- WIT stereo exports  
- Multi-camera N&gt;2  

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-18-stereo-pnp.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — run tasks in this session with executing-plans checkpoints  

Which approach?
