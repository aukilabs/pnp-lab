# Multi-View Monocular Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-view intrinsic + Brown–Conrady distortion estimation to `pnp-core` (and thin bindings) so callers recover a calibrated `Camera` from many views of a known planar target—especially QR outer corners (TL→TR→BR→BL + `physical_size`).

**Architecture:** Two-stage pipeline: (A) Zhang-style planar initialization (homographies → initial `K`, poses via existing `solve_pnp`), then (B) joint Levenberg–Marquardt over intrinsics, optional distortion, and per-view rvec/tvec minimizing OpenCV-frame reprojection residuals via `Camera::project`. No OpenCV runtime. Primary consumer is camcalibdb/qrkit (out of repo).

**Tech Stack:** Rust `pnp-core` (`no_std` + `alloc`, nalgebra, libm); reuse `Camera`, `solve_pnp`, `pose_tools`, LM patterns from `iterative.rs` / `multiview_solve.rs`; Python Maturin; C FFI + WIT/Expo as thin follow-ons.

## Global Constraints

- **No OpenCV runtime** in shipped code; nalgebra + libm only.
- **`pnp-core` remains `no_std` + `alloc`**.
- **Conventions (do not invent new ones):**
  - Image pixels: OpenCV (top-left origin, +X right, +Y down)
  - Distortion: OpenCV order `k1,k2,p1,p2[,k3[,k4,k5,k6]]`; lengths 0 / 4 / 5 / 8 only in `Camera` storage (see packing for `dist_len=2`)
  - `Camera::project`: OpenCV camera frame (+Z forward)
  - Square corners: **TL → TR → BR → BL**
  - **BA internal poses:** OpenCV object-in-camera (rvec/tvec)
  - **Public `CalibrationResult.object_poses`:** OpenGL object pose, **same as `solve_pnp`**
- **Skew:** always 0 (not estimated).
- **Single-view calibration:** not supported (`min_views` default 3).
- **Do not change** mono/stereo/multiview PnP APIs except additive re-exports.
- Prefer existing `PnpError` variants; avoid new FFI/WIT error enum variants in v1.
- Tests required; `cargo fmt`; public rustdoc; `CHANGELOG.md` Unreleased.
- CI: `cargo test --workspace --exclude pnpkit-python` must stay green; Python job when Python API lands.

## Locked design decisions

1. Skew = 0 always.  
2. Primary target: planar square / QR; general API takes arbitrary shared `object_points`.  
3. Default options: `fix_aspect_ratio = true`, `dist_len = 5`, `min_views = 3` (document that phones need ~8–15 diverse views in practice).  
4. `object_poses` in result: **OpenGL** (match `solve_pnp`). Optimize in OpenCV frame internally.  
5. Point ordering: parallel arrays (same order every view); no id-matching in v1.  
6. Square object model (meters if `physical_size` is meters), centered on origin, **Z = 0**, half-side `h = physical_size / 2`:

   | Corner | Object point |
   |--------|----------------|
   | TL | `(-h, +h, 0)` |
   | TR | `(+h, +h, 0)` |
   | BR | `(+h, -h, 0)` |
   | BL | `(-h, -h, 0)` |

   Rationale: local +X = TL→TR, local +Y toward top of marker (matches square-pose axis spirit). **Document this table in rustdoc**—do not use a third convention.

7. Distortion packing into `Camera.dist`:

   | `dist_len` (optimized) | Stored `Camera.dist` |
   |------------------------|----------------------|
   | 0 | `[]` |
   | 2 | `[k1,k2,0,0,0]` (length 5; only k1,k2 free in BA) |
   | 4 | `[k1,k2,p1,p2]` |
   | 5 | `[k1,k2,p1,p2,k3]` |
   | 8 | full rational 8 |

   Reject other `dist_len` with `SolverFailed`.

8. Implementation order: **pinhole BA green first** (`dist_len=0`), then free dist coeffs.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `crates/pnp-core/src/calibrate.rs` | Public API + Zhang init + BA + tests |
| `crates/pnp-core/src/lib.rs` | `mod calibrate` + re-exports |
| `crates/pnp-core/tests/calibrate.rs` | Optional integration/synthetic cases if unit file grows large |
| `bindings/python/...` | Thin API |
| `crates/pnp-ffi/src/lib.rs` + header | C API |
| `crates/pnp-wasm/wit/pnp.wit` + guest | Additive export (bump if needed per repo practice) |
| `bindings/expo-pnp/...` | Thin TS/native (optional same PR if time) |
| `README.md`, `CHANGELOG.md` | User docs |

Private helpers may live as `mod` subsections inside `calibrate.rs` for v1 (homography, zhang, lm, pack). Split only if file exceeds ~800–1000 lines.

---

### Task 1: Public types + validation + square object points

**Files:**
- Create: `crates/pnp-core/src/calibrate.rs` (types + validation + square points only)
- Modify: `crates/pnp-core/src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct CalibrateOptions {
      pub min_views: usize,              // default 3
      pub fix_aspect_ratio: bool,        // default true
      pub fix_principal_point: bool,     // default false
      pub dist_len: usize,               // 0|2|4|5|8
      pub max_iterations: usize,         // default 100
      pub function_tolerance: f64,       // default 1e-10
      pub rms_success_threshold: Option<f64>, // None = no hard fail on RMS
  }
  impl Default for CalibrateOptions { ... }

  pub struct CalibrationView {
      pub image_points: Vec<Vector2>,
  }

  pub struct CalibrationResult {
      pub camera: Camera,
      pub rms_reprojection_error: f64,
      pub per_view_rms: Vec<f64>,
      pub object_poses: Vec<Pose>,  // OpenGL object pose per used view
      pub views_used: usize,
  }

  /// Shared square object points TL..BL for `physical_size`.
  pub fn square_object_points(physical_size: f64) -> Result<[Vector3; 4], PnpError>;

  pub(crate) fn validate_calibrate_inputs(
      object_points: &[Vector3],
      views: &[CalibrationView],
      image_width: u32,
      image_height: u32,
      options: &CalibrateOptions,
  ) -> Result<(), PnpError>;
  ```

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn square_object_points_order_and_size() {
    let pts = square_object_points(0.2).unwrap();
    let h = 0.1;
    assert!((pts[0].x + h).abs() < 1e-12 && (pts[0].y - h).abs() < 1e-12); // TL
    assert!((pts[1].x - h).abs() < 1e-12 && (pts[1].y - h).abs() < 1e-12); // TR
    assert!((pts[2].x - h).abs() < 1e-12 && (pts[2].y + h).abs() < 1e-12); // BR
    assert!((pts[3].x + h).abs() < 1e-12 && (pts[3].y + h).abs() < 1e-12); // BL
    // side length
    let d = ((pts[1].x - pts[0].x).powi(2) + (pts[1].y - pts[0].y).powi(2)).sqrt();
    assert!((d - 0.2).abs() < 1e-12);
}

#[test]
fn validate_rejects_too_few_views_and_bad_dist_len() {
    let opts = CalibrateOptions { min_views: 3, ..Default::default() };
    let obj = square_object_points(0.1).unwrap();
    let views = vec![CalibrationView { image_points: vec![Vector2::new(0.0,0.0); 4] }; 2];
    assert_eq!(
        validate_calibrate_inputs(&obj, &views, 640, 480, &opts),
        Err(PnpError::InsufficientPoints)
    );
    let mut bad = CalibrateOptions::default();
    bad.dist_len = 3;
    let views3 = vec![CalibrationView { image_points: vec![Vector2::new(0.0,0.0); 4] }; 3];
    assert!(validate_calibrate_inputs(&obj, &views3, 640, 480, &bad).is_err());
}
```

- [ ] **Step 2: Run — FAIL**

```bash
cargo test -p pnp-core square_object --locked
```

- [ ] **Step 3: Implement types, `Default`, validation, `square_object_points`**

Validation rules:

| Check | Error |
|-------|--------|
| `image_width < 2` or `image_height < 2` | `SolverFailed` |
| `views.len() < options.min_views` | `InsufficientPoints` |
| any view `image_points.len() != object_points.len()` | `MismatchedCounts` |
| `object_points.len() < 4` | `InsufficientPoints` |
| non-finite coordinates | `SolverFailed` |
| `dist_len` not in `{0,2,4,5,8}` | `SolverFailed` |
| (square path later) `physical_size <= 0` or non-finite | `SolverFailed` |

- [ ] **Step 4: PASS + export from lib.rs**

```bash
cargo test -p pnp-core calibrate --locked
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(core): calibration types, validation, and square object points"
```

---

### Task 2: Homography DLT (planar Z=0)

**Files:**
- Modify: `crates/pnp-core/src/calibrate.rs` (private `homography` section)

**Interfaces:**
- Produces (crate-private):
  ```rust
  /// 3x3 row-major or nalgebra Matrix3; document storage.
  fn estimate_homography_dlt(
      object_xy: &[(f64, f64)], // Z=0 plane coords
      image_uv: &[Vector2],
  ) -> Result<nalgebra::Matrix3<f64>, PnpError>;
  ```

- [ ] **Step 1: Failing test** — known homography recovery

```rust
#[test]
fn homography_recovers_similarity() {
    // object square corners; H = K * [r1 r2 t] synthetic
    // apply H to project; estimate_homography_dlt; compare up to scale
}
```

- [ ] **Step 2: Implement DLT**  
  - Build 2n×9 design matrix, SVD null-space, reshape to 3×3, normalize `H[2,2]=1` if possible.  
  - Require ≥4 non-collinear points; reject degeneracy with `SolverFailed`.

- [ ] **Step 3: PASS**

- [ ] **Step 4: Commit** `feat(core): planar homography DLT for calibration init`

---

### Task 3: Zhang initial intrinsics

**Files:**
- Modify: `calibrate.rs`

**Interfaces:**
```rust
/// From ≥3 homographies (object Z=0 → image), recover fx,fy,cx,cy (skew=0).
fn zhang_initial_intrinsics(
    homographies: &[nalgebra::Matrix3<f64>],
    image_width: u32,
    image_height: u32,
) -> Result<(f64, f64, f64, f64), PnpError>;
```

- [ ] **Step 1: Failing synthetic test**  
  Generate 5+ poses, true K, planar points → Hᵢ via projection (pinhole) → Zhang recovers fx,fy,cx,cy within ~1% / few px.

- [ ] **Step 2: Implement Zhang closed form** (zero skew).  
  Fallback if ill-conditioned:
  ```rust
  fx = fy = max(w,h) as f64;
  cx = w as f64 * 0.5;
  cy = h as f64 * 0.5;
  ```
  (Call site may still use this fallback when homography stage fails.)

- [ ] **Step 3: PASS**

- [ ] **Step 4: Commit** `feat(core): Zhang initial K from planar homographies`

---

### Task 4: Parameter packing + residual/RMS using Camera::project

**Files:**
- Modify: `calibrate.rs`

**Interfaces:**
```rust
struct CalibState {
    // free intrinsic + dist parameters according to options
    // plus per-view rvec[3], tvec[3] in OpenCV convention
}

fn pack_state(...) -> Vec<f64>;
fn unpack_state(params: &[f64], ...) -> (Camera, Vec<( [f64;3], NaVector3<f64>)>);
fn camera_from_params(...) -> Result<Camera, PnpError>; // enforces dist packing table
fn rms_reprojection(
    object_points: &[Vector3],
    views: &[CalibrationView],
    camera: &Camera,
    poses_cv: &[(/*R,t OpenCV*/)],
) -> (f64, Vec<f64>);
```

**Pose conversion:**

- Seed from `solve_pnp` returns OpenGL → convert with `from_opengl_to_opencv` before packing rvec/tvec.  
- On success, convert each refined OpenCV pose to OpenGL for `CalibrationResult.object_poses`.

**Residuals for point j in view i:**

```text
X_cam = R_i * X_j + t_i   // OpenCV
u_hat = camera.project(X_cam)
e = u_hat - image_points[i][j]  // distorted space (project already distorts)
```

- [ ] **Step 1: Unit test** — pack/unpack roundtrip; RMS ~0 for perfect synthetic  
- [ ] **Step 2: Implement**  
- [ ] **Step 3: Commit** `feat(core): calibration state packing and RMS helpers`

---

### Task 5: LM joint refinement (pinhole only first)

**Files:**
- Modify: `calibrate.rs`

**Interfaces:**
```rust
fn refine_calibration_lm(
    object_points: &[Vector3],
    views: &[CalibrationView],
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
    camera0: Camera,
    poses_cv0: Vec<(/*rvec,tvec*/)>,
) -> Result<(Camera, Vec<Pose /*OpenGL*/>, f64, Vec<f64>), PnpError>;
```

**State vector (pinhole, free aspect, free principal):**  
`[fx, fy, cx, cy, r1(3), t1(3), …, rN(3), tN(3)]`

With `fix_aspect_ratio`: drop `fy` (set `fy=fx`).  
With `fix_principal_point`: lock cx,cy to `w/2,h/2` (or initial).

**Jacobian:** finite differences on free params (acceptable for v1). Document in module rustdoc.

**LM:** mirror damping style from `multiview_solve` / `iterative` (`lambda`, max iters, cost delta tolerance).

- [ ] **Step 1: Failing end-to-end pinhole test** (Task 6 will flesh calibrate_camera; here test refine in isolation or thin wrapper)

```rust
#[test]
fn refine_pinhole_from_noisy_init() {
    // true camera; synthetic views; start K off by 5%; after refine RMS < 1e-2 and fx error < 1e-3 rel
}
```

- [ ] **Step 2: Implement LM**  
- [ ] **Step 3: PASS**  
- [ ] **Step 4: Commit** `feat(core): pinhole multi-view calibration LM refine`

---

### Task 6: `calibrate_camera` full pipeline (init + BA)

**Files:**
- Modify: `calibrate.rs`
- Export public `calibrate_camera`

**Pipeline:**

1. `validate_calibrate_inputs`  
2. For each view: if planar Z≈0 for all object points, compute homography; else skip Zhang path for that view  
3. If ≥3 valid H: `zhang_initial_intrinsics`; else fallback K  
4. Apply option flags (fix aspect → fy=fx; fix pp → center)  
5. Init `Camera` with `dist` zeros of packed length  
6. Per view: build landmarks/observations → `solve_pnp(..., Iterative or EPnP)` → OpenCV rvec/tvec; drop views that fail PnP  
7. If remaining views < `min_views` → `InsufficientPoints`  
8. `refine_calibration_lm`  
9. If `rms_success_threshold` is `Some(t)` and RMS > t → `SolverFailed`  
10. Return `CalibrationResult`

- [ ] **Step 1: Failing public API test**

```rust
#[test]
fn calibrate_camera_recovers_pinhole_synthetic() {
    // N=10 views, varied tilt, 3x3 grid on Z=0, noise-free
    // options: fix_aspect_ratio=true, dist_len=0
    // assert fx/fy rel err < 1e-3, cx/cy < 0.5, RMS < 1e-2
}
```

- [ ] **Step 2: Implement `calibrate_camera`**  
- [ ] **Step 3: PASS**  
- [ ] **Step 4: Commit** `feat(core): calibrate_camera multi-view pinhole pipeline`

---

### Task 7: Distortion in BA (`dist_len` 2/4/5/8)

**Files:**
- Modify: `calibrate.rs` pack/unpack/LM free parameters

- [ ] **Step 1: Failing test** — ground-truth `Camera` with dist length 5; recover with `dist_len=5`  
  Use strong tilt diversity. Tolerances: RMS near 0; k1 within loose abs tol (e.g. 0.02) if identifiable.

- [ ] **Step 2: Extend state with free dist coeffs** (respect packing table; fixed zeros for unused slots when dist_len=2)

- [ ] **Step 3: PASS pinhole + distorted synthetic**

- [ ] **Step 4: Commit** `feat(core): joint calibration with Brown-Conrady distortion`

---

### Task 8: `calibrate_from_square_views` + error-case tests

**Files:**
- Modify: `calibrate.rs`
- Export from `lib.rs`

```rust
pub fn calibrate_from_square_views(
    corners_per_view: &[[Vector2; 4]],
    physical_size: f64,
    image_width: u32,
    image_height: u32,
    options: &CalibrateOptions,
) -> Result<CalibrationResult, PnpError> {
    if !physical_size.is_finite() || physical_size <= 0.0 {
        return Err(PnpError::SolverFailed);
    }
    let object = square_object_points(physical_size)?;
    let views: Vec<CalibrationView> = corners_per_view
        .iter()
        .map(|c| CalibrationView {
            image_points: c.to_vec(),
        })
        .collect();
    calibrate_camera(&object, &views, image_width, image_height, options)
}
```

- [ ] **Step 1: Tests**

```rust
#[test]
fn calibrate_from_square_views_noise_free() { ... }

#[test]
fn calibrate_from_square_views_rejects_bad_size() {
    assert!(calibrate_from_square_views(&[], 0.0, 640, 480, &CalibrateOptions::default()).is_err());
}

#[test]
fn calibrate_rejects_identical_poses_cleanly() {
    // all views same pose — should return Err(SolverFailed) or high RMS fail, not panic
}
```

- [ ] **Step 2: PASS**

- [ ] **Step 3: Commit** `feat(core): calibrate_from_square_views for QR/planar quads`

---

### Task 9: Python bindings

**Files:**
- `bindings/python/src/lib.rs`
- `bindings/python/python/auki_pnpkit/__init__.py`
- `__init__.pyi`
- `bindings/python/tests/test_calibrate.py`
- `bindings/python/README.md`

**API:**

```python
result = auki_pnpkit.calibrate_from_square_views(
    corners,              # sequence of 4 points or (N,4,2) array
    physical_size=0.05,
    image_size=(1920, 1080),  # or image_width=, image_height=
    fix_aspect_ratio=True,
    dist_len=5,
    min_views=3,
)
# result["camera"] = {fx, fy, cx, cy, dist}
# result["rms_reprojection_error"]
# result["per_view_rms"]
# result["object_poses"]
# result["views_used"]
```

Also export `calibrate_camera` if straightforward.

- [ ] **Step 1: Failing pytest** (noise-free square synthetic in Python or call into small fixture)  
- [ ] **Step 2: Implement parse + pyfunctions**  
- [ ] **Step 3: `./scripts/check-python.sh` PASS**  
- [ ] **Step 4: Commit** `feat(python): multi-view camera calibration bindings`

---

### Task 10: C FFI

**Files:**
- `crates/pnp-ffi/src/lib.rs`
- Regenerate `include/pnp.h`; copy to Expo header paths

**Suggested C surface (keep simple):**

```c
typedef struct pnp_calibrate_options_t {
  uintptr_t min_views;
  bool fix_aspect_ratio;
  bool fix_principal_point;
  uintptr_t dist_len;
  uintptr_t max_iterations;
  double function_tolerance;
  double rms_success_threshold; // <0 means disabled
} pnp_calibrate_options_t;

// Out: camera via pnp_camera_t, rms, per_view_rms buffer, poses buffer
pnp_error_t peyote_pnp_calibrate_from_square_views(
  const pnp_vector2_t *corners, // N*4
  uintptr_t num_views,
  double physical_size,
  uint32_t image_width,
  uint32_t image_height,
  const pnp_calibrate_options_t *options,
  pnp_camera_t *out_camera,
  double *out_rms,
  double *out_per_view_rms, // len num_views or views_used
  pnp_pose_t *out_poses,
  uintptr_t *out_views_used
);
```

(Document buffer ownership and max sizes; or return heap-less fixed max views e.g. 64.)

- [ ] Tests in pnp-ffi  
- [ ] Commit `feat(ffi): multi-view square calibration C API`

---

### Task 11: WIT / WASM (additive)

**Files:**
- `crates/pnp-wasm/wit/pnp.wit` — add records + `calibrate-from-square-views`  
- Prefer **additive** change under `auki:pnp@0.2.0` if tooling allows; else bump to `0.3.0` per repo practice  
- Regenerate bindings; implement guest  

- [ ] Smoke compile `cargo check -p pnp-wasm`  
- [ ] Commit `feat(wasm): export multi-view calibration`

If WASM regeneration is painful in CI, document as follow-up and keep Python+FFI as the binding acceptance bar for v1.

---

### Task 12: Expo thin wrappers (optional same PR)

**Files:** TS types + iOS/Android native methods calling FFI when present.

Web: throw “not supported” like other native-only methods.

Skip if timeboxed; README note “native via FFI / Python first”.

---

### Task 13: Docs, changelog, full verification

**Files:** `README.md`, `CHANGELOG.md`, module rustdoc

README section **Camera calibration (multi-view)**:

- Requires diverse views (tilts); 4 pts/view needs many angles  
- `fix_aspect_ratio` recommended for phones  
- Example: synthetic or QR-shaped call  
- Link to `calibrate_from_square_views`  

CHANGELOG Unreleased:

- Multi-view monocular calibration (`calibrate_camera`, `calibrate_from_square_views`)  
- Python / FFI as applicable  

**Verify:**

```bash
cargo fmt --all -- --check
cargo test --workspace --locked --exclude pnpkit-python
cargo check -p pnp-core --no-default-features --locked
./scripts/check-python.sh   # if Python touched
cargo test -p pnp-ffi --locked  # if FFI touched
```

- [ ] Commit `docs: multi-view camera calibration`

---

## Implementation notes for the agent

### Use existing pieces

| Need | Reuse |
|------|--------|
| Distorted projection | `Camera::project` |
| Distortion model | `Camera` / `distort_normalized` |
| Initial per-view pose | `solve_pnp` + `from_opengl_to_opencv` |
| rvec ↔ R | `rodrigues` |
| LM damping patterns | `multiview_solve.rs` / `iterative.rs` |

### Conditioning

- Synthetic sets **must** include non-frontal views (Zhang needs tilt).  
- Pure frontal parallel planes → expect `SolverFailed` or fallback + high RMS.  
- Scale fixed by `physical_size` / object metric.

### Finite-diff Jacobian

Acceptable for v1. If slow with many views×points, optimize later (analytic or denser structure). Open a PR note if wall time > few seconds for N=20, 4 pts.

### Error mapping (v1, no new enum)

| Case | `PnpError` |
|------|------------|
| Too few views/points | `InsufficientPoints` |
| Length mismatch | `MismatchedCounts` |
| Bad sizes, dist_len, non-finite, non-converge, RMS threshold | `SolverFailed` |

### Tolerance table (noise-free synthetic)

| Param | Tol |
|-------|-----|
| fx, fy | rel &lt; 1e-3 |
| cx, cy | &lt; 0.5 px |
| RMS | &lt; 1e-2 px |
| dist k1.. | case-dependent; assert finite + improves RMS vs pinhole init |

### Suggested internal task order (pinhole → dist)

1. Types/validation/square points  
2. Homography  
3. Zhang  
4. Pack/RMS  
5. LM pinhole  
6. `calibrate_camera`  
7. Distortion  
8. Square convenience  
9. Bindings + docs  

---

## Acceptance criteria (checklist)

- [ ] `calibrate_from_square_views` recovers known synthetic intrinsics (pinhole + dist=5) within tolerances  
- [ ] `calibrate_camera` works for ≥4 non-square planar points (e.g. 3×3 grid) in a unit test  
- [ ] Invalid inputs → clear `PnpError`, no panic  
- [ ] `cargo check -p pnp-core --no-default-features` OK  
- [ ] Python and/or FFI can call the API  
- [ ] Workspace tests pass  
- [ ] README + CHANGELOG updated  

---

## Out of scope

- QR detection, video I/O, phone DB, web UI (camcalibdb)  
- Single-view calibration  
- Online / rolling calibration  
- Multi-camera extrinsics  
- OpenCV runtime dependency  
- Making mono `solve_pnp` a multiview wrapper (separate decision; skip)

---

## Downstream consumer (context only)

camcalibdb will: qrkit corners → `calibrate_from_square_views` → store `Camera` + metadata. This plan unblocks that; do not implement camcalibdb here.

---

## Self-review

| Spec item | Task |
|-----------|------|
| Zhang + BA pipeline | 2–7 |
| Square convenience | 1, 8 |
| General object_points API | 6 |
| Distortion 0/2/4/5/8 packing | 1, 7 |
| OpenGL poses in result | 4, 6 |
| no_std / no OpenCV | Global |
| Python / FFI / WASM / Expo | 9–12 |
| Docs | 13 |
| Synthetic + error tests | 1, 6, 7, 8 |

**Placeholder scan:** none intentional.  
**Type consistency:** `CalibrationResult.camera: Camera`, poses OpenGL, pixels distorted OpenCV.

---

## Execution handoff

Plan complete and saved to:

**`docs/superpowers/plans/2026-07-18-multiview-camera-calibration.md`**

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement in this session with checkpoints  

Which approach?
