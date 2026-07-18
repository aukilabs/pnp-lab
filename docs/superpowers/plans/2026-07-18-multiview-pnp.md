# Multi-View PnP (N-View) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make joint multi-view PnP the core algorithm for N ≥ 1 cameras; implement stereo (N=2) and monocular (N=1) as thin special cases without changing mono public API behavior.

**Architecture:** Introduce `MultiViewRig` (ordered views; index 0 = primary) and sparse multi-view landmark observations. Generalize the existing stereo joint LM residual to loop over all views: for each observation in view `c`, transform the object point from primary frame into view `c` via fixed extrinsics, then pinhole-project. Seed with monocular `solve_pnp` on the primary view (or the richest view if primary is sparse—v2). Keep `StereoRig` / `solve_pnp_stereo` as wrappers that build a 2-view rig so existing tests and bindings keep working. Pairwise triangulation remains 2-view-only.

**Tech Stack:** Rust `pnp-core` (`no_std` + `alloc`, nalgebra, libm); existing `Camera`, `Pose`, `solve_pnp`, `stereo_solve` LM pattern; C FFI + Python as follow-on after core equivalence tests pass.

## Global Constraints

- **Mono public API unchanged in behavior:** `solve_pnp`, `solve_pnp_camera_pose`, mono square APIs keep the same signatures and numerical results (within float noise).
- **Stereo public API unchanged in behavior:** `StereoRig`, `StereoLandmarkObservation`, `solve_pnp_stereo`, `triangulate_midpoint`, `estimate_square_pose_from_stereo_pixels` remain; implement stereo solve **via** multiview.
- `pnp-core` stays `no_std` + `alloc`.
- Image / distortion / OpenCV↔OpenGL conventions unchanged.
- **Primary view = `views[0]`.** Extrinsic per view: **`from_primary`** maps a 3D point in the **primary** OpenCV frame into that view’s OpenCV frame: `X_c = R * X_primary + t`. For view 0, `from_primary` is identity. Stereo right maps to `from_primary = right_from_left` (same meaning as today).
- Distortion v1: undistort then ideal pinhole residual (same as stereo today).
- Joint refine uses finite-difference Jacobian over 6 pose params (same as stereo LM) unless a later task says otherwise.
- Equivalence gates are mandatory: N=1 vs mono, N=2 vs current stereo suite.
- Update CHANGELOG; public items get rustdoc.
- No stereo matching, rectification, free camera extrinsics BA, or N-view square required in this plan (stereo square stays as-is).

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `crates/pnp-core/src/multiview.rs` | `MultiViewRig`, `CameraView`, `MultiViewObservation`, validation, `StereoRig` → multiview conversion |
| `crates/pnp-core/src/multiview_solve.rs` | `solve_pnp_multiview`, `solve_pnp_multiview_camera_pose`, joint LM |
| `crates/pnp-core/src/stereo.rs` | Keep `StereoRig` / `StereoLandmarkObservation`; add `to_multiview()` / `From` |
| `crates/pnp-core/src/stereo_solve.rs` | Thin wrappers calling multiview (delete duplicated LM) |
| `crates/pnp-core/src/lib.rs` | Module exports |
| `crates/pnp-core/tests/multiview.rs` | N=1/N=2 equivalence + N=3 synthetic |
| `crates/pnp-core/tests/stereo.rs` | Must keep passing without semantic change |
| `crates/pnp-ffi/src/lib.rs` | Optional later task: multiview C API (not required for core merge) |
| `bindings/python/...` | Optional later: multiview Python (not required for core merge) |
| `README.md`, `CHANGELOG.md` | Document multiview as core; stereo as N=2 helper |

**Out of scope this plan:** Expo, WIT, rewriting triangulation to N rays, stereo square rewrite.

---

### Task 1: MultiViewRig types and conversion from StereoRig

**Files:**
- Create: `crates/pnp-core/src/multiview.rs`
- Modify: `crates/pnp-core/src/lib.rs`
- Modify: `crates/pnp-core/src/stereo.rs` (add conversion)
- Test: unit tests in `multiview.rs`

**Interfaces:**
- Consumes: `Camera`, `Pose`, `PnpError`, `Vector2`, `StereoRig`
- Produces:
  ```rust
  pub struct CameraView {
      pub camera: Camera,
      /// Maps primary OpenCV coords → this view: `X_view = R * X_primary + t`.
      /// Identity for the primary view.
      pub from_primary: Pose,
  }

  pub struct MultiViewRig {
      /// `views[0]` is primary. Length ≥ 1.
      pub views: Vec<CameraView>,
  }

  pub struct MultiViewObservation {
      pub id: String,
      /// Length must equal `rig.views.len()`. `None` = not observed in that view.
      pub pixels: Vec<Option<Vector2>>,
  }

  impl MultiViewRig {
      pub fn new(views: Vec<CameraView>) -> Result<Self, PnpError>;
      pub fn primary(&self) -> &CameraView;
      pub fn num_views(&self) -> usize;
      /// Convenience: two-view rig with identity primary extrinsics.
      pub fn from_stereo(stereo: &StereoRig) -> Self;
  }

  impl StereoRig {
      pub fn to_multiview(&self) -> MultiViewRig;
  }
  ```

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Quaternion, Vector3};
    use crate::{Camera, Pose, StereoRig};

    #[test]
    fn multiview_rejects_empty_views() {
        assert!(MultiViewRig::new(vec![]).is_err());
    }

    #[test]
    fn multiview_primary_from_primary_must_be_identity_translation() {
        let cam = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let bad = CameraView {
            camera: cam.clone(),
            from_primary: Pose::new(Vector3::new(1.0, 0.0, 0.0), Quaternion::identity()),
        };
        assert!(MultiViewRig::new(vec![bad]).is_err());
    }

    #[test]
    fn stereo_to_multiview_has_two_views() {
        let left = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
        let right = left.clone();
        let stereo = StereoRig::new(
            left,
            right,
            Pose::new(Vector3::new(0.12, 0.0, 0.0), Quaternion::identity()),
        )
        .unwrap();
        let mv = stereo.to_multiview();
        assert_eq!(mv.num_views(), 2);
        assert!((mv.views[0].from_primary.position.length()).abs() < 1e-15);
        assert!((mv.views[1].from_primary.position.x - 0.12).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test -p pnp-core multiview --locked
```

- [ ] **Step 3: Implement types**

Validation rules for `MultiViewRig::new`:

1. `views.len() >= 1`
2. View 0 `from_primary.position.length() < 1e-9` and rotation near identity (dot of quat with identity ≥ 1 - 1e-6, or check rvec ~ 0) — **or** force-normalize view 0 to identity on construction (prefer force-set identity for robustness)
3. All cameras already validated by `Camera::new` / `pinhole`
4. For non-primary views, `from_primary` rotation norm finite; optional: warn-level not required

`from_stereo`:

```rust
pub fn from_stereo(s: &StereoRig) -> Self {
    MultiViewRig {
        views: vec![
            CameraView {
                camera: s.left.clone(),
                from_primary: Pose::identity(),
            },
            CameraView {
                camera: s.right.clone(),
                from_primary: s.right_from_left,
            },
        ],
    }
}
```

(Use `MultiViewRig::new(...).expect(...)` only in tests; `from_stereo` can call `new` and `unwrap` is OK if construction is infallible given a valid `StereoRig`, or return `Result`.)

Export from `lib.rs`:

```rust
pub mod multiview;
pub use multiview::{CameraView, MultiViewObservation, MultiViewRig};
```

- [ ] **Step 4: Tests PASS**

```bash
cargo test -p pnp-core multiview --locked
cargo test -p pnp-core stereo_rig --locked
```

- [ ] **Step 5: Commit**

```bash
git add crates/pnp-core/src/multiview.rs crates/pnp-core/src/lib.rs crates/pnp-core/src/stereo.rs
git commit -m "feat(core): MultiViewRig types and StereoRig conversion"
```

---

### Task 2: `solve_pnp_multiview` joint LM (port from stereo)

**Files:**
- Create: `crates/pnp-core/src/multiview_solve.rs`
- Modify: `crates/pnp-core/src/lib.rs`
- Test: unit tests in `multiview_solve.rs` + start `tests/multiview.rs`

**Interfaces:**
- Consumes: `MultiViewRig`, `MultiViewObservation`, `Landmark`, `solve_pnp`, `pose_tools`, `rodrigues`, `transform_point`, `Camera` undistort/project pattern from `stereo_solve.rs`
- Produces:
  ```rust
  pub fn solve_pnp_multiview(
      landmarks: &[Landmark],
      observations: &[MultiViewObservation],
      rig: &MultiViewRig,
      method: SolvePnpMethod,
  ) -> Result<Pose, PnpError>;

  pub fn solve_pnp_multiview_camera_pose(
      landmarks: &[Landmark],
      observations: &[MultiViewObservation],
      rig: &MultiViewRig,
      method: SolvePnpMethod,
  ) -> Result<Pose, PnpError>;
  ```

**Algorithm (copy structure from `stereo_solve.rs`, generalize loops):**

1. Validate `observations[i].pixels.len() == rig.num_views()` else `MismatchedCounts`
2. Match landmark by id for each observation
3. Undistort each present pixel with `rig.views[c].camera.undistort_pixel`
4. Count projections; require `2 * n_proj >= 6`
5. Seed: build mono lists from **primary** view pixels only → `solve_pnp(..., &rig.primary().camera, method)`  
   - If primary has zero observations: return `InsufficientPoints` (v1; same as stereo left-only seed rule)
6. Convert seed OpenGL → OpenCV; LM on rvec/tvec
7. Residual for each pair and each view `c` with a pixel:
   - `X_primary = R * X_object + t`
   - `X_c = transform_point(&views[c].from_primary, X_primary)`
   - residual = project(views[c].camera, X_c) - uv
8. Return refined pose OpenCV → OpenGL

Reuse LM constants and finite-diff Jacobian pattern from `stereo_solve.rs` (`MAX_LM_ITERS`, `FD_EPS`, etc.).

- [ ] **Step 1: Failing test — N=1 matches mono on reference set0**

In `tests/multiview.rs` (or unit test with synthetic data):

```rust
#[test]
fn multiview_n1_matches_solve_pnp_synthetic() {
    // Same synthetic used in stereo tests: single camera, known pose.
    // pose_mv = solve_pnp_multiview(..., rig_n1, Iterative)
    // pose_mono = solve_pnp(...)
    // assert position error < 1e-6 and rotation angle < 1e-6
}
```

- [ ] **Step 2: Run — FAIL**

```bash
cargo test -p pnp-core --test multiview --locked
```

- [ ] **Step 3: Implement multiview_solve.rs**  
  Port helpers from stereo_solve (`project_pinhole`, `pose_to_rvec_tvec`, LM loop) as private functions in the new module (do not leave duplication permanently—Task 3 deletes stereo’s copy).

- [ ] **Step 4: PASS N=1 test + full pnp-core**

```bash
cargo test -p pnp-core --locked
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(core): solve_pnp_multiview joint LM for N views"
```

---

### Task 3: Rewrite stereo solve as multiview wrapper + N=2 equivalence

**Files:**
- Modify: `crates/pnp-core/src/stereo_solve.rs` (thin wrappers)
- Modify: `crates/pnp-core/tests/stereo.rs` (must still pass unchanged expectations)
- Modify: `crates/pnp-core/tests/multiview.rs` (N=2 equivalence test)

**Interfaces:**
- `solve_pnp_stereo` becomes:

```rust
pub fn solve_pnp_stereo(
    landmarks: &[Landmark],
    observations: &[StereoLandmarkObservation],
    rig: &StereoRig,
    method: SolvePnpMethod,
) -> Result<Pose, PnpError> {
    let mv_rig = rig.to_multiview();
    let mv_obs: Vec<MultiViewObservation> = observations
        .iter()
        .map(|o| MultiViewObservation {
            id: o.id.clone(),
            pixels: vec![o.left, o.right],
        })
        .collect();
    solve_pnp_multiview(landmarks, &mv_obs, &mv_rig, method)
}
```

Same for `solve_pnp_stereo_camera_pose`.

Delete private LM/residual code from `stereo_solve.rs` once wrappers work.

- [ ] **Step 1: Add equivalence test**

```rust
#[test]
fn multiview_n2_matches_stereo_api() {
    // Build same landmarks/obs/rig as stereo recovery test
    // pose_s = solve_pnp_stereo(...)
    // pose_m = solve_pnp_multiview(..., stereo.to_multiview(), converted obs)
    // assert nearly equal
}
```

- [ ] **Step 2: Rewrite stereo_solve; run**

```bash
cargo test -p pnp-core --test stereo --locked
cargo test -p pnp-core --test multiview --locked
cargo test -p pnp-core --locked
```

Expected: all existing stereo tests still pass (behavior gate).

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(core): implement stereo PnP via multiview solver"
```

---

### Task 4: N=3 synthetic recovery test

**Files:**
- Modify: `crates/pnp-core/tests/multiview.rs`

**Interfaces:** none new

- [ ] **Step 1: Write N=3 test**

Construct:

- Primary camera at identity  
- View 1: +0.1 m X translation  
- View 2: +0.05 m Y translation (or small rotation)  
- Same K for simplicity  
- Known object points + known object-in-primary OpenCV pose  
- Project to all three views with pinhole  
- `solve_pnp_multiview` recovers pose within `1e-3` position and `1e-3` rad  

Also:

```rust
#[test]
fn multiview_partial_observations() {
    // Landmark visible only in views 0 and 2 (pixels[1] = None) still recovers
}
```

- [ ] **Step 2: Run — should PASS with Task 2 implementation**

```bash
cargo test -p pnp-core --test multiview --locked
```

If it fails, fix multiview residual indexing (not stereo wrappers).

- [ ] **Step 3: Commit**

```bash
git commit -m "test(core): N=3 multiview PnP recovery and partial views"
```

---

### Task 5: Seed improvement (optional but recommended)

**Files:**
- Modify: `crates/pnp-core/src/multiview_solve.rs`
- Test: `tests/multiview.rs`

**Behavior:**

- v1 (Task 2): seed only on primary; if primary empty → `InsufficientPoints`
- v2 (this task): if primary lacks enough points, pick the view with the most observations ≥ min for `method`; run mono PnP in that view’s frame; convert object pose into primary frame:

```text
// Object-in-view_c (OpenCV): T_c
// from_primary for view c: T_{c←primary}  (X_c = R X_p + t)
// We need object-in-primary: T_p such that X_c = T_{c←primary} * T_p * X_o
// => T_p = inv(T_{c←primary}) * T_c
```

Use `invert_pose` + compose transforms (add `compose_poses(a, b)` in `pose_tools` if missing: apply b then a, document order).

- [ ] **Step 1: Failing test** — primary has no pixels, view 1 has all points; still recovers  
- [ ] **Step 2: Implement seed selection + pose transport to primary**  
- [ ] **Step 3: PASS; stereo left-empty still fails if only right (unless right has enough—should now succeed)**  
  - Update stereo docs if right-only now works  
- [ ] **Step 4: Commit** `feat(core): multiview seed from richest view when primary sparse`

If time-boxed, this task can be deferred; note in CHANGELOG as follow-up.

---

### Task 6: Docs and changelog

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `crates/pnp-core/src/lib.rs` crate-level docs (mention multiview)
- Modify: `CONTRIBUTING.md` module list if present

- [ ] **Step 1: Document**

  - Multi-view is the general joint-PnP API  
  - Stereo = N=2 helper  
  - Extrinsic: `from_primary`  
  - Observation: aligned `pixels` vec with `None` holes  
  - Example: build 2-view rig via `MultiViewRig::from_stereo` or manual 3-view  

- [ ] **Step 2: CHANGELOG Unreleased**

  - Added `MultiViewRig`, `solve_pnp_multiview`  
  - Stereo solve now delegates to multiview (no intentional behavior change)  

- [ ] **Step 3: Full verify**

```bash
cargo fmt --all -- --check
cargo test --workspace --locked --exclude pnpkit-python
cargo check -p pnp-core --no-default-features --locked
./scripts/check-python.sh
```

- [ ] **Step 4: Commit** `docs: multi-view PnP as core joint solver`

---

### Task 7 (optional follow-on): Python multiview bindings

**Files:** `bindings/python/...`

Only after Tasks 1–6 merge if product needs N>2 from Python.

```python
solve_pnp_multiview(landmarks, observations, rig, method="iterative")
# rig = {"views": [{"camera": {...}, "from_primary": pose_dict}, ...]}
# observations = [{"id": "0", "pixels": [[u,v], None, [u2,v2]]}, ...]
```

Stereo Python APIs stay; no need to remove.

- [ ] Implement + `check-python.sh` + commit `feat(python): multi-view PnP bindings`

---

### Task 8 (optional follow-on): C FFI multiview

**Files:** `crates/pnp-ffi/src/lib.rs`, header copy to Expo

```c
// Flexible but C-unfriendly: pass view count + flat arrays, or max N=8 fixed.
// Prefer: document stereo FFI as N=2 path; multiview FFI when a C caller needs it.
```

Defer unless a C client requires N>2.

---

## Implementation notes

### Residual indexing

Do not assume 2 residuals per landmark. Count only present pixels when allocating residual vector (same as stereo’s `count_projections`).

### Identity primary

Forcing `views[0].from_primary = Pose::identity()` in `MultiViewRig::new` avoids brittle float checks and matches stereo semantics.

### Duplication budget

Accept temporary copy of LM code in Task 2; Task 3 must remove the stereo duplicate so there is **one** joint residual implementation.

### Equivalence tolerances

| Comparison | Position | Rotation |
|------------|----------|----------|
| N=1 vs mono (same method, same points) | &lt; 1e-6 | &lt; 1e-6 rad |
| N=2 multiview vs stereo API | &lt; 1e-6 | &lt; 1e-6 rad |
| Synthetic recovery (noise-free) | &lt; 1e-3 | &lt; 1e-3 rad |

### Pose composition helper

If missing, add to `pose_tools.rs`:

```rust
/// Compose transforms: `out = a * b` means apply `b` first, then `a`
/// (standard matrix multiplication on SE(3)).
pub fn compose_poses(a: &Pose, b: &Pose) -> Pose;
```

Needed for Task 5 seed transport: `T_primary = inv(from_primary_c) * T_view_c`.

---

## Self-review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| MultiViewRig / observations | 1 |
| Joint N-view LM | 2 |
| Stereo as wrapper | 3 |
| N=1 ≡ mono | 2 |
| N=2 ≡ stereo suite | 3 |
| N=3 + partial views | 4 |
| Better seed | 5 (optional) |
| Docs | 6 |
| Python/FFI multiview | 7–8 optional |

**Placeholder scan:** none intentional.

**Type consistency:** `from_primary` everywhere; stereo `right_from_left` maps to view 1 `from_primary`; OpenGL public returns preserved.

---

## Out of scope

- Changing stereo square to multiview  
- Free extrinsics / full BA  
- Analytic Jacobian  
- Rectification / matching  

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-18-multiview-pnp.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement in this session with checkpoints  

Which approach?
