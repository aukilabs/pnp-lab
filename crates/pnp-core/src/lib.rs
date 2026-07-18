//! # pnp-core
//!
//! Pure-Rust [Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point)
//! solvers for monocular pose estimation.
//!
//! ## Capabilities
//!
//! - Classic PnP from 3D landmarks and 2D image observations
//!   ([`solve_pnp`], [`solve_pnp_camera_pose`])
//! - Square-marker pose from four corner rays or image pixels
//!   ([`estimate_square_pose_from_rays`], [`estimate_square_pose_from_pixels`])
//! - Calibrated monocular [`Camera`] with optional Brown–Conrady distortion
//!
//! ## Coordinate conventions
//!
//! | Space | Convention |
//! |-------|------------|
//! | Image pixels | OpenCV: origin top-left, +X right, +Y down |
//! | Algebraic solvers | OpenCV camera frame (+Z forward) |
//! | [`solve_pnp`] result | **OpenGL** object pose (Y-up, Z-backward) |
//! | Distortion coeffs | OpenCV order `k1,k2,p1,p2[,k3[,k4,k5,k6]]` |
//! | Square corners | TL → TR → BR → BL |
//!
//! ## `no_std`
//!
//! This crate is `no_std` with `alloc` when built without the default `std`
//! feature. Floating-point math uses `libm`.
//!
//! ## Example
//!
//! ```rust
//! use pnp_core::{
//!     solve_pnp, Camera, Landmark, LandmarkObservation, SolvePnpMethod, Vector2, Vector3,
//! };
//!
//! let camera = Camera::pinhole(800.0, 800.0, 320.0, 240.0).unwrap();
//! let landmarks = vec![
//!     Landmark { id: "0".into(), position: Vector3::new(-0.1, -0.1, 0.0) },
//!     Landmark { id: "1".into(), position: Vector3::new(0.1, -0.1, 0.0) },
//!     Landmark { id: "2".into(), position: Vector3::new(0.1, 0.1, 0.0) },
//!     Landmark { id: "3".into(), position: Vector3::new(-0.1, 0.1, 0.0) },
//! ];
//! // Synthetic observations for a camera looking at the square (illustration only).
//! let observations = vec![
//!     LandmarkObservation { id: "0".into(), position: Vector2::new(220.0, 140.0) },
//!     LandmarkObservation { id: "1".into(), position: Vector2::new(420.0, 140.0) },
//!     LandmarkObservation { id: "2".into(), position: Vector2::new(420.0, 340.0) },
//!     LandmarkObservation { id: "3".into(), position: Vector2::new(220.0, 340.0) },
//! ];
//! let _pose = solve_pnp(&landmarks, &observations, &camera, SolvePnpMethod::Iterative);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod absolute_orientation;
pub mod camera;
pub mod epnp;
pub mod iterative;
pub mod pose_tools;
pub mod rodrigues;
pub mod solve;
pub mod sqpnp;
pub mod square_pose;
pub mod stereo;
pub mod stereo_solve;
pub mod triangulate;
pub mod types;

pub use absolute_orientation::absolute_orientation;
pub use camera::Camera;
pub use solve::{camera_pose_from_solve_pnp_pose, solve_pnp, solve_pnp_camera_pose};
pub use square_pose::{estimate_square_pose_from_pixels, estimate_square_pose_from_rays};
pub use stereo::{StereoLandmarkObservation, StereoRig};
pub use stereo_solve::{solve_pnp_stereo, solve_pnp_stereo_camera_pose};
pub use triangulate::triangulate_midpoint;
pub use types::*;
