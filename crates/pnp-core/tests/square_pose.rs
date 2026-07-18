use pnp_core::{estimate_square_pose_from_rays, PnpError, Quaternion, Ray3, Vector3};

const EPS_POSITION: f64 = 1e-4;
const EPS_GEOMETRY: f64 = 1e-5;

fn add(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn scale(v: Vector3, s: f64) -> Vector3 {
    Vector3::new(v.x * s, v.y * s, v.z * s)
}

fn length(v: Vector3) -> f64 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

fn normalize(v: Vector3) -> Vector3 {
    let len = length(v);
    scale(v, 1.0 / len)
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn dot(a: Vector3, b: Vector3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn distance(a: Vector3, b: Vector3) -> f64 {
    length(sub(a, b))
}

fn reconstruct(ray: Ray3, distance: f64) -> Vector3 {
    add(ray.origin, scale(normalize(ray.direction), distance))
}

fn rotate_vector(q: Quaternion, v: Vector3) -> Vector3 {
    let vector_quat = Quaternion::new(v.x, v.y, v.z, 0.0);
    let rotated = q.multiply(&vector_quat).multiply(&q.conjugate());
    Vector3::new(rotated.x, rotated.y, rotated.z)
}

#[test]
fn estimates_exact_tilted_square_pose_from_corner_rays() {
    let physical_size = 0.8;
    let half = physical_size * 0.5;
    let center = Vector3::new(0.35, 0.2, 2.6);
    let camera_origin = Vector3::new(-0.25, -0.15, -1.2);

    let yaw = 0.37_f64;
    let pitch = -0.24_f64;
    let right = normalize(Vector3::new(yaw.cos(), pitch.sin(), -yaw.sin()));
    let forward_seed = normalize(Vector3::new(yaw.sin(), 0.35, yaw.cos()));
    let up = normalize(cross(forward_seed, right));
    let forward = normalize(cross(right, up));

    let corners = [
        add(center, add(scale(right, -half), scale(up, half))),
        add(center, add(scale(right, half), scale(up, half))),
        add(center, add(scale(right, half), scale(up, -half))),
        add(center, add(scale(right, -half), scale(up, -half))),
    ];

    let rays = corners.map(|corner| Ray3 {
        origin: camera_origin,
        direction: sub(corner, camera_origin),
    });

    let estimate = estimate_square_pose_from_rays(rays, physical_size).unwrap();

    assert!((estimate.pose.position.x - center.x).abs() < EPS_POSITION);
    assert!((estimate.pose.position.y - center.y).abs() < EPS_POSITION);
    assert!((estimate.pose.position.z - center.z).abs() < EPS_POSITION);
    assert!(
        estimate.confidence > 0.95,
        "confidence was {}",
        estimate.confidence
    );
    assert!(
        estimate.normalized_corner_error < 1e-6,
        "normalized corner error was {}",
        estimate.normalized_corner_error
    );

    let solved = [
        reconstruct(rays[0], estimate.ray_distances[0]),
        reconstruct(rays[1], estimate.ray_distances[1]),
        reconstruct(rays[2], estimate.ray_distances[2]),
        reconstruct(rays[3], estimate.ray_distances[3]),
    ];
    let diagonal = physical_size * 2.0_f64.sqrt();

    assert!((distance(solved[1], solved[0]) - physical_size).abs() < EPS_GEOMETRY);
    assert!((distance(solved[2], solved[1]) - physical_size).abs() < EPS_GEOMETRY);
    assert!((distance(solved[3], solved[2]) - physical_size).abs() < EPS_GEOMETRY);
    assert!((distance(solved[0], solved[3]) - physical_size).abs() < EPS_GEOMETRY);
    assert!((distance(solved[2], solved[0]) - diagonal).abs() < EPS_GEOMETRY);
    assert!((distance(solved[3], solved[1]) - diagonal).abs() < EPS_GEOMETRY);

    let estimated_right = normalize(rotate_vector(
        estimate.pose.rotation,
        Vector3::new(1.0, 0.0, 0.0),
    ));
    let estimated_up = normalize(rotate_vector(
        estimate.pose.rotation,
        Vector3::new(0.0, 1.0, 0.0),
    ));
    let estimated_forward = normalize(rotate_vector(
        estimate.pose.rotation,
        Vector3::new(0.0, 0.0, 1.0),
    ));

    assert!(
        dot(estimated_right, right) > 1.0 - 1e-6,
        "estimated right {:?} expected {:?}",
        estimated_right,
        right
    );
    assert!(
        dot(estimated_up, up) > 1.0 - 1e-6,
        "estimated up {:?} expected {:?}",
        estimated_up,
        up
    );
    assert!(
        dot(estimated_forward, forward) > 1.0 - 1e-6,
        "estimated forward {:?} expected {:?}",
        estimated_forward,
        forward
    );
}

#[test]
fn rejects_invalid_square_pose_inputs() {
    let valid_ray = Ray3 {
        origin: Vector3::new(0.0, 0.0, 0.0),
        direction: Vector3::new(0.0, 0.0, 1.0),
    };
    let zero_direction_ray = Ray3 {
        origin: Vector3::new(0.0, 0.0, 0.0),
        direction: Vector3::new(0.0, 0.0, 0.0),
    };

    assert_eq!(
        estimate_square_pose_from_rays([valid_ray; 4], 0.0),
        Err(PnpError::SolverFailed)
    );
    assert_eq!(
        estimate_square_pose_from_rays([valid_ray, valid_ray, zero_direction_ray, valid_ray,], 0.4,),
        Err(PnpError::SolverFailed)
    );
}

#[test]
fn rejects_degenerate_identical_rays() {
    let ray = Ray3 {
        origin: Vector3::new(0.0, 0.0, 0.0),
        direction: Vector3::new(0.0, 0.0, 1.0),
    };

    assert_eq!(
        estimate_square_pose_from_rays([ray; 4], 0.4),
        Err(PnpError::SolverFailed)
    );
}
