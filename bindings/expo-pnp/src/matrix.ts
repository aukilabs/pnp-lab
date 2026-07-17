import type { PnpPose } from "./Pnp.types";

const MIN_QUATERNION_LENGTH = 0.0001;
const IDENTITY_QUATERNION: [number, number, number, number] = [0, 0, 0, 1];

/**
 * Converts a PnP pose into the same column-major TRS matrix convention used by
 * expo-ar/three.js in this app.
 *
 * The returned matrix maps points from the pose's local coordinate system into
 * the parent coordinate system of the pose. For the square-ray API, the native
 * estimate's pose maps QR-local coordinates into the current camera frame.
 */
export function pnpPoseToMatrix(pose: PnpPose): number[] {
  const [x, y, z, w] = normalizeQuaternion([
    pose.rotation.x,
    pose.rotation.y,
    pose.rotation.z,
    pose.rotation.w,
  ]);
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  return [
    1 - (yy + zz), xy + wz, xz - wy, 0,
    xy - wz, 1 - (xx + zz), yz + wx, 0,
    xz + wy, yz - wx, 1 - (xx + yy), 0,
    pose.position.x, pose.position.y, pose.position.z, 1,
  ];
}

/**
 * Recovers the camera pose in the known object's coordinate system.
 *
 * `knownObjectPoseMatrix` maps object-local points into the app/domain frame:
 *   domainFromObject
 *
 * `solvedObjectPoseInCameraMatrix` maps the same object-local points into the
 * current camera frame:
 *   cameraFromObject
 *
 * The camera pose we need for calibration is therefore:
 *   domainFromCamera = domainFromObject * inverse(cameraFromObject)
 */
export function cameraPoseFromKnownObjectPose({
  knownObjectPoseMatrix,
  solvedObjectPoseInCameraMatrix,
}: {
  knownObjectPoseMatrix: number[];
  solvedObjectPoseInCameraMatrix: number[];
}): number[] {
  return mat4Multiply(
    knownObjectPoseMatrix,
    mat4RigidInverse(solvedObjectPoseInCameraMatrix),
  );
}

function mat4Multiply(a: number[], b: number[]) {
  const out = new Array<number>(16);

  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }

  return out;
}

function mat4RigidInverse(matrix: number[]) {
  // Inverse of a rigid column-major transform:
  //   [R | t]^-1 = [R^T | -R^T*t]
  //
  // Keep this layout identical to modules/expo-ar/src/math.ts. A transposed
  // copy here would still pass simple translation tests, but would rotate the
  // recovered calibration origin incorrectly as soon as the scanned QR is not
  // axis-aligned with the camera.
  return [
    matrix[0], matrix[4], matrix[8], 0,
    matrix[1], matrix[5], matrix[9], 0,
    matrix[2], matrix[6], matrix[10], 0,
    -(matrix[0] * matrix[12] + matrix[1] * matrix[13] + matrix[2] * matrix[14]),
    -(matrix[4] * matrix[12] + matrix[5] * matrix[13] + matrix[6] * matrix[14]),
    -(matrix[8] * matrix[12] + matrix[9] * matrix[13] + matrix[10] * matrix[14]),
    1,
  ];
}

function normalizeQuaternion(
  quaternion: [number, number, number, number],
): [number, number, number, number] {
  const length = Math.hypot(...quaternion);
  if (length <= MIN_QUATERNION_LENGTH) return IDENTITY_QUATERNION;

  return quaternion.map((value) => value / length) as [
    number,
    number,
    number,
    number,
  ];
}
