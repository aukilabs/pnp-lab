import type {
  PnpCamera,
  PnpMatrix3x3,
  PnpRay,
  PnpSquareRays,
  PnpVector2,
} from "./Pnp.types";
import { cameraFromMatrix } from "./Pnp.types";

function resolveCamera(
  camera: PnpCamera | PnpMatrix3x3,
): PnpCamera {
  if ("fx" in camera && "fy" in camera && "cx" in camera && "cy" in camera) {
    return camera;
  }
  return cameraFromMatrix(camera as PnpMatrix3x3);
}

/**
 * Converts a pixel in a pinhole (optionally distorted) camera image into a
 * camera-space ray matching expo-ar's OpenGL-style camera convention.
 *
 * The caller is responsible for keeping `pixel` and `camera` in the same
 * image coordinate system. Valid pairs are:
 *   - AR-buffer pixels with frame intrinsics,
 *   - oriented viewport pixels with a projection-derived camera.
 *
 * Distortion (if present) is inverted on the normalized plane before the ray
 * is formed. The pinhole model after undistortion is:
 *   px = fx * cameraX / depth + cx
 *   py = fy * -cameraY / depth + cy
 *
 * Since the camera looks down negative Z, a pixel ray from the camera origin is:
 *   x = (px - cx) / fx
 *   y = -(py - cy) / fy
 *   z = -1
 *
 * The native Rust solver also exposes this via `Camera::unproject_opengl_ray`.
 */
export function cameraPixelToCameraRay({
  camera,
  cameraMatrix,
  pixel,
}: {
  camera?: PnpCamera;
  /** @deprecated Prefer `camera`. */
  cameraMatrix?: PnpMatrix3x3;
  pixel: PnpVector2;
}): PnpRay {
  const resolved = resolveCamera(
    camera ??
      cameraMatrix ??
      (() => {
        throw new Error("camera (or cameraMatrix) is required");
      })(),
  );

  const { fx, fy, cx, cy } = resolved;
  if (!isUsableFocalLength(fx) || !isUsableFocalLength(fy)) {
    throw new Error("Invalid camera for PnP ray conversion.");
  }

  const undistorted = undistortPixel(resolved, pixel);

  return {
    direction: {
      x: normalizeSignedZero((undistorted.x - cx) / fx),
      y: normalizeSignedZero(-(undistorted.y - cy) / fy),
      z: -1,
    },
    origin: { x: 0, y: 0, z: 0 },
  };
}

export function cameraPixelsToSquareRays({
  camera,
  cameraMatrix,
  pixels,
}: {
  camera?: PnpCamera;
  /** @deprecated Prefer `camera`. */
  cameraMatrix?: PnpMatrix3x3;
  pixels: readonly [PnpVector2, PnpVector2, PnpVector2, PnpVector2];
}): PnpSquareRays {
  return pixels.map((pixel) =>
    cameraPixelToCameraRay({ camera, cameraMatrix, pixel }),
  ) as [PnpRay, PnpRay, PnpRay, PnpRay];
}

export function viewportPixelToCameraRay(args: {
  camera?: PnpCamera;
  cameraMatrix?: PnpMatrix3x3;
  pixel: PnpVector2;
}): PnpRay {
  return cameraPixelToCameraRay(args);
}

export function viewportPixelsToSquareRays(args: {
  camera?: PnpCamera;
  cameraMatrix?: PnpMatrix3x3;
  pixels: readonly [PnpVector2, PnpVector2, PnpVector2, PnpVector2];
}): PnpSquareRays {
  return cameraPixelsToSquareRays(args);
}

function undistortPixel(camera: PnpCamera, pixel: PnpVector2): PnpVector2 {
  const dist = camera.dist ?? [];
  if (dist.length === 0 || dist.every((c) => c === 0)) {
    return pixel;
  }

  const { fx, fy, cx, cy } = camera;
  let x = (pixel.x - cx) / fx;
  let y = (pixel.y - cy) / fy;
  const xd = x;
  const yd = y;

  for (let i = 0; i < 10; i++) {
    const [px, py] = distortNormalized(x, y, dist);
    const errX = px - xd;
    const errY = py - yd;
    x -= errX;
    y -= errY;
    if (errX * errX + errY * errY < 1e-12) {
      break;
    }
  }

  return { x: fx * x + cx, y: fy * y + cy };
}

function distortNormalized(
  x: number,
  y: number,
  dist: readonly number[],
): [number, number] {
  const k1 = dist[0] ?? 0;
  const k2 = dist[1] ?? 0;
  const p1 = dist[2] ?? 0;
  const p2 = dist[3] ?? 0;
  const k3 = dist[4] ?? 0;
  const k4 = dist[5] ?? 0;
  const k5 = dist[6] ?? 0;
  const k6 = dist[7] ?? 0;

  const r2 = x * x + y * y;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  let radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  if (dist.length >= 8) {
    const denom = 1 + k4 * r2 + k5 * r4 + k6 * r6;
    if (Math.abs(denom) > 1e-12) {
      radial /= denom;
    }
  }
  const xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [xd, yd];
}

function isUsableFocalLength(value: number): boolean {
  return Number.isFinite(value) && Math.abs(value) > 1e-12;
}

function normalizeSignedZero(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
