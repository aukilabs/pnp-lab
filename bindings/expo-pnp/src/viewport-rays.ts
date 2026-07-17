import type {
  PnpMatrix3x3,
  PnpRay,
  PnpSquareRays,
  PnpVector2,
} from "./Pnp.types";

/**
 * Converts a pixel in a pinhole camera image into a camera-space ray matching
 * expo-ar's OpenGL-style camera convention.
 *
 * The caller is responsible for keeping `pixel` and `cameraMatrix` in the same
 * image coordinate system. Valid pairs are:
 *   - AR-buffer pixels with frame.intrinsics,
 *   - oriented viewport pixels with a projection-derived viewport matrix.
 *
 * The pinhole model is:
 *   px = fx * cameraX / depth + cx
 *   py = fy * -cameraY / depth + cy
 *
 * Since the camera looks down negative Z, a pixel ray from the camera origin is:
 *   x = (px - cx) / fx
 *   y = -(py - cy) / fy
 *   z = -1
 *
 * The native Rust solver normalizes directions internally, so preserving this
 * unnormalized ray is preferable: it keeps the formula legible and avoids
 * introducing another floating-point normalization step on the JS side.
 */
export function cameraPixelToCameraRay({
  cameraMatrix,
  pixel,
}: {
  cameraMatrix: PnpMatrix3x3;
  pixel: PnpVector2;
}): PnpRay {
  const [fx, , , , fy, , cx, cy] = cameraMatrix.m;
  if (!isUsableFocalLength(fx) || !isUsableFocalLength(fy)) {
    throw new Error("Invalid viewport camera matrix for PnP ray conversion.");
  }

  return {
    direction: {
      x: normalizeSignedZero((pixel.x - cx) / fx),
      y: normalizeSignedZero(-(pixel.y - cy) / fy),
      z: -1,
    },
    origin: { x: 0, y: 0, z: 0 },
  };
}

export function cameraPixelsToSquareRays({
  cameraMatrix,
  pixels,
}: {
  cameraMatrix: PnpMatrix3x3;
  pixels: readonly [PnpVector2, PnpVector2, PnpVector2, PnpVector2];
}): PnpSquareRays {
  return pixels.map((pixel) =>
    cameraPixelToCameraRay({ cameraMatrix, pixel }),
  ) as [PnpRay, PnpRay, PnpRay, PnpRay];
}

export function viewportPixelToCameraRay(args: {
  cameraMatrix: PnpMatrix3x3;
  pixel: PnpVector2;
}): PnpRay {
  // Backwards-compatible name for callers that explicitly work in oriented
  // viewport pixels with a projection-derived camera matrix.
  return cameraPixelToCameraRay(args);
}

export function viewportPixelsToSquareRays(args: {
  cameraMatrix: PnpMatrix3x3;
  pixels: readonly [PnpVector2, PnpVector2, PnpVector2, PnpVector2];
}): PnpSquareRays {
  // Backwards-compatible name for callers that explicitly work in oriented
  // viewport pixels with a projection-derived camera matrix.
  return cameraPixelsToSquareRays(args);
}

function isUsableFocalLength(value: number) {
  return Number.isFinite(value) && Math.abs(value) > 0.000001;
}

function normalizeSignedZero(value: number) {
  return Object.is(value, -0) ? 0 : value;
}
