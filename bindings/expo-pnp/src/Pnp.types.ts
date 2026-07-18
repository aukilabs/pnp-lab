export type ExpoPnpModuleEvents = Record<string, never>;

export type SolvePnpMethod = "epnp" | "iterative" | "sqpnp";

export type PnpVector2 = {
  x: number;
  y: number;
};

export type PnpVector3 = {
  x: number;
  y: number;
  z: number;
};

export type PnpQuaternion = {
  x: number;
  y: number;
  z: number;
  w: number;
};

export type PnpPose = {
  position: PnpVector3;
  rotation: PnpQuaternion;
};

/**
 * Calibrated monocular camera.
 *
 * `dist` is OpenCV-ordered Brown–Conrady coeffs: 0 / 4 / 5 / 8 elements.
 * Omit or pass `[]` for an ideal pinhole.
 */
export type PnpCamera = {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  dist?: readonly number[];
};

/** @deprecated Prefer {@link PnpCamera}. Column-major K only. */
export type PnpMatrix3x3Elements = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/** @deprecated Prefer {@link PnpCamera}. */
export type PnpMatrix3x3 = {
  /**
   * Column-major elements: [fx, 0, 0, 0, fy, 0, cx, cy, 1].
   */
  m: PnpMatrix3x3Elements;
};

export type NativePnpMatrix3x3 = {
  m: number[];
};

export type NativePnpPose = PnpPose;

export type PnpLandmark = {
  id: string;
  position: PnpVector3;
};

export type PnpLandmarkObservation = {
  id: string;
  position: PnpVector2;
};

export type PnpRay = {
  origin: PnpVector3;
  direction: PnpVector3;
};

export type SquarePoseEstimate = {
  pose: PnpPose;
  confidence: number;
  normalizedCornerError: number;
  rayDistances: [number, number, number, number];
};

export type PnpSquareRays = readonly [PnpRay, PnpRay, PnpRay, PnpRay];

export type PnpSquarePixels = readonly [PnpVector2, PnpVector2, PnpVector2, PnpVector2];

export type PnpNativeModule = {
  solvePnpCameraPose(
    landmarks: readonly PnpLandmark[],
    observations: readonly PnpLandmarkObservation[],
    camera: PnpCamera,
    method: SolvePnpMethod,
  ): Promise<PnpPose>;

  estimateSquarePoseFromRays(
    rays: PnpSquareRays,
    physicalSize: number,
  ): Promise<SquarePoseEstimate>;

  estimateSquarePoseFromPixels?(
    pixels: PnpSquarePixels,
    physicalSize: number,
    camera: PnpCamera,
  ): Promise<SquarePoseEstimate>;
};

/** Build a {@link PnpCamera} from a column-major K matrix (optional dist). */
export function cameraFromMatrix(
  matrix: PnpMatrix3x3 | NativePnpMatrix3x3,
  dist: readonly number[] = [],
): PnpCamera {
  const [fx, , , , fy, , cx, cy] = matrix.m;
  return { fx, fy, cx, cy, dist: [...dist] };
}
