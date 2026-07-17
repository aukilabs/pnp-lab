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

export type PnpNativeModule = {
  solvePnpCameraPose(
    landmarks: readonly PnpLandmark[],
    observations: readonly PnpLandmarkObservation[],
    cameraMatrix: PnpMatrix3x3,
    method: SolvePnpMethod,
  ): Promise<PnpPose>;

  estimateSquarePoseFromRays(
    rays: PnpSquareRays,
    physicalSize: number,
  ): Promise<SquarePoseEstimate>;
};
