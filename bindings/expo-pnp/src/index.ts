import PnpModule from "./PnpModule";
import type {
  PnpLandmark,
  PnpLandmarkObservation,
  PnpMatrix3x3,
  PnpPose,
  PnpSquareRays,
  SolvePnpMethod,
  SquarePoseEstimate,
} from "./Pnp.types";

export * from "./Pnp.types";
export {
  cameraPoseFromKnownObjectPose,
  pnpPoseToMatrix,
} from "./matrix";
export {
  cameraPixelToCameraRay,
  cameraPixelsToSquareRays,
  viewportPixelToCameraRay,
  viewportPixelsToSquareRays,
} from "./viewport-rays";

export function solvePnpCameraPose(
  landmarks: readonly PnpLandmark[],
  observations: readonly PnpLandmarkObservation[],
  cameraMatrix: PnpMatrix3x3,
  method: SolvePnpMethod = "iterative",
): Promise<PnpPose> {
  return PnpModule.solvePnpCameraPose(
    landmarks,
    observations,
    cameraMatrix,
    method,
  );
}

export function estimateSquarePoseFromRays(
  rays: PnpSquareRays,
  physicalSize: number,
): Promise<SquarePoseEstimate> {
  return PnpModule.estimateSquarePoseFromRays(rays, physicalSize);
}
