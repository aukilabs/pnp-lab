import PnpModule from "./PnpModule";
import type {
  PnpCamera,
  PnpLandmark,
  PnpLandmarkObservation,
  PnpPose,
  PnpSquarePixels,
  PnpSquareRays,
  SolvePnpMethod,
  SquarePoseEstimate,
} from "./Pnp.types";
import { cameraPixelsToSquareRays } from "./viewport-rays";

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
  camera: PnpCamera,
  method: SolvePnpMethod = "iterative",
): Promise<PnpPose> {
  return PnpModule.solvePnpCameraPose(
    landmarks,
    observations,
    camera,
    method,
  );
}

export function estimateSquarePoseFromRays(
  rays: PnpSquareRays,
  physicalSize: number,
): Promise<SquarePoseEstimate> {
  return PnpModule.estimateSquarePoseFromRays(rays, physicalSize);
}

export function estimateSquarePoseFromPixels(
  pixels: PnpSquarePixels,
  physicalSize: number,
  camera: PnpCamera,
): Promise<SquarePoseEstimate> {
  if (typeof PnpModule.estimateSquarePoseFromPixels === "function") {
    return PnpModule.estimateSquarePoseFromPixels(pixels, physicalSize, camera);
  }
  // Fallback: undistort/unproject in JS then use the ray API.
  const rays = cameraPixelsToSquareRays({ camera, pixels });
  return PnpModule.estimateSquarePoseFromRays(rays, physicalSize);
}
