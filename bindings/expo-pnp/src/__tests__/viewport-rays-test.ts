import {
  cameraPoseFromKnownObjectPose,
  pnpPoseToMatrix,
} from "../matrix";
import {
  cameraPixelToCameraRay,
  viewportPixelToCameraRay,
  viewportPixelsToSquareRays,
} from "../viewport-rays";

describe("local PnP viewport ray helpers", () => {
  it("converts the principal point to a forward OpenGL camera ray", () => {
    const ray = cameraPixelToCameraRay({
      camera: { fx: 100, fy: 200, cx: 50, cy: 60 },
      pixel: { x: 50, y: 60 },
    });

    expect(ray).toEqual({
      direction: { x: 0, y: 0, z: -1 },
      origin: { x: 0, y: 0, z: 0 },
    });
  });

  it("still accepts legacy column-major cameraMatrix", () => {
    const ray = viewportPixelToCameraRay({
      cameraMatrix: { m: [100, 0, 0, 0, 200, 0, 50, 60, 1] },
      pixel: { x: 50, y: 60 },
    });

    expect(ray.direction).toEqual({ x: 0, y: 0, z: -1 });
  });

  it("keeps viewport Y-down pixels aligned with camera Y-up rays", () => {
    const ray = viewportPixelToCameraRay({
      camera: { fx: 100, fy: 200, cx: 50, cy: 60 },
      pixel: { x: 150, y: -40 },
    });

    expect(ray.direction).toEqual({
      x: 1,
      y: 0.5,
      z: -1,
    });
  });

  it("builds TL/TR/BR/BL square rays without rotating the scanner order", () => {
    const rays = viewportPixelsToSquareRays({
      camera: { fx: 100, fy: 100, cx: 100, cy: 50 },
      pixels: [
        { x: 0, y: 0 },
        { x: 200, y: 0 },
        { x: 200, y: 100 },
        { x: 0, y: 100 },
      ],
    });

    expect(rays.map((ray) => ray.direction)).toEqual([
      { x: -1, y: 0.5, z: -1 },
      { x: 1, y: 0.5, z: -1 },
      { x: 1, y: -0.5, z: -1 },
      { x: -1, y: -0.5, z: -1 },
    ]);
  });

  it("undistorts before forming rays when dist coeffs are provided", () => {
    const camera = {
      fx: 100,
      fy: 100,
      cx: 0,
      cy: 0,
      dist: [0.1, 0, 0, 0],
    };
    // Distorted pixel of an ideal principal-ish offset; after undistort the
    // ray x component should move closer to the ideal pinhole value.
    const distorted = cameraPixelToCameraRay({
      camera,
      pixel: { x: 50, y: 0 },
    });
    const pinhole = cameraPixelToCameraRay({
      camera: { fx: 100, fy: 100, cx: 0, cy: 0 },
      pixel: { x: 50, y: 0 },
    });
    // With positive k1, undistort pulls the normalized radius in slightly.
    expect(Math.abs(distorted.direction.x)).toBeLessThan(Math.abs(pinhole.direction.x));
  });

  it("recovers the camera pose from a known object pose and object-in-camera pose", () => {
    const knownObjectPoseMatrix = pnpPoseToMatrix({
      position: { x: 10, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
    });
    const solvedObjectPoseInCameraMatrix = pnpPoseToMatrix({
      position: { x: 0, y: 0, z: -2 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
    });

    const cameraPose = cameraPoseFromKnownObjectPose({
      knownObjectPoseMatrix,
      solvedObjectPoseInCameraMatrix,
    });

    expect(cameraPose[12]).toBeCloseTo(10);
    expect(cameraPose[13]).toBeCloseTo(0);
    expect(cameraPose[14]).toBeCloseTo(2);
  });
});
