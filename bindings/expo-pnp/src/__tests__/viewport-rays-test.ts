import {
  cameraPoseFromKnownObjectPose,
  pnpPoseToMatrix,
} from "../matrix";
import {
  viewportPixelToCameraRay,
  viewportPixelsToSquareRays,
} from "../viewport-rays";

describe("local PnP viewport ray helpers", () => {
  it("converts the viewport principal point to a forward OpenGL camera ray", () => {
    const ray = viewportPixelToCameraRay({
      cameraMatrix: { m: [100, 0, 0, 0, 200, 0, 50, 60, 1] },
      pixel: { x: 50, y: 60 },
    });

    expect(ray).toEqual({
      direction: { x: 0, y: 0, z: -1 },
      origin: { x: 0, y: 0, z: 0 },
    });
  });

  it("keeps viewport Y-down pixels aligned with camera Y-up rays", () => {
    const ray = viewportPixelToCameraRay({
      cameraMatrix: { m: [100, 0, 0, 0, 200, 0, 50, 60, 1] },
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
      cameraMatrix: { m: [100, 0, 0, 0, 100, 0, 100, 50, 1] },
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

  it("inverts rotated object-in-camera poses without transposing the convention", () => {
    const knownObjectPoseMatrix = pnpPoseToMatrix({
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
    });
    const solvedObjectPoseInCameraMatrix = pnpPoseToMatrix({
      position: { x: 1, y: 2, z: -3 },
      rotation: { x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 },
    });

    const cameraPose = cameraPoseFromKnownObjectPose({
      knownObjectPoseMatrix,
      solvedObjectPoseInCameraMatrix,
    });

    expectMatrixClose(
      mat4Multiply(cameraPose, solvedObjectPoseInCameraMatrix),
      knownObjectPoseMatrix,
    );
  });
});

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

function expectMatrixClose(actual: number[], expected: number[]) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(value).toBeCloseTo(expected[index]);
  });
}
