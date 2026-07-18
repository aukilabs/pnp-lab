import ExpoModulesCore
import Foundation

public class PnpModule: Module {
  public func definition() -> ModuleDefinition {
    Name("Pnp")

    AsyncFunction("solvePnpCameraPose") {
      (
        landmarks: [[String: Any]],
        observations: [[String: Any]],
        camera: [String: Any],
        method: String
      ) throws -> [String: Any] in
      try self.solvePnpCameraPose(
        landmarks: landmarks,
        observations: observations,
        camera: camera,
        method: method
      )
    }

    AsyncFunction("estimateSquarePoseFromRays") { (rays: [Any], physicalSize: Double) throws -> [String: Any] in
      try self.estimateSquarePoseFromRays(rays: rays, physicalSize: physicalSize)
    }
  }

  private func solvePnpCameraPose(
    landmarks: [[String: Any]],
    observations: [[String: Any]],
    camera: [String: Any],
    method: String
  ) throws -> [String: Any] {
    let nativeMethod = try self.method(from: method)
    let nativeCamera = try self.makeCamera(camera)
    let nativeLandmarks = try landmarks.map { item in
      try self.makeLandmark(item)
    }
    let nativeObservations = try observations.map { item in
      try self.makeObservation(item)
    }

    do {
      let pose = try PnpNative.solveCameraPose(
        landmarks: nativeLandmarks,
        observations: nativeObservations,
        camera: nativeCamera,
        method: nativeMethod
      )
      return poseToRecord(pose)
    } catch let error as PnpNative.PnpError {
      throw mapPnpError(error)
    }
  }

  private func estimateSquarePoseFromRays(rays: [Any], physicalSize: Double) throws -> [String: Any] {
    let nativeRays = try rays.enumerated().map { index, value in
      try self.makeRay(value, label: "rays[\(index)]")
    }

    do {
      let result = try PnpNative.estimateSquarePoseFromRays(
        rays: nativeRays,
        physicalSize: physicalSize
      )
      return [
        "pose": poseToRecord(result.pose),
        "confidence": result.confidence,
        "normalizedCornerError": result.normalizedCornerError,
        "rayDistances": result.rayDistances,
      ]
    } catch let error as PnpNative.PnpError {
      throw mapPnpError(error)
    }
  }

  private func makeLandmark(_ value: [String: Any]) throws -> PnpNative.Landmark {
    PnpNative.Landmark(
      id: try string(value["id"], label: "landmark.id"),
      position: try makeVector3(value["position"], label: "landmark.position")
    )
  }

  private func makeObservation(_ value: [String: Any]) throws -> PnpNative.LandmarkObservation {
    PnpNative.LandmarkObservation(
      id: try string(value["id"], label: "observation.id"),
      position: try makeVector2(value["position"], label: "observation.position")
    )
  }

  private func makeCamera(_ value: [String: Any]) throws -> PnpNative.Camera {
    let fx = try double(value["fx"], label: "camera.fx")
    let fy = try double(value["fy"], label: "camera.fy")
    let cx = try double(value["cx"], label: "camera.cx")
    let cy = try double(value["cy"], label: "camera.cy")

    var dist: [Double] = []
    if let rawDist = value["dist"] as? [Any] {
      dist = try rawDist.enumerated().map { index, item in
        try double(item, label: "camera.dist[\(index)]")
      }
    }

    do {
      return try PnpNative.Camera(fx: fx, fy: fy, cx: cx, cy: cy, dist: dist)
    } catch let error as PnpNative.PnpError {
      throw mapPnpError(error)
    }
  }

  private func makeRay(_ value: Any, label: String) throws -> PnpNative.Ray {
    guard let record = value as? [String: Any] else {
      throw PnpInvalidInputException("\(label) must be an object")
    }

    return PnpNative.Ray(
      origin: try makeVector3(record["origin"], label: "\(label).origin"),
      direction: try makeVector3(record["direction"], label: "\(label).direction")
    )
  }

  private func makeVector2(_ value: Any?, label: String) throws -> PnpNative.Vector2 {
    guard let record = value as? [String: Any] else {
      throw PnpInvalidInputException("\(label) must be an object")
    }

    return PnpNative.Vector2(
      x: try double(record["x"], label: "\(label).x"),
      y: try double(record["y"], label: "\(label).y")
    )
  }

  private func makeVector3(_ value: Any?, label: String) throws -> PnpNative.Vector3 {
    guard let record = value as? [String: Any] else {
      throw PnpInvalidInputException("\(label) must be an object")
    }

    return PnpNative.Vector3(
      x: try double(record["x"], label: "\(label).x"),
      y: try double(record["y"], label: "\(label).y"),
      z: try double(record["z"], label: "\(label).z")
    )
  }

  private func method(from value: String) throws -> PnpNative.Method {
    guard let method = PnpNative.Method(rawValue: value) else {
      throw PnpInvalidInputException("Unsupported PnP method: \(value)")
    }
    return method
  }

  private func poseToRecord(_ pose: PnpNative.Pose) -> [String: Any] {
    [
      "position": [
        "x": pose.position.x,
        "y": pose.position.y,
        "z": pose.position.z,
      ],
      "rotation": [
        "x": pose.rotation.x,
        "y": pose.rotation.y,
        "z": pose.rotation.z,
        "w": pose.rotation.w,
      ],
    ]
  }

  private func string(_ value: Any?, label: String) throws -> String {
    guard let string = value as? String else {
      throw PnpInvalidInputException("\(label) must be a string")
    }
    return string
  }

  private func double(_ value: Any?, label: String) throws -> Double {
    if value is Bool {
      throw PnpInvalidInputException("\(label) must be a number")
    }

    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
      throw PnpInvalidInputException("\(label) must be a number")
    }

    let doubleValue = number.doubleValue
    guard doubleValue.isFinite else {
      throw PnpInvalidInputException("\(label) must be a finite number")
    }

    return doubleValue
  }

  private func mapPnpError(_ error: PnpNative.PnpError) -> Exception {
    switch error {
    case let .invalidInput(message):
      return PnpInvalidInputException(message)
    case let .nativeFailure(message):
      return PnpNativeFailureException(message)
    }
  }
}

private class PnpInvalidInputException: GenericException<String>, @unchecked Sendable {
  override var reason: String { "Invalid PnP input: \(param)" }
}

private class PnpNativeFailureException: GenericException<String>, @unchecked Sendable {
  override var reason: String { "Rust PnP failed: \(param)" }
}
