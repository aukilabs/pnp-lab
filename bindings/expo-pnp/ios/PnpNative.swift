import Darwin
import Foundation
import PnpRust

/// Native-callable PnP facade (iOS counterpart to Android `PnpNative`).
/// Expo module methods and future iOS calibration both go through here.
public enum PnpNative {
  public static let isAvailable = true

  public static func estimateSquarePoseFromRays(
    rays: [Ray],
    physicalSize: Double
  ) throws -> SquarePoseEstimate {
    guard rays.count == 4 else {
      throw PnpError.invalidInput(
        "estimateSquarePoseFromRays requires exactly 4 rays ordered top-left, top-right, bottom-right, bottom-left"
      )
    }

    guard physicalSize.isFinite, physicalSize > 0 else {
      throw PnpError.invalidInput("physicalSize must be a finite positive number")
    }

    let nativeRays = rays.map { ray in
      pnp_ray_t(
        origin: makeNativeVector3(ray.origin),
        direction: makeNativeVector3(ray.direction)
      )
    }

    let result = nativeRays.withUnsafeBufferPointer { buffer in
      pnp_estimate_square_pose_from_rays(
        buffer.baseAddress,
        UInt(buffer.count),
        physicalSize
      )
    }

    guard result.error == PNP_OK else {
      throw PnpError.nativeFailure(pnpErrorName(result.error))
    }

    return SquarePoseEstimate(
      pose: makePublicPose(result.pose),
      confidence: result.confidence,
      normalizedCornerError: result.normalized_corner_error,
      rayDistances: [
        result.ray_distances.0,
        result.ray_distances.1,
        result.ray_distances.2,
        result.ray_distances.3,
      ]
    )
  }

  public static func solveCameraPose(
    landmarks: [Landmark],
    observations: [LandmarkObservation],
    camera: Camera,
    method: Method
  ) throws -> Pose {
    if landmarks.isEmpty || observations.isEmpty {
      throw PnpError.invalidInput("Landmarks and observations must be non-empty")
    }

    var allocatedStrings: [UnsafeMutablePointer<CChar>] = []
    defer {
      for pointer in allocatedStrings {
        free(pointer)
      }
    }

    let nativeLandmarks = try landmarks.map { item in
      try makeNativeLandmark(item, allocatedStrings: &allocatedStrings)
    }
    let nativeObservations = try observations.map { item in
      try makeNativeObservation(item, allocatedStrings: &allocatedStrings)
    }
    let nativeMethod = makeNativeMethod(method)

    let result = try withNativeCamera(camera) { nativeCamera in
      nativeLandmarks.withUnsafeBufferPointer { landmarkBuffer in
        nativeObservations.withUnsafeBufferPointer { observationBuffer in
          pnp_solve_camera_pose(
            landmarkBuffer.baseAddress,
            UInt(landmarkBuffer.count),
            observationBuffer.baseAddress,
            UInt(observationBuffer.count),
            nativeCamera,
            nativeMethod
          )
        }
      }
    }

    guard result.error == PNP_OK else {
      throw PnpError.nativeFailure(pnpErrorName(result.error))
    }

    return makePublicPose(result.pose)
  }

  // MARK: - Public value types

  public struct Vector2: Equatable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
      self.x = x
      self.y = y
    }
  }

  public struct Vector3: Equatable {
    public let x: Double
    public let y: Double
    public let z: Double

    public init(x: Double, y: Double, z: Double) {
      self.x = x
      self.y = y
      self.z = z
    }
  }

  public struct Ray: Equatable {
    public let origin: Vector3
    public let direction: Vector3

    public init(origin: Vector3, direction: Vector3) {
      self.origin = origin
      self.direction = direction
    }
  }

  public struct Quaternion: Equatable {
    public let x: Double
    public let y: Double
    public let z: Double
    public let w: Double

    public init(x: Double, y: Double, z: Double, w: Double) {
      self.x = x
      self.y = y
      self.z = z
      self.w = w
    }
  }

  public struct Pose: Equatable {
    public let position: Vector3
    public let rotation: Quaternion

    public init(position: Vector3, rotation: Quaternion) {
      self.position = position
      self.rotation = rotation
    }
  }

  public struct Landmark: Equatable {
    public let id: String
    public let position: Vector3

    public init(id: String, position: Vector3) {
      self.id = id
      self.position = position
    }
  }

  public struct LandmarkObservation: Equatable {
    public let id: String
    public let position: Vector2

    public init(id: String, position: Vector2) {
      self.id = id
      self.position = position
    }
  }

  public struct Camera: Equatable {
    public let fx: Double
    public let fy: Double
    public let cx: Double
    public let cy: Double
    /// OpenCV-ordered distortion coeffs (empty = pinhole).
    public let dist: [Double]

    public init(fx: Double, fy: Double, cx: Double, cy: Double, dist: [Double] = []) throws {
      guard fx.isFinite, fy.isFinite, cx.isFinite, cy.isFinite else {
        throw PnpError.invalidInput("camera intrinsics must be finite")
      }
      guard abs(fx) > 1e-12, abs(fy) > 1e-12 else {
        throw PnpError.invalidInput("camera focal lengths must be non-zero")
      }
      guard dist.allSatisfy({ $0.isFinite }) else {
        throw PnpError.invalidInput("distortion coefficients must be finite")
      }
      guard dist.isEmpty || dist.count == 4 || dist.count == 5 || dist.count == 8 else {
        throw PnpError.invalidInput("dist must have 0, 4, 5, or 8 coefficients")
      }
      self.fx = fx
      self.fy = fy
      self.cx = cx
      self.cy = cy
      self.dist = dist
    }
  }

  public enum Method: String {
    case epnp
    case iterative
    case sqpnp
  }

  public struct SquarePoseEstimate: Equatable {
    public let pose: Pose
    public let confidence: Double
    public let normalizedCornerError: Double
    public let rayDistances: [Double]
  }

  public enum PnpError: Error, LocalizedError, Equatable {
    case invalidInput(String)
    case nativeFailure(String)

    public var errorDescription: String? {
      switch self {
      case let .invalidInput(message):
        return "Invalid PnP input: \(message)"
      case let .nativeFailure(message):
        return "Rust PnP failed: \(message)"
      }
    }
  }

  // MARK: - FFI helpers

  private static func makeNativeLandmark(
    _ value: Landmark,
    allocatedStrings: inout [UnsafeMutablePointer<CChar>]
  ) throws -> pnp_landmark_t {
    pnp_landmark_t(
      id: try cString(value.id, label: "landmark.id", allocatedStrings: &allocatedStrings),
      position: makeNativeVector3(value.position)
    )
  }

  private static func makeNativeObservation(
    _ value: LandmarkObservation,
    allocatedStrings: inout [UnsafeMutablePointer<CChar>]
  ) throws -> pnp_landmark_observation_t {
    pnp_landmark_observation_t(
      id: try cString(value.id, label: "observation.id", allocatedStrings: &allocatedStrings),
      position: pnp_vector2_t(x: value.position.x, y: value.position.y)
    )
  }

  private static func cString(
    _ value: String,
    label: String,
    allocatedStrings: inout [UnsafeMutablePointer<CChar>]
  ) throws -> UnsafePointer<CChar>? {
    guard let duplicated = strdup(value) else {
      throw PnpError.invalidInput("Failed to allocate \(label)")
    }
    allocatedStrings.append(duplicated)
    return UnsafePointer(duplicated)
  }

  private static func withNativeCamera<T>(
    _ value: Camera,
    _ body: (UnsafePointer<pnp_camera_t>) throws -> T
  ) rethrows -> T {
    if value.dist.isEmpty {
      var native = pnp_camera_t(
        fx: value.fx,
        fy: value.fy,
        cx: value.cx,
        cy: value.cy,
        dist: nil,
        dist_len: 0
      )
      return try body(&native)
    }

    return try value.dist.withUnsafeBufferPointer { buffer in
      var native = pnp_camera_t(
        fx: value.fx,
        fy: value.fy,
        cx: value.cx,
        cy: value.cy,
        dist: buffer.baseAddress,
        dist_len: UInt(buffer.count)
      )
      return try body(&native)
    }
  }

  private static func makeNativeVector3(_ value: Vector3) -> pnp_vector3_t {
    pnp_vector3_t(x: value.x, y: value.y, z: value.z)
  }

  private static func makeNativeMethod(_ value: Method) -> pnp_method_t {
    switch value {
    case .epnp:
      return PNP_METHOD_EPNP
    case .iterative:
      return PNP_METHOD_ITERATIVE
    case .sqpnp:
      return PNP_METHOD_SQPNP
    }
  }

  private static func makePublicPose(_ pose: pnp_pose_t) -> Pose {
    Pose(
      position: Vector3(x: pose.position.x, y: pose.position.y, z: pose.position.z),
      rotation: Quaternion(
        x: pose.rotation.x,
        y: pose.rotation.y,
        z: pose.rotation.z,
        w: pose.rotation.w
      )
    )
  }

  private static func pnpErrorName(_ error: pnp_error_t) -> String {
    switch error {
    case PNP_OK:
      return "PNP_OK"
    case PNP_ERROR_INSUFFICIENT_POINTS:
      return "PNP_ERROR_INSUFFICIENT_POINTS"
    case PNP_ERROR_SOLVER_FAILED:
      return "PNP_ERROR_SOLVER_FAILED"
    case PNP_ERROR_MISMATCHED_COUNTS:
      return "PNP_ERROR_MISMATCHED_COUNTS"
    case PNP_ERROR_NULL_POINTER:
      return "PNP_ERROR_NULL_POINTER"
    case PNP_ERROR_INVALID_STRING:
      return "PNP_ERROR_INVALID_STRING"
    default:
      return "PNP_ERROR_UNKNOWN_\(Int(error.rawValue))"
    }
  }
}
