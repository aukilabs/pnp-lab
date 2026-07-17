package expo.modules.pnp

import expo.modules.kotlin.exception.CodedException

object PnpNative {
  val isAvailable: Boolean
    get() = PnpJni.isAvailable

  val loadError: Throwable?
    get() = PnpJni.loadError

  fun estimateSquarePoseFromRays(
    rays: List<DoubleArray>,
    physicalSize: Double
  ): Map<String, Any> {
    if (!PnpJni.isAvailable) {
      throw CodedException(
        "Local PnP JNI libraries are unavailable: ${PnpJni.loadError?.message ?: "unknown load failure"}"
      )
    }

    if (rays.size != SQUARE_RAY_COUNT) {
      throw CodedException(
        "estimateSquarePoseFromRays requires exactly 4 rays ordered top-left, top-right, bottom-right, bottom-left"
      )
    }

    if (!physicalSize.isFinite() || physicalSize <= 0.0) {
      throw CodedException("physicalSize must be a finite positive number")
    }

    return try {
      PnpJni.estimateSquarePoseFromRays(rays.toTypedArray(), physicalSize)
    } catch (error: RuntimeException) {
      throw CodedException(
        "Rust PnP square pose estimation failed: ${error.message ?: error.javaClass.simpleName}",
        error
      )
    }
  }

  private const val SQUARE_RAY_COUNT = 4
}
