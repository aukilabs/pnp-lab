package expo.modules.pnp

import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class PnpModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("Pnp")

    AsyncFunction("solvePnpCameraPose") { landmarks: List<Map<String, Any?>>, observations: List<Map<String, Any?>>, cameraMatrix: Map<String, Any?>, method: String ->
      solvePnpCameraPose(landmarks, observations, cameraMatrix, method)
    }

    AsyncFunction("estimateSquarePoseFromRays") { rays: List<Map<String, Any?>>, physicalSize: Double ->
      estimateSquarePoseFromRays(rays, physicalSize)
    }
  }

  private fun solvePnpCameraPose(
    @Suppress("UNUSED_PARAMETER") landmarks: List<Map<String, Any?>>,
    @Suppress("UNUSED_PARAMETER") observations: List<Map<String, Any?>>,
    @Suppress("UNUSED_PARAMETER") cameraMatrix: Map<String, Any?>,
    @Suppress("UNUSED_PARAMETER") method: String,
  ): Map<String, Any> {
    throw CodedException("solvePnpCameraPose is not implemented yet.")
  }

  private fun estimateSquarePoseFromRays(
    rays: List<Map<String, Any?>>,
    physicalSize: Double,
  ): Map<String, Any> {
    val nativeRays = rays.mapIndexed { index, ray ->
      makeRay(ray, "rays[$index]")
    }

    return PnpNative.estimateSquarePoseFromRays(nativeRays, physicalSize)
  }

  private fun makeRay(value: Map<String, Any?>, label: String): DoubleArray {
    val origin = makeVector3(value["origin"], "$label.origin")
    val direction = makeVector3(value["direction"], "$label.direction")

    return doubleArrayOf(
      origin[0],
      origin[1],
      origin[2],
      direction[0],
      direction[1],
      direction[2],
    )
  }

  private fun makeVector3(value: Any?, label: String): DoubleArray {
    val record = value as? Map<*, *>
      ?: throw PnpInvalidInputException("$label must be an object")

    return doubleArrayOf(
      finiteDouble(record["x"], "$label.x"),
      finiteDouble(record["y"], "$label.y"),
      finiteDouble(record["z"], "$label.z"),
    )
  }

  private fun finiteDouble(value: Any?, label: String): Double {
    val number = value as? Number
      ?: throw PnpInvalidInputException("$label must be a number")

    val doubleValue = number.toDouble()
    if (!doubleValue.isFinite()) {
      throw PnpInvalidInputException("$label must be a finite number")
    }

    return doubleValue
  }
}

private class PnpInvalidInputException(message: String) :
  CodedException("Invalid PnP input: $message")
