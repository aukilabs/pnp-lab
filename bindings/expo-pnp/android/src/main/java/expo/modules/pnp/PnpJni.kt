package expo.modules.pnp

object PnpJni {
  private val loadResult = runCatching {
    System.loadLibrary("pnp_ffi")
    System.loadLibrary("pnp_jni")
  }

  val isAvailable: Boolean = loadResult.isSuccess
  val loadError: Throwable? = loadResult.exceptionOrNull()

  external fun estimateSquarePoseFromRays(
    rays: Array<DoubleArray>,
    physicalSize: Double,
  ): Map<String, Any>
}
