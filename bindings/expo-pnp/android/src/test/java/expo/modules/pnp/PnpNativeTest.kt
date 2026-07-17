package expo.modules.pnp

import org.junit.Assert.assertEquals
import org.junit.Test

class PnpNativeTest {
  @Test
  fun exposesJniAvailabilityForNativeCallers() {
    assertEquals(PnpJni.isAvailable, PnpNative.isAvailable)
    assertEquals(PnpJni.loadError, PnpNative.loadError)
  }
}
