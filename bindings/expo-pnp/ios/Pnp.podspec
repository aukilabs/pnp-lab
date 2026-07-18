Pod::Spec.new do |s|
  s.name           = 'Pnp'
  s.version        = '0.1.0'
  s.summary        = 'PnP pose solving for Expo (PnPLab)'
  s.description    = 'Expo module wrapping the PnPLab Rust pose solver'
  s.author         = 'Auki Labs'
  s.homepage       = 'https://aukilabs.com'
  s.license        = 'MIT'
  s.platforms      = {
    :ios => '15.1'
  }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  # force_load per-SDK so device and simulator each pull their xcframework slice
  # (both slices are named libpnp_ffi.a — required by CocoaPods vendored_frameworks).
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS[sdk=iphoneos*]' =>
      '$(inherited) -ObjC -force_load "$(PODS_TARGET_SRCROOT)/PnpRust.xcframework/ios-arm64/libpnp_ffi.a"',
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' =>
      '$(inherited) -ObjC -force_load "$(PODS_TARGET_SRCROOT)/PnpRust.xcframework/ios-arm64-simulator/libpnp_ffi.a"',
  }

  s.source_files = 'PnpModule.swift', 'PnpNative.swift'
  s.vendored_frameworks = 'PnpRust.xcframework'
  s.libraries = 'c++'
end
