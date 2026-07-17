cmake_minimum_required(VERSION 3.22.1)
project(peyote_pnp_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header + prebuilt Rust cdylib live next to this CMakeLists under the Expo package.
set(PNP_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PNP_JNI_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs")

add_library(pnp_jni SHARED pnp_jni.cpp)
target_link_options(pnp_jni PRIVATE
  "-Wl,-z,max-page-size=16384"
  "-Wl,-z,common-page-size=16384"
)
target_include_directories(pnp_jni PRIVATE "${PNP_CPP_DIR}/include")
target_link_directories(pnp_jni PRIVATE
  "${PNP_JNI_LIBS}/${ANDROID_ABI}"
)

find_library(log-lib log)
target_link_libraries(pnp_jni peyote_pnp_ffi ${log-lib})
