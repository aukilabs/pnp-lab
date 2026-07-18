#include <jni.h>

#include <array>
#include <cmath>

extern "C" {
#include "pnp.h"
}

namespace {

constexpr int kRayCount = 4;
constexpr int kRayComponentCount = 6;

void throwException(JNIEnv* env, const char* className, const char* message) {
  jclass exceptionClass = env->FindClass(className);
  if (exceptionClass != nullptr) {
    env->ThrowNew(exceptionClass, message);
  }
}

const char* pnpErrorName(pnp_error_t error) {
  switch (error) {
    case PNP_OK:
      return "PNP_OK";
    case PNP_ERROR_INSUFFICIENT_POINTS:
      return "PNP_ERROR_INSUFFICIENT_POINTS";
    case PNP_ERROR_SOLVER_FAILED:
      return "PNP_ERROR_SOLVER_FAILED";
    case PNP_ERROR_MISMATCHED_COUNTS:
      return "PNP_ERROR_MISMATCHED_COUNTS";
    case PNP_ERROR_NULL_POINTER:
      return "PNP_ERROR_NULL_POINTER";
    case PNP_ERROR_INVALID_STRING:
      return "PNP_ERROR_INVALID_STRING";
    default:
      return "PNP_ERROR_UNKNOWN";
  }
}

jobject newHashMap(JNIEnv* env) {
  jclass mapClass = env->FindClass("java/util/HashMap");
  if (mapClass == nullptr) return nullptr;
  jmethodID constructor = env->GetMethodID(mapClass, "<init>", "()V");
  if (constructor == nullptr) return nullptr;
  return env->NewObject(mapClass, constructor);
}

jobject newDouble(JNIEnv* env, double value) {
  jclass doubleClass = env->FindClass("java/lang/Double");
  if (doubleClass == nullptr) return nullptr;
  jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
  if (valueOf == nullptr) return nullptr;
  return env->CallStaticObjectMethod(doubleClass, valueOf, value);
}

void putObject(JNIEnv* env, jobject map, const char* key, jobject value) {
  if (env->ExceptionCheck() || map == nullptr || value == nullptr) return;

  jclass mapClass = env->GetObjectClass(map);
  if (mapClass == nullptr) return;

  jmethodID put = env->GetMethodID(
    mapClass,
    "put",
    "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;"
  );
  if (put == nullptr) return;

  jstring javaKey = env->NewStringUTF(key);
  if (javaKey == nullptr) return;

  jobject previous = env->CallObjectMethod(map, put, javaKey, value);
  if (previous != nullptr) env->DeleteLocalRef(previous);
  env->DeleteLocalRef(javaKey);
}

void putDouble(JNIEnv* env, jobject map, const char* key, double value) {
  jobject boxed = newDouble(env, value);
  putObject(env, map, key, boxed);
  if (boxed != nullptr) env->DeleteLocalRef(boxed);
}

jobject vectorToMap(JNIEnv* env, const pnp_vector3_t& vector) {
  jobject map = newHashMap(env);
  putDouble(env, map, "x", vector.x);
  putDouble(env, map, "y", vector.y);
  putDouble(env, map, "z", vector.z);
  return map;
}

jobject quaternionToMap(JNIEnv* env, const pnp_quaternion_t& quaternion) {
  jobject map = newHashMap(env);
  putDouble(env, map, "x", quaternion.x);
  putDouble(env, map, "y", quaternion.y);
  putDouble(env, map, "z", quaternion.z);
  putDouble(env, map, "w", quaternion.w);
  return map;
}

jobject poseToMap(JNIEnv* env, const pnp_pose_t& pose) {
  jobject map = newHashMap(env);
  jobject position = vectorToMap(env, pose.position);
  putObject(env, map, "position", position);
  if (position != nullptr) env->DeleteLocalRef(position);

  jobject rotation = quaternionToMap(env, pose.rotation);
  putObject(env, map, "rotation", rotation);
  if (rotation != nullptr) env->DeleteLocalRef(rotation);

  return map;
}

jobject rayDistancesToList(JNIEnv* env, const double distances[kRayCount]) {
  jclass listClass = env->FindClass("java/util/ArrayList");
  if (listClass == nullptr) return nullptr;

  jmethodID constructor = env->GetMethodID(listClass, "<init>", "(I)V");
  jmethodID add = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
  if (constructor == nullptr || add == nullptr) return nullptr;

  jobject list = env->NewObject(listClass, constructor, kRayCount);
  for (int i = 0; i < kRayCount; i++) {
    jobject boxed = newDouble(env, distances[i]);
    env->CallBooleanMethod(list, add, boxed);
    if (boxed != nullptr) env->DeleteLocalRef(boxed);
  }

  return list;
}

bool readRay(JNIEnv* env, jobject row, pnp_ray_t* outRay) {
  if (row == nullptr) {
    throwException(env, "java/lang/IllegalArgumentException", "Each ray must be a DoubleArray");
    return false;
  }

  auto values = static_cast<jdoubleArray>(row);
  if (env->GetArrayLength(values) != kRayComponentCount) {
    throwException(env, "java/lang/IllegalArgumentException", "Each ray must contain 6 doubles");
    return false;
  }

  jboolean isCopy = JNI_FALSE;
  jdouble* components = env->GetDoubleArrayElements(values, &isCopy);
  if (components == nullptr) return false;

  for (int i = 0; i < kRayComponentCount; i++) {
    if (!std::isfinite(components[i])) {
      env->ReleaseDoubleArrayElements(values, components, JNI_ABORT);
      throwException(env, "java/lang/IllegalArgumentException", "Ray components must be finite");
      return false;
    }
  }

  *outRay = pnp_ray_t{
    pnp_vector3_t{components[0], components[1], components[2]},
    pnp_vector3_t{components[3], components[4], components[5]},
  };

  env->ReleaseDoubleArrayElements(values, components, JNI_ABORT);
  return true;
}

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_expo_modules_pnp_PnpJni_estimateSquarePoseFromRays(
  JNIEnv* env,
  jobject /*thiz*/,
  jobjectArray rays,
  jdouble physicalSize
) {
  if (rays == nullptr || env->GetArrayLength(rays) != kRayCount) {
    throwException(
      env,
      "java/lang/IllegalArgumentException",
      "estimateSquarePoseFromRays requires exactly 4 rays"
    );
    return nullptr;
  }

  if (!std::isfinite(physicalSize) || physicalSize <= 0.0) {
    throwException(
      env,
      "java/lang/IllegalArgumentException",
      "physicalSize must be a finite positive number"
    );
    return nullptr;
  }

  std::array<pnp_ray_t, kRayCount> nativeRays{};
  for (int i = 0; i < kRayCount; i++) {
    jobject row = env->GetObjectArrayElement(rays, i);
    const bool ok = readRay(env, row, &nativeRays[i]);
    if (row != nullptr) env->DeleteLocalRef(row);
    if (!ok) return nullptr;
  }

  pnp_square_pose_estimate_t result = pnp_estimate_square_pose_from_rays(
    nativeRays.data(),
    nativeRays.size(),
    physicalSize
  );

  if (result.error != PNP_OK) {
    throwException(env, "java/lang/IllegalStateException", pnpErrorName(result.error));
    return nullptr;
  }

  jobject map = newHashMap(env);
  jobject pose = poseToMap(env, result.pose);
  putObject(env, map, "pose", pose);
  if (pose != nullptr) env->DeleteLocalRef(pose);

  putDouble(env, map, "confidence", result.confidence);
  putDouble(env, map, "normalizedCornerError", result.normalized_corner_error);

  jobject rayDistances = rayDistancesToList(env, result.ray_distances);
  putObject(env, map, "rayDistances", rayDistances);
  if (rayDistances != nullptr) env->DeleteLocalRef(rayDistances);

  return map;
}
