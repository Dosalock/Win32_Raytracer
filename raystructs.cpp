#pragma once

#define __CUDACC__


#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct float3x3 {
  float4 m00_m01_m02, m10_m11_m12, m20_m21_m22;
};

/**
 * @brief Overloaded make_float4 to be a three dimensional vector
 * @param x - x-axis coordinate
 * @param y - y-axis coordinate
 * @param z - z-axis coordinate
 * @return float4{x,y,z,0}
 */
inline __device__ float4 make_float4(const float &x, const float &y,
                                     const float &z) {
  return make_float4(x, y, z, 0);
}

inline __device__ float4 operator*(const float &a, const float4 &b) {
  return make_float4(a * b.x, a * b.y, a * b.z);
}

inline __device__ float4 operator-(const float4 &a, const float4 &b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float4 operator+(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float dot(const float4 &a, const float4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + 0;
}

inline __device__ float4 normalize(const float4 &v) {
  return __fsqrt_rd(dot(v, v)) * v;
}

inline __device__ float4 invert(const float4 &v) {
  return make_float4(v.x * -1, v.y * -1, v.z * -1);
}

inline __device__ double len(const float4 &v) { return __fsqrt_rd(dot(v, v)); }

inline __device__ float4 mul(const float4 &a, const float3x3 &b) {
  return make_float4(dot(b.m00_m01_m02, a), dot(b.m10_m11_m12, a),
                     dot(b.m20_m21_m22, a));
}

inline __device__ float degrees_to_radians(const float &degrees) {
  return 0.017453292 * degrees; /* PI / 180 * degrees */
}

inline __device__ float4 rotate_yaw(const float4 &a, const float &yaw) {
  float rad = degrees_to_radians(yaw);
  float cos_y = __cosf(rad);
  float sin_y = __sinf(rad);

  return make_float4(a.x * cos_y + a.z * sin_y, a.y,
                     -a.x * sin_y + a.z * cos_y);
}

inline __device__ float4 rotate_pitch(const float4 &a, const float pitch) {
  float rad = degrees_to_radians(pitch);
  float cos_x = __cosf(rad);
  float sin_x = __sinf(rad);

  return make_float4(a.x, a.y * cos_x - a.z * sin_x, a.y * sin_x + a.z * cos_x);
}

inline __device__ float4 rotate_roll(const float4 &a, const float &roll) {
  float rad = degrees_to_radians(roll);
  float cos_z = __cosf(rad);
  float sin_z = __sinf(rad);

  return make_float4(a.x * cos_z - a.y * sin_z, a.x * sin_z + a.y * cos_z, a.z);
}

inline __device__ float4 apply_camera_rotation(const float4 &cam,
                                               const float &yaw,
                                               const float &pitch,
                                               const float &roll) {
  float4 a = {};
  a = rotate_yaw(cam, yaw);
  a = rotate_pitch(cam, pitch);
  a = rotate_roll(cam, roll);

  return a;
}

inline __device__ float4 calc_forward_euler_angle(const float &yaw,
                                                  const float &pitch) {
  float r_pitch = degrees_to_radians(pitch);
  float r_yaw = degrees_to_radians(yaw);
  float cos_pitch = __cosf(r_pitch);
  float sin_pitch = __sinf(r_pitch);
  float cos_yaw = __cosf(r_yaw);
  float sin_yaw = __sinf(r_yaw);

  return make_float4(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
}
