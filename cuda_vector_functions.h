#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "cuda_matrix_struct.h"


inline __device__ float4 make_float4(const float &x,
                                     const float &y,
                                     const float &z)
{
    return make_float4(x, y, z, 0);
}

/**
 * @brief Overload multiplication to handle float ab
 */
inline __device__ float4 operator* (const float &a, const float4 &b)
{
    return make_float4(a * b.x, a * b.y, a * b.z);
}

/**
 * @brief Subtraction operator for proper float4 - float4 behaviour
 */
inline __device__ float4 operator- (const float4 &a, const float4 &b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Addition operator for proper float4 - float4 behaviour
 */
inline __device__ float4 operator+ (const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Dot Multiplication, not only x,y,z are multiplied w is set to 0
 * @param a - float4 vector
 * @param b - float4 vector
 * @return Dot multiplication of a and b
 */
inline __device__ float dot(const float4 &a, const float4 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + 0;
}

/**
 * @brief Normalize vector
 * @param v - Vector to be normalized
 * @return normalized vector of *v*
 */
inline __device__ float4 normalize(const float4 &v)
{
    return v / sqrtf(dot(v, v));
}

/**
 * @brief Invert vector
 * @param v - Vector to be inverted
 * @return Inverted vector of *v*
 */
inline __device__ float4 invert(const float4 &v)
{
    return make_float4(v.x * -1, v.y * -1, v.z * -1);
}

/**
 * @brief Returns length of vector
 * @param v - Vector to be measured
 * @return Length as double of vector *v*
 */
inline __device__ float len(const float4 &v)
{
    return sqrtf(dot(v, v));
}

/**
 * @brief Matrix mutliplication of a vector and a matrix
 * @param a - Vector to be multiplied
 * @param b - Matrix to be multiplied
 * @return Vector *a* multiplied with matrix *b*
 */
inline __device__ float4 mul(const float4 &a, const float3x3 &b)
{
    return make_float4(dot(b.m00_m01_m02, a),
                       dot(b.m10_m11_m12, a),
                       dot(b.m20_m21_m22, a));
}

/**
 * @brief Fast appoximate degrees to radian
 * @param degrees - Degrees
 * @return Radian conversion of *degrees*
 */
inline __device__ float degrees_to_radians(const float &degrees)
{
    return 0.017453292 * degrees; /* PI / 180 * degrees */
}

inline __device__ float4 rotate_yaw(const float4 &a, const float &yaw)
{
    float rad = degrees_to_radians(yaw);
    float cos_y = cosf(rad);
    float sin_y = sinf(rad);

    return make_float4(a.x * cos_y + a.z * sin_y,
                       a.y,
                       -a.x * sin_y + a.z * cos_y);
}

inline __device__ float4 rotate_pitch(const float4 &a, const float pitch)
{
    float rad = degrees_to_radians(pitch);
    float cos_x = cosf(rad);
    float sin_x = sinf(rad);

    return make_float4(a.x,
                       a.y * cos_x - a.z * sin_x,
                       a.y * sin_x + a.z * cos_x);
}

inline __device__ float4 rotate_roll(const float4 &a, const float &roll)
{
    float rad = degrees_to_radians(roll);
    float cos_z = cosf(rad);
    float sin_z = sinf(rad);

    return make_float4(a.x * cos_z - a.y * sin_z,
                       a.x * sin_z + a.y * cos_z,
                       a.z);
}

inline __device__ float4 apply_camera_rotation(const float4 &cam,
                                               const float &yaw,
                                               const float &pitch,
                                               const float &roll)
{
    float4 a = cam;

    float4 rad = {
        degrees_to_radians(yaw),
        degrees_to_radians(pitch),
        degrees_to_radians(roll),
    };

    float3x3 x_rotation = {
        { 1, 0,               0               },
        { 0, cosf(rad.x), sinf(rad.x) },
        { 0, sinf(rad.x), cosf(rad.x) },
    };

    float3x3 y_rotation = {
        { cosf(rad.y), 0, -sinf(rad.y) },
        { 0,               1, 0                },
        { sinf(rad.y), 0, cosf(rad.y)  },
    };

    float3x3 z_rotation = {
        { cosf(rad.z), -sinf(rad.z), 0 },
        { sinf(rad.z), cosf(rad.z),  0 },
        { 0,               0,                1 },
    };

    a = mul(a, x_rotation);
    a = mul(a, y_rotation);
    a = mul(a, z_rotation);

    return a;
}

inline __device__ float4 calc_forward_euler_angle(const float &yaw,
                                                  const float &pitch)
{
    float r_pitch = degrees_to_radians(pitch);
    float r_yaw = degrees_to_radians(yaw);
    float cos_pitch = cosf(r_pitch);
    float sin_pitch = sinf(r_pitch);
    float cos_yaw = cosf(r_yaw);
    float sin_yaw = sinf(r_yaw);

    return make_float4(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
}