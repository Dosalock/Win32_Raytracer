#pragma once

#define __CUDACC__

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Camera
{
    float4 position;
    float  yaw;
    float  pitch;
    float  roll;
};

struct Sphere
{
    float4   center;
    float    radius;
    COLORREF color;
    int      specularity;
    float    reflective;
    float    sRadius;

    __device__ Sphere ( float4   center      = { },
                                 float    radius      = 0,
                                 COLORREF color       = RGB( 0, 0, 0 ),
                                 int      specularity = 0,
                                 float    reflective  = 0 ) :
        center( center ),
        radius( radius ),
        color( color ),
        specularity( specularity ),
        reflective( reflective ),
        sRadius( radius * radius )
    {
    }
};

struct Light
{
    enum LightType
    {
        directional,
        point,
        ambient
    } type;

    float  intensity;
    float4 pos;
};

struct intersection
{
    Sphere *sphere;
    float   point;

    __shared__ intersection ( Sphere *sphere, float point )
    {
        this->sphere = sphere;
        this->point  = point;
    }
};

struct float3x3
{
    float4 m00_m01_m02;
    float4 m10_m11_m12;
    float4 m20_m21_m22;

    __device__ float3x3 ( float4 m00_m01_m02,
                                   float4 m10_m11_m12,
                                   float4 m20_m21_m22 )
    {
        this->m00_m01_m02 = m00_m01_m02;
        this->m10_m11_m12 = m10_m11_m12;
        this->m20_m21_m22 = m20_m21_m22;
    }
};

/**
 * @brief Overloaded make_float4 to be a three dimensional vector
 * @param x - x-axis coordinate
 * @param y - y-axis coordinate
 * @param z - z-axis coordinate
 * @return float4{x,y,z,0}
 */
inline __device__ float4 make_float4 ( const float &x,
                                       const float &y,
                                       const float &z )
{
    return make_float4( x, y, z, 0 );
}

/**
 * @brief Overload multiplication to handle float ab
 */
inline __device__ float4 operator* ( const float &a, const float4 &b )
{
    return make_float4( a * b.x, a * b.y, a * b.z );
}

/**
 * @brief Subtraction operator for proper float4 - float4 behaviour
 */
inline __device__ float4 operator- ( const float4 &a, const float4 &b )
{
    return make_float4( a.x - b.x, a.y - b.y, a.z - b.z );
}

/**
 * @brief Addition operator for proper float4 - float4 behaviour
 */
inline __device__ float4 operator+ ( const float4 &a, const float4 &b )
{
    return make_float4( a.x + b.x, a.y + b.y, a.z + b.z );
}

/**
 * @brief Dot Multiplication, not only x,y,z are multiplied w is set to 0
 * @param a - float4 vector
 * @param b - float4 vector
 * @return Dot multiplication of a and b
 */
inline __device__ float dot ( const float4 &a, const float4 &b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z + 0;
}

/**
 * @brief Normalize vector
 * @param v - Vector to be normalized
 * @return normalized vector of *v*
 */
inline __device__ float4 normalize ( const float4 &v )
{
    return __fsqrt_rd( dot( v, v ) ) * v;
}

/**
 * @brief Invert vector
 * @param v - Vector to be inverted
 * @return Inverted vector of *v*
 */
inline __device__ float4 invert ( const float4 &v )
{
    return make_float4( v.x * -1, v.y * -1, v.z * -1 );
}

/**
 * @brief Returns length of vector
 * @param v - Vector to be measured
 * @return Length as double of vector *v*
 */
inline __device__ float len ( const float4 &v )
{
    return __fsqrt_rd( dot( v, v ) );
}

/**
 * @brief Matrix mutliplication of a vector and a matrix
 * @param a - Vector to be multiplied
 * @param b - Matrix to be multiplied
 * @return Vector *a* multiplied with matrix *b*
 */
inline __device__ float4 mul ( const float4 &a, const float3x3 &b )
{
    return make_float4( dot( b.m00_m01_m02, a ),
                        dot( b.m10_m11_m12, a ),
                        dot( b.m20_m21_m22, a ) );
}

/**
 * @brief Fast appoximate degrees to radian
 * @param degrees - Degrees
 * @return Radian conversion of *degrees*
 */
inline __device__ float degrees_to_radians ( const float &degrees )
{
    return 0.017453292 * degrees; /* PI / 180 * degrees */
}

inline __device__ float4 rotate_yaw ( const float4 &a, const float &yaw )
{
    float rad   = degrees_to_radians( yaw );
    float cos_y = __cosf( rad );
    float sin_y = __sinf( rad );

    return make_float4( a.x * cos_y + a.z * sin_y,
                        a.y,
                        -a.x * sin_y + a.z * cos_y );
}

inline __device__ float4 rotate_pitch ( const float4 &a, const float pitch )
{
    float rad   = degrees_to_radians( pitch );
    float cos_x = __cosf( rad );
    float sin_x = __sinf( rad );

    return make_float4( a.x,
                        a.y * cos_x - a.z * sin_x,
                        a.y * sin_x + a.z * cos_x );
}

inline __device__ float4 rotate_roll ( const float4 &a, const float &roll )
{
    float rad   = degrees_to_radians( roll );
    float cos_z = __cosf( rad );
    float sin_z = __sinf( rad );

    return make_float4( a.x * cos_z - a.y * sin_z,
                        a.x * sin_z + a.y * cos_z,
                        a.z );
}

inline __device__ float4 apply_camera_rotation ( const float4 &cam,
                                                 const float  &yaw,
                                                 const float  &pitch,
                                                 const float  &roll )
{
    float4 a = cam;


    float4 rad = {
        degrees_to_radians( yaw ),
        degrees_to_radians( pitch ),
        degrees_to_radians( roll ),
    };


    float3x3 x_rotation = {
        { 1, 0,               0               },
        { 0, __cosf( rad.x ), __sinf( rad.x ) },
        { 0, __sinf( rad.x ), __cosf( rad.x ) },
    };

    float3x3 y_rotation = {
        { __cosf( rad.y ), 0, -__sinf( rad.y ) },
        { 0,               1, 0                },
        { __sinf( rad.y ), 0, __cosf( rad.y )  },
    };

    float3x3 z_rotation = {
        { __cosf( rad.z ), -__sinf( rad.z ), 0 },
        { __sinf( rad.z ), __cosf( rad.z ),  0 },
        { 0,               0,                1 },
    };

    a = mul( a, x_rotation );
    a = mul( a, y_rotation );
    a = mul( a, z_rotation );

    return a;
}

inline __device__ float4 calc_forward_euler_angle ( const float &yaw,
                                                    const float &pitch )
{
    float r_pitch   = degrees_to_radians( pitch );
    float r_yaw     = degrees_to_radians( yaw );
    float cos_pitch = __cosf( r_pitch );
    float sin_pitch = __sinf( r_pitch );
    float cos_yaw   = __cosf( r_yaw );
    float sin_yaw   = __sinf( r_yaw );

    return make_float4( cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw );
}
