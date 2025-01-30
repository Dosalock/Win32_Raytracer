#pragma once
/*********************************************************************
 * @file	raystructs.h
 * @brief	Structs for raytracing
 *			3D vectors, spheres, lights....
 *
 * @author	Johan Karlsson - github.com/dosalock
 * @date	November 2024
 *********************************************************************/


/*-----------------------------Includes------------------------------*/
#include "Windows.h"
#include <cmath>
#define PI 3.14159265358979323846

/*-----------------------------Structs-------------------------------*/

/**
 * @struct Vect3D raystruct.h
 * @brief Three dimensional vector
 *
 */
struct Vect3D
{
    float x; // @brief Represents the vector's position along the X-axis
    float y; // @brief Represents the vector's position along the Y-axis
    float z; // @brief Represents the vector's position along the Z-axis

    Vect3D ( float x = 0, float y = 0, float z = 0 ) :
        x( x ), y( y ), z( z )
    {
    }

    Vect3D operator- ( const Vect3D &other ) const
    {
        return Vect3D( x - other.x, y - other.y, z - other.z );
    }

    Vect3D operator+ ( const Vect3D &other ) const
    {
        return Vect3D( x + other.x, y + other.y, z + other.z );
    }

    Vect3D operator* ( const Vect3D &other ) const
    {
        return Vect3D( x * other.x, y * other.y, z * other.z );
    }

    Vect3D operator/ ( const Vect3D &other ) const
    {
        return Vect3D( x / other.x, y / other.y, z / other.z );
    }

    Vect3D operator/ ( const float &other ) const
    {
        return Vect3D( x / other, y / other, z / other );
    }

    Vect3D operator* ( const float &other ) const
    {
        return Vect3D( x * other, y * other, z * other );
    }

    Vect3D cross ( const Vect3D &other ) const
    {
        return Vect3D( y * other.z - z * other.y,
                       z * other.x - x * other.z,
                       x * other.y - y * other.z );
    }

    Vect3D invert ( ) const
    {
        return Vect3D( x * -1, y * -1, z * -1 );
    }

    float len ( ) const
    {
        return sqrt( x * x + y * y + z * z );
    }

    Vect3D norm ( ) const
    {
        return Vect3D( x / len( ), y / len( ), z / len( ) );
    }

    float dot ( const Vect3D &other ) const
    {
        return x * other.x + y * other.y + z * other.z;
    }
};

/**
 * @struct Sphere raystruct.h
 * @brief Sphere with position, radius, color, and specularity
 */
struct Sphere
{
    Vect3D center;
    float radius;
    COLORREF color;
    int specularity;
    float reflective;
    float raidus_squared;

    Sphere ( Vect3D center    = { },
             float radius     = 0,
             COLORREF color   = RGB( 0, 0, 0 ),
             int specularity  = 0,
             float reflective = 0 ) :
        center( center ),
        radius( radius ),
        color( color ),
        specularity( specularity ),
        reflective( reflective ),
        raidus_squared( radius * radius )
    {
    }
};

/**
 * @struct QuadraticRoots raystruct.h
 * @brief Simple two float struct to return answer from the quadratic formula
 */
struct QuadraticRoots
{
    float t1;
    float t2;

    QuadraticRoots ( float t1 = 0, float t2 = 0 ) :
        t1( t1 ), t2( t2 )
    {
    }
};

/**
 * @struct Light raystruct.h
 * @brief Light for rendering, different modes - DIRECTIONAL, POINT, and AMBIENT
 */
struct Light
{
    enum LightType
    {
        DIRECTIONAL,
        POINT,
        AMBIENT
    } type;

    float intensity;
    Vect3D pos;
};

struct Camera
{
    Vect3D position;
    float yaw;
    float pitch;
    float roll;

    float DegreesToRadian ( float degrees )
    {
        return degrees * PI / 180.0;
    }

    /**
     * @brief Rotate around Y-axis (left-right rotation)
     * @param[in] direction - D returned by CanvasToViewPort()
     * @param[in] yaw - Degrees to rotate
     * @return Rotated vector
     */
    Vect3D RotateYaw ( Vect3D direction, float yaw )
    {
        float rad  = DegreesToRadian( yaw );
        float cosY = cos( rad );
        float sinY = sin( rad );

        return Vect3D( direction.x * cosY + direction.z * sinY,
                       direction.y,
                       -direction.x * sinY + direction.z * cosY );
    }

    /**
     * @brief Rotate around X-axis (up-down rotation)
     * @param direction - D returned by CanvasToViewPort()
     * @param pitch - Degrees to rotate
     * @return Rotated vector
     */
    Vect3D RotatePitch ( Vect3D direction, float pitch )
    {
        float rad  = DegreesToRadian( pitch );
        float cosX = cos( rad );
        float sinX = sin( rad );

        return Vect3D( direction.x,
                       direction.y * cosX - direction.z * sinX,
                       direction.y * sinX + direction.z * cosX );
    }

    /**
     * @brief Rotate around Z-axis (side-side rotation)
     * @param direction - D returned by CanvasToViewPort()
     * @param roll - Degrees to rotate
     * @return Rotated Vector
     */
    Vect3D RotateRoll ( Vect3D direction, float roll )
    {
        float rad  = DegreesToRadian( roll );
        float cosZ = cos( rad );
        float sinZ = sin( rad );

        return Vect3D( direction.x * cosZ - direction.y * sinZ,
                       direction.x * sinZ + direction.y * cosZ,
                       direction.z );
    }

    Vect3D ApplyCameraRotation ( Vect3D direction, Camera cam )
    {
        direction = RotateYaw( direction, cam.yaw );
        direction = RotatePitch( direction, cam.pitch );
        direction = RotateRoll( direction, cam.roll );

        return direction;
    }

    /**
     * @brief Calculates normalized vector with forward direction for use in " W
     * = move forward "
     * @return Normalized vector
     */
    Vect3D CalculateForwardFromEuler ( )
    {
        float rPitch   = DegreesToRadian( pitch );
        float rYaw     = DegreesToRadian( yaw );
        float cosPitch = cos( rPitch );
        float sinPitch = sin( rPitch );
        float cosYaw   = cos( rYaw );
        float sinYaw   = sin( rYaw );

        return Vect3D( cosPitch * sinYaw, sinPitch, cosPitch * cosYaw ).norm( );
    }

    /**
     * @brief Moves camera forward
     * @param move_speed - Movement multiplier, backwards < 0 < forewards
     */
    void MoveForward ( float moveSpeed )
    {
        Vect3D forward  = CalculateForwardFromEuler( );
        position.x     += forward.x * moveSpeed;
        position.y     += forward.y * moveSpeed;
        position.z     += forward.z * moveSpeed;
    }

    /**
     * @brief Moves camera sideways
     * @param move_speed - Movemet multiplier, right < 0 < left
     */
    void MoveSideways ( float moveSpeed )
    {
        Vect3D right  = CalculateForwardFromEuler( ).cross( Vect3D( 0, 1, 0 ) );
        position.x   += right.x * moveSpeed;
        position.y   += right.y * moveSpeed;
        position.z   += right.z * moveSpeed;
    }
};

/**
 * @struct Intersection raystruct.h
 * @brief Pointer to sphere and number T, signifies an intersection between a
 * sphere and a vector */
struct Intersection
{
    Sphere *sphere;
    float point;

    Intersection ( Sphere *closest_sphere = NULL, float closest_t = INFINITY ) :
        sphere( closest_sphere ), point( closest_t )
    {
    }
};