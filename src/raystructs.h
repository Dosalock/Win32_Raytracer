/*******************************************************************************
 *
 *  @brief     Raytrace structs
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 ******************************************************************************/

#ifndef __RAYSTRUCTS_H__
#define __RAYSTRUCTS_H__

/*-----------------------------Includes------------------------------*/
#include <Windows.h>
#include <cmath>

/*-----------------------------Structs-------------------------------*/

/**
 * @struct WideColor raystructs.h raystructs
 * @brief A color struct with 32 bit channels to prevent overflowing
 */
struct WideColor
{
    uint32_t red;
    uint32_t green;
    uint32_t blue;

    WideColor ( uint32_t red = 0, uint32_t green = 0, uint32_t blue = 0 ) :
        red( red ), green( green ), blue( blue )
    {
    }

    WideColor operator+ ( const WideColor &other ) const
    {
        return WideColor( red + other.red,
                          green + other.green,
                          blue + other.blue );
    }

    WideColor operator- ( const WideColor &other ) const
    {
        return WideColor( red - other.red,
                          green - other.green,
                          blue - other.blue );
    }
};

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
        return sqrtf( x * x + y * y + z * z );
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
    WideColor color;
    uint32_t specularity;
    float reflective;
    float raidus_squared;

    Sphere ( Vect3D center        = { },
             float radius         = 0.0f,
             WideColor color      = { 0, 0, 0 },
             uint32_t specularity = 0,
             float reflective     = 0.0f ) :
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

private:

    /**
     * @brief Given a degree it calculates the radian equivalent
     * @param degrees - Angle in degrees
     * @return Radian equivalent of given angle
     */
    float DegreesToRadian ( _In_ const float &degrees )
    {
        return degrees * 0.01745329252f; /* degrees * PI / 180.0f */
    }

    /**
     * @brief Rotate around Y-axis (left-right rotation)
     * @param[in] direction - D returned by CanvasToViewPort()
     * @param[in] yaw - Degrees to rotate
     * @return Rotated vector
     */
    Vect3D RotateYaw ( _In_ const Vect3D &direction, _In_ const float &yaw )
    {
        float rad  = DegreesToRadian( yaw );
        float cosY = cosf( rad );
        float sinY = sinf( rad );

        return Vect3D( direction.x * cosY + direction.z * sinY,
                       direction.y,
                       -direction.x * sinY + direction.z * cosY );
    }

    /**
     * @brief Rotate around X-axis (up-down rotation)
     * @param[in] direction - D returned by CanvasToViewPort()
     * @param[in] pitch - Degrees to rotate
     * @return Rotated vector
     */
    Vect3D RotatePitch ( _In_ const Vect3D &direction, _In_ const float &pitch )
    {
        float rad  = DegreesToRadian( pitch );
        float cosX = cosf( rad );
        float sinX = sinf( rad );

        return Vect3D( direction.x,
                       direction.y * cosX - direction.z * sinX,
                       direction.y * sinX + direction.z * cosX );
    }

    /**
     * @brief Rotate around Z-axis (side-side rotation)
     * @param[in] direction - D returned by CanvasToViewPort()
     * @param[in] roll - Degrees to rotate
     * @return Rotated Vector
     */
    Vect3D RotateRoll ( _In_ const Vect3D &direction, _In_ const float &roll )
    {
        float rad  = DegreesToRadian( roll );
        float cosZ = cosf( rad );
        float sinZ = sinf( rad );

        return Vect3D( direction.x * cosZ - direction.y * sinZ,
                       direction.x * sinZ + direction.y * cosZ,
                       direction.z );
    }

    /**
     * @brief Calculates which way is forward for the camera
     * @note for use in " W = move forward "
     * @return Normalized vector
     */
    Vect3D CalculateForwardFromEuler ( )
    {
        float rPitch   = DegreesToRadian( pitch );
        float rYaw     = DegreesToRadian( yaw );
        float cosPitch = cosf( rPitch );
        float sinPitch = sinf( rPitch );
        float cosYaw   = cosf( rYaw );
        float sinYaw   = sinf( rYaw );

        return Vect3D( cosPitch * sinYaw, sinPitch, cosPitch * cosYaw ).norm( );
    }

public:

    /**
     * @brief Moves camera forward
     * @param[in] move_speed - Movement multiplier, backwards < 0 < forewards
     */
    void MoveForward ( _In_ const float &moveSpeed )
    {
        Vect3D forward  = CalculateForwardFromEuler( );
        position.x     += forward.x * moveSpeed;
        position.y     += forward.y * moveSpeed;
        position.z     += forward.z * moveSpeed;
    }

    /**
     * @brief Applies rotation to camera vector, this enables turning
     * @param[in,out] direction - Current viewing direction
     * @param[in,out] cam - Camera to apply rotation to
     * @return
     */
    Vect3D ApplyCameraRotation (_Inout_ Vect3D &direction, _Inout_ Camera &cam )
    {
        direction = RotateYaw( direction, cam.yaw );
        direction = RotatePitch( direction, cam.pitch );
        direction = RotateRoll( direction, cam.roll );

        return direction;
    }

    /**
     * @brief Moves camera sideways
     * @param[in] move_speed - Movemet multiplier, right < 0 < left
     */
    void MoveSideways ( _In_ const float &moveSpeed )
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

#endif // !__RAYSTRUCTS_H__