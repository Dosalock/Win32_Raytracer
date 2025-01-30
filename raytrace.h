/*******************************************************************************
 *
 *  @file      raytrace.h
 *  @brief     Raytrace functions, draw loop vector calculations
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 ******************************************************************************/
#pragma once


/*------------------Includes---------------------*/
#include "raystructs.h"

/*------------Variable Declarations---------------*/


/*------------Function Declarations---------------*/

/**
 *  @brief Returns a ray reflected in account with the normal
 *  @param[in] ray_to_reflect - Ray that you want its reflection of
 *  @param[in] sphere_normal - The normal
 *  @return - The reflected ray
 */
Vect3D ReflectRay ( Vect3D ray_to_reflect, Vect3D sphere_normal );


/**
 * @brief Sets positions and value of objects and lights
 * @param[in,out] scene - Pointer to where we allocate the spheres
 * @param[in,out] lights - Pointer to where we allocate the lights
 */
void CreateScene ( Sphere *scene, Light *lights );


/**
 * @brief Calculates how bright a point is
 * @param[in] intersection_point - Intersection
 * @param[in] normal_sphere_vector - Normalized vector from center of sphere
 * @param[in] point_to_camera - Vector from point to camera
 * @param[in] sphere_specularity - Specularity value of object
 * @return Float - Intensity multiplier
 */
float CalcLight ( Vect3D intersection_point,
                  Vect3D normalized_sphere_vector,
                  Vect3D point_to_camera,
                  uint32_t sphere_specularity,
                  Sphere *scene,
                  Light *lights );

/**
 * @brief Main draw function, sets all the pixel values
 * @param[in,out] p_lpv_bits - Pointer to buffer of viewport pixels
 * @param[in]     width - Viewport width in pixels
 * @param[in]     height - Viewport height in pixels
 * @param[in]     camera - Player camera
 * @param[in]     scene - Pointer to scene objects (spheres)
 * @param[in]     lights - Pointer to scene lights
 */
void Draw ( BYTE **p_lpv_bits,
            int width,
            int height,
            Camera camera,
            Sphere *scene,
            Light *lights );

/**
 * @brief Initzialises the scene, bitmap height & width etc.
 * @param[in,out] p_lpv_bits - Pointer to buffer of viewport pixels
 * @param[in,out] p_h_bitmap - Handle to a bitmap
 * @param[in]     window - Handle to rectangle of viewport
 */
void Init ( BYTE **p_lpv_bits, HBITMAP *p_h_bitmap, RECT *window );


/**
 * @brief Returns points of intersection between a ray and sphere
 * @param[in] origin - Point of ray origin
 * @param[in] direction_from_origin - Ray direction from origin
 * @param[in] sphere - Sphere to check if ray will intersect
 * @return  Root of possible point,
 *			INFINITY if no points found
 */
float IntersectRaySphere ( Vect3D origin,
                           Vect3D direction_from_origin,
                           Sphere sphere,
                           float direction_from_origin_dot_product );


/**
 * @brief Calculates corresponding point in space of pixel[x][y]
 * @param[in] x - Pixel co-ordinate in x direction
 * @param[in] y - Pixel co-ordinate in y direction
 * @param[in] width - Viewport width
 * @param[in] height - Viewport height
 * @return Translated canvas coordinate of specified pixel
 */
Vect3D CanvasToViewport ( int x, int y, int width, int height );


/**
 * @brief Calculate the color of the pixel at a point direction_from_origin
 * @param[in] origin - Origin of ray, usually camera position
 * @param[in] direction_from_origin - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of poitns along ray
 * @param[in] recursionDepth - How many times to calculate reflections
 * @return Color of point direction_from_origin from point
 */
COLORREF TraceRay ( Vect3D O,
                    Vect3D D,
                    float t_min,
                    float t_max,
                    int recursionDepth,
                    Sphere *scene,
                    Light *lights );


/**
 * @brief
 * @param[in] origin - Origin of ray, usually camera position
 * @param[in] direction_from_origin - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of points along ray
 * @return Point of intersection to a sphere from origin with direction
 * direction_from_origin
 */
Intersection ClosestIntersection ( Vect3D origin,
                                   Vect3D direction_from_origin,
                                   float t_min,
                                   float t_max,
                                   Sphere *scene );


/*------------Template Declarations---------------*/
template<typename T>
concept Scalar = std::is_scalar_v<T>;

/**
 * @brief Returns true if a value is between, low and high
 * @tparam T - Any scalar, std::is_scalar_v<T> == true
 * @param value - value to check if its in bounds
 * @param low - lower bound
 * @param high - upper bound
 * @return
 */
template<Scalar T>
bool IsInBounds ( const T &value, const T &low, const T &high )
{
    return !( value < low ) && ( value < high );
}
