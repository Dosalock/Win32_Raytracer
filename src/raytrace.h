/*******************************************************************************
 *
 *  @file      raytrace.h
 *  @brief     Raytrace functions, draw loop vector calculations
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 ******************************************************************************/
#ifndef __RAYTRACE_H__
#define __RAYTRACE_H__

/*------------------Includes---------------------*/
#include "helper.h"
#include "raystructs.h"
#include <vector>
/*------------Variable Declarations---------------*/


/*------------Function Declarations---------------*/

/**
 *  @brief Returns a ray reflected in account with the normal
 *  @param[in] ray_to_reflect - Ray that you want its reflection of
 *  @param[in] sphere_normal - The normal
 *  @return - The reflected ray
 */
Vect3D ReflectRay ( _In_ const Vect3D ray_to_reflect,
                    _In_ const Vect3D sphere_normal );


/**
 * @brief Sets positions and value of objects and lights
 * @param[in,out] scene - Pointer to where we allocate the spheres
 * @param[in,out] lights - Pointer to where we allocate the lights
 */
void CreateScene ( _Out_ std::vector<Sphere> &scene,
                   _Out_ std::vector<Light> &lights );


/**
 * @brief Calculates how bright a point is
 * @param[in] intersection_point - Intersection
 * @param[in] normal_sphere_vector - Normalized vector from center of sphere
 * @param[in] point_to_camera - Vector from point to camera
 * @param[in] sphere_specularity - Specularity value of object
 * @return Float - Intensity multiplier
 */
float CalcLight ( _In_ const Vect3D intersection_point,
                  _In_ const Vect3D normalized_sphere_vector,
                  _In_ const Vect3D point_to_camera,
                  _In_ const uint32_t sphere_specularity,
                  _In_ const std::vector<Sphere> &scene,
                  _In_ const std::vector<Light> &lights );

/**
 * @brief Main draw function, sets all the pixel values
 * @param[in,out] p_lpv_bits - Pointer to buffer of viewport pixels
 * @param[in]     width - Viewport width in pixels
 * @param[in]     height - Viewport height in pixels
 * @param[in]     camera - Player camera
 * @param[in]     scene - Pointer to scene objects (spheres)
 * @param[in]     lights - Pointer to scene lights
 */
void Draw ( _Inout_ BYTE **p_lpv_bits,
            _In_ const uint16_t width,
            _In_ const uint16_t height,
            _In_ Camera camera,
            _In_ const std::vector<Sphere> &scene,
            _In_ const std::vector<Light> &lights );

/**
 * @brief Initzialises the scene, bitmap height & width etc.
 * @param[in,out] p_lpv_bits - Pointer to buffer of viewport pixels
 * @param[in,out] p_h_bitmap - Handle to a bitmap
 * @param[in]     window - Handle to rectangle of viewport
 */
void Init ( _Inout_ BYTE **p_lpv_bits,
            _Inout_ HBITMAP *p_h_bitmap,
            _In_ const RECT *window );


/**
 * @brief Returns points of intersection between a ray and sphere
 * @param[in] origin - Point of ray origin
 * @param[in] direction_from_origin - Ray direction from origin
 * @param[in] sphere - Sphere to check if ray will intersect
 * @param[in] direction_from_origin_dot_product - Vector dot product
 * @return  Root of possible point,
 *			INFINITY if no points found
 */
float IntersectRaySphere ( _In_ const Vect3D origin,
                           _In_ const Vect3D direction_from_origin,
                           _In_ const Sphere sphere,
                           _In_ const float direction_from_origin_dot_product );


/**
 * @brief Calculates corresponding point in space of pixel[x][y]
 * @param[in] x - Pixel co-ordinate in x direction
 * @param[in] y - Pixel co-ordinate in y direction
 * @param[in] width - Viewport width
 * @param[in] height - Viewport height
 * @return Translated canvas coordinate of specified pixel
 */
Vect3D CanvasToViewport ( _In_ const uint16_t x,
                          _In_ const uint16_t y,
                          _In_ const uint16_t width,
                          _In_ const uint16_t height );


/**
 * @brief Calculate the color of the pixel at a point direction_from_origin
 *
 * @param[in] origin - Origin of ray, usually camera position
 * @param[in] direction_from_origin - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of poitns along ray
 * @param[in] recursion_depth - How many times to calculate reflections
 *
 * @return Color of point direction_from_origin from point
 */
WideColor TraceRay ( _In_ const Vect3D origin,
                     _In_ const Vect3D destination,
                     _In_ const float t_min,
                     _In_ const float t_max,
                     _In_ const uint8_t recursion_depth,
                     _In_ const std::vector<Sphere> &scene,
                     _In_ const std::vector<Light> &lights );


/**
 * @brief Calculates the closest point from origin to a sphere
 *
 * @param[in] origin - Origin of ray, usually camera position
 * @param[in] direction_from_origin - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of points along ray
 * @param[in] scene - Pointer to scene buffer with objects(spheres)
 *
 * @return Intersection with the closest point and intersecting sphere
 */
Intersection ClosestIntersection ( _In_ const Vect3D origin,
                                   _In_ const Vect3D direction_from_origin,
                                   _In_ const float t_min,
                                   _In_ const float t_max,
                                   _In_ const std::vector<Sphere> &scene );


#endif // !__RAYTRACE_H__
