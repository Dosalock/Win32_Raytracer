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
 *  @brief Returns reflected ray of R from normal vector N
 *  @param[in] R - Ray that you want its reflection of
 *  @param[in] N - The normal
 *  @retval   - The reflected ray
 */
Vect3D ReflectRay ( Vect3D R, Vect3D N );


/**
 * @brief Sets positions and value of objects and lights
 */
void CreateScene ( Sphere* scene, Light *lights );


/**
 * @brief Calculates how bright a point is
 * @param[in] P - Intersection
 * @param[in] N - Normalized vector from center of sphere
 * @param[in] V - Vector from point to camera
 * @param[in] s - Specularity value of object
 * @return Intensity multiplier
 */
double CalcLight ( Vect3D P, Vect3D N, Vect3D V, int s, Sphere *scene, Light *lights);

/**
 * @brief Main draw function, sets all the pixel values
 * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels
 * @param[in] width - Viewport width in pixels
 * @param[in] height - Viewport height in pixels
 */
void Draw ( BYTE **pLpvBits, int width, int height, Camera cam, Sphere* scene, Light *lights );

/**
 * @brief Initzialises the scene, bitmap height & width etc.
 * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels
 * @param[in] window - Handle to rectangle of viewport
 * @param[in,out] pHBitmap - Handle to a bitmap
 */
void Init ( BYTE **pLpvBits, RECT *window, HBITMAP *pHBitmap );


/**
 * @brief Returns points of intersection between a ray and sphere
 * @param[in] O - Point of ray origin
 * @param[in] D - Ray direction from O
 * @param[in] sphere - Sphere to check if ray D from origin O will intersect
 * @return  Root of possible point,
 *			INFINITY if no points found
 */
float
	IntersectRaySphere ( Vect3D O, Vect3D D, Sphere sphere, double dDot );


/**
 * @brief Calculates corresponding point in space of pixel[x][y]
 * @param[in] x - Pixel co-ordinate in x direction
 * @param[in] y - Pixel co-ordinate in y direction
 * @param[in] width - Viewport width
 * @param[in] height - Viewport height
 * @return Point in space of specified pixel
 */
Vect3D CanvasToViewport ( int x, int y, int width, int height );


/**
 * @brief Calculate the color of the pixel at a point D
 * @param[in] O - Origin of ray, usually camera position
 * @param[in] D - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of poitns along ray
 * @param[in] recursionDepth - How many times to calculate reflections
 * @return Color of point D from point
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
 * @param[in] O - Origin of ray, usually camera position
 * @param[in] D - Ray direction
 * @param[in] t_min - Minimum range of points along ray
 * @param[in] t_max - Maximum range of points along ray
 * @return Point of intersection to a sphere from O with direction D
 */
Intersection
	ClosestIntersection ( Vect3D O, Vect3D D, double t_min, double t_max, Sphere* scene);


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
