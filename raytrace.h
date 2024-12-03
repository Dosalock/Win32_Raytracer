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
//#include "raystructs.h"
#include "raytrace.cuh"

/*------------Variable Declarations---------------*/

/*------------Function Declarations---------------*/

///**
// *  @brief Returns reflected ray of R from normal vector N  
// *  @param[in] R - Ray that you want its reflection of
// *  @param[in] N - The normal
// *  @retval   - The reflected ray
// */
//Vect3D ReflectRay(const Vect3D &R, const Vect3D &N);
//
//
///**
// * @brief Sets positions and value of objects and lights
// */
//void CreateScene();
//
//
///**
// * @brief Calculates how bright a point is
// * @param[in] P - Intersection 
// * @param[in] N - Normalized vector from center of sphere
// * @param[in] V - Vector from point to camera
// * @param[in] s - Specularity value of object 
// * @return Intensity multiplier
// */
//double CalcLight(const Vect3D &P, const Vect3D &N, const Vect3D &V, const int &s);
//
/**
 * @brief Main draw function, sets all the pixel values
 * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels 
 * @param[in] width - Viewport width in pixels
 * @param[in] height - Viewport height in pixels
 */
void Draw(BYTE **pLpvBits, const int &width, const int &height, Camera &cam);
//
///**
// * @brief Initzialises the scene, bitmap height & width etc.
// * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels
// * @param[in] window - Handle to rectangle of viewport
// * @param[in,out] pHBitmap - Handle to a bitmap 
// */
//void Init(BYTE **pLpvBits, const RECT *window, HBITMAP *pHBitmap);
//
//
///**
// * @brief Returns points of intersection between a ray and sphere
// * @param[in] O - Point of ray origin
// * @param[in] D - Ray direction from O
// * @param[in] sphere - Sphere to check if ray D from origin O will intersect
// * @return t, smallest of the two roots or INFINITY if no point intersects
// */
//double IntersectRaySphere(const Vect3D &O, const Vect3D &D, const Sphere &sphere,  const double &dDot);
//
//
///**
// * @brief Calculates corresponding point in space of pixel[x][y]
// * @param[in] x - Pixel co-ordinate in x direction 
// * @param[in] y - Pixel co-ordinate in y direction
// * @param[in] width - Viewport width
// * @param[in] height - Viewport height
// * @return Point in space of specified pixel
// */
//Vect3D CanvasToViewport(const int &x, const int &y, const int &width, const int &height);
//
//
///**
// * @brief Calculate the color of the pixel at a point D
// * @param[in] O - Origin of ray, usually camera position 
// * @param[in] D - Ray direction
// * @param[in] t_min - Minimum range of points along ray
// * @param[in] t_max - Maximum range of poitns along ray
// * @param[in] recursionDepth - How many times to calculate reflections
// * @return Color of point D from point  
// */
//COLORREF TraceRay(const Vect3D &O, const Vect3D &D, const double &t_min, const double &t_max, const int &recursionDepth);
//
//
///**
// * @brief 
// * @param[in] O - Origin of ray, usually camera position 
// * @param[in] D - Ray direction
// * @param[in] t_min - Minimum range of points along ray
// * @param[in] t_max - Maximum range of points along ray
// * @return Point of intersection to a sphere from O with direction D
// */
//Intersection ClosestIntersection(const Vect3D &O, const Vect3D &D, const double &t_min, const double &t_max);

/*------------Template Declarations---------------*/

template <typename T>
concept Scalar = std::is_scalar_v<T>;

template <Scalar T>
bool IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}
