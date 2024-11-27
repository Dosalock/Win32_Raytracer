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
#define _USE_MATH_DEFINES

#include <cmath>
#include "Windows.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
/*-----------------------------Structs-------------------------------*/

/**
 * @struct Vect3D raystruct.h
 * @brief Three dimensional vector
 * 
 */
//struct Vect3D 
//{
//	double x; // @brief Represents the vector's position along the X-axis
//	double y; // @brief Represents the vector's position along the Y-axis
//	double z; // @brief Represents the vector's position along the Z-axis
//
//	Vect3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) { }
//	
//	__host__ __device__ Vect3D operator-(const Vect3D &other) const { return Vect3D(x - other.x, y - other.y, z - other.z); }
//	__host__ __device__ Vect3D operator+(const Vect3D &other) const { return Vect3D(x + other.x, y + other.y, z + other.z); }
//	__host__ __device__ Vect3D operator*(const Vect3D &other) const { return Vect3D(x * other.x, y * other.y, z * other.z); }
//	__host__ __device__ Vect3D operator/(const Vect3D &other) const { return Vect3D(x / other.x, y / other.y, z / other.z); }
//	__host__ __device__ Vect3D operator/(const double &other) const { return Vect3D(x / other, y / other, z / other); }
//	__host__ __device__ Vect3D operator*(const double &other) const { return Vect3D(x * other, y * other, z * other); }
//	__host__ __device__ Vect3D cross(const Vect3D &other) const { return Vect3D(y * other.z - z * other.y, z*other.x - x*other.z, x*other.y - y*other.z); }
//	__host__ __device__ Vect3D invert() const { return Vect3D(x * -1, y * -1, z * -1); }
//	__host__ __device__ double len() const { return rsqrtf(x * x + y * y + z * z); }
//	__host__ __device__ Vect3D norm() const { return Vect3D(x, y, z) / len(); }
//	__host__ __device__ double dot(const Vect3D &other) const { return x * other.x + y * other.y + z * other.z; }
//};


/**
 * @struct Sphere raystruct.h
 * @brief Sphere with position, radius, color, and specularity
 */
struct Sphere
{
	Vect3D center;
	double radius;
	COLORREF color;
	int specularity;
	double reflective;
	double sRadius;

	Sphere(
		Vect3D center = {},
		double radius = 0,
		COLORREF color = RGB(0, 0, 0),
		int specularity = 0,
		double reflective = 0)
		:
		center(center),
		radius(radius),
		color(color),
		specularity(specularity),
		reflective(reflective),
		sRadius(radius*radius){}
};


/**
 * @struct QuadraticRoots raystruct.h
 * @brief Simple two double struct to return answer from the quadratic formula
 */
struct QuadraticRoots
{
	double t1;
	double t2;
	QuadraticRoots(double t1 = 0, double t2 = 0) : t1(t1), t2(t2) {}
};


/**
 * @struct Light raystruct.h
 * @brief Light for rendering, different modes - DIRECTIONAL, POINT, and AMBIENT
 */
struct Light 
{
	enum LightType { DIRECTIONAL, POINT, AMBIENT }  type;
	double intensity;
	Vect3D pos;
};

struct Camera
{
	Vect3D position;
	double yaw;
	double pitch;
	double roll;

	double DegreesToRadian(double degrees)
	{
		return degrees * M_PI / 180.0;
	}

	/**
	 * @brief Rotate around Y-axis (left-right rotation)
	 * @param[in] direction - D returned by CanvasToViewPort()
	 * @param[in] yaw - Degrees to rotate 
	 * @return Rotated vector
	 */
	Vect3D RotateYaw(Vect3D direction, double yaw)
	{
		double rad = DegreesToRadian(yaw);
		double cosY = cos(rad);
		double sinY = sin(rad);
		
		return Vect3D(
			direction.x * cosY + direction.z * sinY,
			direction.y,
			-direction.x * sinY + direction.z * cosY
		);
	}

	/**
	 * @brief Rotate around X-axis (up-down rotation)
	 * @param direction - D returned by CanvasToViewPort()
	 * @param pitch - Degrees to rotate
	 * @return Rotated vector
	 */
	Vect3D RotatePitch(Vect3D direction, double pitch)
	{
		
		double rad = DegreesToRadian(pitch);
		double cosX = cos(rad);
		double sinX = sin(rad);

		return Vect3D(
			direction.x,
			direction.y * cosX - direction.z * sinX,
			direction.y * sinX + direction.z * cosX
		);
	}

	/**
	 * @brief Rotate around Z-axis (side-side rotation)
	 * @param direction - D returned by CanvasToViewPort()
	 * @param roll - Degrees to rotate
	 * @return Rotated Vector
	 */
	Vect3D RotateRoll(Vect3D direction, double roll)
	{

		double rad = DegreesToRadian(roll);
		double cosZ = cos(rad);
		double sinZ = sin(rad);
		
		return Vect3D(
			direction.x * cosZ - direction.y * sinZ,
			direction.x * sinZ + direction.y * cosZ,
			direction.z
		);
	}

	Vect3D ApplyCameraRotation(Vect3D direction, Camera cam)
	{
		direction = RotateYaw(direction, cam.yaw);
		direction = RotatePitch(direction, cam.pitch);
		direction = RotateRoll(direction, cam.roll);

		return direction;

	}


	/**
	 * @brief Calculates normalized vector with forward direction for use in " W = move forward "
	 * @return Normalized vector
	 */
	Vect3D CalculateForwardFromEuler()
	{	double rPitch = DegreesToRadian(pitch);
		double rYaw = DegreesToRadian(yaw);
		float cosPitch = cos(rPitch);
		float sinPitch = sin(rPitch);
		float cosYaw = cos(rYaw);
		float sinYaw = sin(rYaw);

		return Vect3D(cosPitch * sinYaw, sinPitch, cosPitch * cosYaw).norm();
	}
	/**
	 * @brief Moves camera forward
	 * @param moveSpeed - Movement multiplier, backwards < 0 < forewards
	 */
	void MoveForward(double moveSpeed)
	{
		Vect3D forward = CalculateForwardFromEuler();
		position.x += forward.x * moveSpeed;
		position.y += forward.y * moveSpeed;
		position.z += forward.z * moveSpeed;
	}
	/**
	 * @brief Moves camera sideways
	 * @param moveSpeed - Movemet multiplier, right < 0 < left
	 */
	void MoveSideways(double moveSpeed)
	{
		Vect3D right = CalculateForwardFromEuler().cross(Vect3D(0,1,0));
		position.x += right.x * moveSpeed;
		position.y += right.y * moveSpeed;
		position.z += right.z * moveSpeed;
	}

};
/**
 * @struct Intersection raystruct.h
 * @brief Pointer to sphere and number T, signifies an intersection between a sphere and a vector */
struct Intersection 
{
	Sphere *closest_sphere;
	double closest_t;

	Intersection(Sphere *closest_sphere = NULL, 
				 double closest_t = INFINITY)
		         :
		         closest_sphere(closest_sphere),
		         closest_t(closest_t) {}
};

