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


/*-----------------------------Structs-------------------------------*/

/**
 * @struct Vect3D raystruct.h
 * @brief Three dimensional vector
 * 
 */
struct Vect3D 
{
	double x; // @brief Represents the vector's position along the X-axis
	double y; // @brief Represents the vector's position along the Y-axis
	double z; // @brief Represents the vector's position along the Z-axis

	Vect3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) { }
	
	Vect3D operator-(const Vect3D& other) const { return Vect3D(x - other.x, y - other.y, z - other.z); }
	Vect3D operator+(const Vect3D& other) const { return Vect3D(x + other.x, y + other.y, z + other.z); }
	Vect3D operator*(const Vect3D& other) const { return Vect3D(x * other.x, y * other.y, z * other.z); }
	Vect3D operator/(const Vect3D& other) const { return Vect3D(x / other.x, y / other.y, z / other.z); }
	Vect3D operator/(const double& other) const { return Vect3D(x / other, y / other, z / other); }
	Vect3D operator*(const double& other) const { return Vect3D(x * other, y * other, z * other); }
	Vect3D norm() const { return Vect3D(x * -1, y * -1, z * -1); }
	double len() const { return sqrt(x * x + y * y + z * z); }
	double dot(const Vect3D& other) const { return x * other.x + y * other.y + z * other.z; }
};


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
		reflective(reflective) {}
};


/**
 * @struct QuadraticAnswer raystruct.h
 * @brief Simple two double struct to return answer from the quadratic formula
 */
struct QuadraticAnswer
{
	double t1;
	double t2;
	QuadraticAnswer(double t1 = 0, double t2 = 0) : t1(t1), t2(t2) {}
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


/**
 * @struct Intersection raystruct.h
 * @brief Pointer to sphere and number T, signifies an intersection between a sphere and a vector
 */
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

