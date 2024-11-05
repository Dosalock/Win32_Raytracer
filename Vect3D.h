#pragma once
#include "Windows.h"
#include <cmath>

enum class LightType
{
	DIRECTIONAL, POINT, AMBIENT
};


struct Vect3D 
{
	double x;
	double y;
	double z;
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

struct Sphere
{
	Vect3D center;
	double radius;
	COLORREF color;

	Sphere(
		Vect3D center = {},
		double radius = 0,
		COLORREF color = RGB(0, 0, 0))
		:
		center(center),
		radius(radius),
		color(color) {}
	
};

struct QuadraticAnswer
{
	double t1;
	double t2;
	QuadraticAnswer(double t1 = 0, double t2 = 0) : t1(t1), t2(t2) {}
};

struct Light 
{
	enum LightType { DIRECTIONAL, POINT, AMBIENT }  type;
	float intensity;
	Vect3D pos;
};
