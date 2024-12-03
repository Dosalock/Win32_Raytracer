#pragma once
#define _USE_MATH_DEFINES


#include "Windows.h"
#include <cstdlib>
#include <tuple>
#include "cuda_structs.cuh"


struct Camera
{
	float4 position;
	float yaw;
	float pitch;
	float roll;
};

struct Sphere
{
	float4 center;
	float radius;
	COLORREF color;
	int specularity;
	float reflective;
	float sRadius;

	Sphere(
		float4 center = {},
		float radius = 0,
		COLORREF color = RGB(0, 0, 0),
		int specularity = 0,
		float reflective = 0)
		:
		center(center),
		radius(radius),
		color(color),
		specularity(specularity),
		reflective(reflective),
		sRadius(radius *radius) {}
};

struct Light
{
	enum LightType { directional, point, ambient }  type;
	float intensity;
	float4 pos;
};

__host__ void Draw_Caller(BYTE **pLpvBits, Camera &cam, Sphere *&scene, Light *&lights);
void ExitCleanup();

__device__ std::tuple<Sphere *, float> ClosestIntersection(const float4 &O, const float4 &D, const double &t_min, const double &t_max);
__device__ float4 CanvasToViewPort(const int &x, const int &y);


