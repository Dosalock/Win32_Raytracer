#pragma once
#include "Windows.h"

typedef struct 
{
	double x;
	double y;
	double z;
} Vect3D;

typedef struct
{
	Vect3D center;
	double radius;
	COLORREF color;
}Sphere, *pSphere;

typedef struct
{
	double t1;
	double t2;
}QuadraticAnswer;
