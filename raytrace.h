#pragma once
#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "raystructs.h"

extern Vect3D D;
extern Vect3D N;
extern Vect3D P;
extern const Vect3D O;
extern Sphere scene[4];
extern Light lights[3];



void CreateScene();

double CalcLight();

void Draw(BYTE **pLpvBits, int width, int height);

void Init(BYTE **pLpvBits, RECT *window, HBITMAP *pHBitmap);

QuadraticAnswer IntersectRaySphere(Vect3D D, Sphere sphere);

Vect3D CanvasToViewport(int x, int y, int width, int height);

COLORREF TraceRay(Vect3D D);

#endif
