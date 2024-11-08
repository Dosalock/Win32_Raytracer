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

extern Sphere scene[4];
extern Light lights[3];


/*------------Function Declarations---------------*/

void CreateScene();

double CalcLight(Vect3D P, Vect3D N, Vect3D V, int s);

void Draw(BYTE **pLpvBits, int width, int height);

void Init(BYTE **pLpvBits, RECT *window, HBITMAP *pHBitmap);

QuadraticAnswer IntersectRaySphere(Vect3D O, Vect3D D, Sphere sphere);

Vect3D CanvasToViewport(int x, int y, int width, int height);

COLORREF TraceRay(Vect3D O, Vect3D D, double t_min, double t_max, int recursionDepth);

Intersection ClosestIntersection(Vect3D O, Vect3D D, double t_min, double t_max);

