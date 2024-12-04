#pragma once
#define _USE_MATH_DEFINES


#include "Windows.h"
#include "cuda_structs.cuh"

#include <cuda_fp16.h>

__host__ void
  Draw_Caller ( BYTE **pLpvBits, Camera &cam, Sphere *scene, Light *lights );

void ExitCleanup ( );

__device__ intersection ClosestIntersection ( const float4 &O,
                                              const float4 &D,
                                              const double &t_min,
                                              const double &t_max );
__device__ float4       CanvasToViewPort ( const int &x, const int &y );
