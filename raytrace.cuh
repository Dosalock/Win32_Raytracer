#pragma once
#define _USE_MATH_DEFINES

#include "Windows.h"
#include <cstdlib>
#include <memory>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "raystructs.h"
#include <cudart_platform.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_helpers.h>
#include <cuda_awbarrier_helpers.h>
__host__ void Draw_Caller(BYTE **pLpvBits, Camera &cam);
void ExitCleanup();

__device__ Intersection ClosestIntersection(const float4 &O, const float4 &D, const double &t_min, const double &t_max);
__device__ float4 CanvasToViewPort(const int &x, const int &y);


