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

__host__ void Draw_Caller(BYTE **pLpvBits);