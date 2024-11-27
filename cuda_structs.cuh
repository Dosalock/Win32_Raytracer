#pragma once
#include <cuda_runtime.h>
#include <cudart_platform.h>
#include <cuda_fp8.h>
#include <cudaTypedefs.h>
#include <cuda_device_runtime_api.h>


namespace f4_util
{
	float dot(float4 a, float4 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	float4 normalize(float4 v)
	{
		return rsqrt(dot(v, v)) * v;
	}
}
