#include "raytrace.cuh"

/*------------Template Declarations---------------*/


//__device__ Sphere scene[4] = {};
//__device__ Light lights[3] = {};
__device__ Sphere scene[];
__device__ Light lights[];

#define HEIGHT 1024
#define WIDTH 1024

__device__ Intersection ClosestIntersection(const Vect3D& O, const Vect3D& D, const double& t_min, const double& t_max);

__device__ bool IntersectionBounds(float T, float t_min, float t_max) {
	return (T > t_min && T < t_max);  // Strict inequality
}

__device__ Vect3D ReflectRay(const Vect3D& R, const Vect3D& N)
{
	return ((N * (N.dot(R))) * 2) - R;
}

__device__ double CalcLight(const Vect3D& P, const Vect3D& N, const Vect3D& V, const int& s)
{
	double intensity = 0.0;
	double t_max = 0;
	Vect3D L = {};
	Vect3D R = {};
	for (int i = 0; i < sizeof(lights) / sizeof(Light); i++)
	{
		if (lights[i].type == lights->AMBIENT)
		{
			intensity += lights[i].intensity;
		}
		else
		{
			if (lights[i].type == lights->POINT)
			{
				L = (lights[i].pos - P);
				t_max = 1;
			}
			else
			{
				L = lights[i].pos;
				t_max = INFINITY;
			}
			L = L;
			Intersection result_closest = ClosestIntersection(P, L, 0.00001, t_max);
			if (result_closest.closest_sphere != NULL)
			{
				continue;
			}

			double n_dot_l = N.dot(L);
			if (n_dot_l > 0)
			{
				intensity += lights[i].intensity * n_dot_l / (N.len() * L.len());
			}

			if (s != -1)
			{
				R = ReflectRay(L, N);
				double r_dot_v = R.dot(V);

				if (r_dot_v > 0)
				{
					intensity += lights[i].intensity * pow(r_dot_v / (R.len() * (V.len())), s);
				}

			}
		}
	}
	return intensity;
}

__device__ double IntersectRaySphere(const Vect3D& O, const Vect3D& D, const Sphere& sphere, const double& dDot)
{
	Vect3D CO = {};
	CO = O - sphere.center;
	
	double a = dDot;
	double b = 2 * CO.dot(D);
	double c = CO.dot(CO) - sphere.sRadius;

	double discr = b * b - 4 * a * c;

	if (discr < 0)
	{
		return INFINITY;
	}
	else if (discr == 0)
	{
		return -b / (2 * a);
	}

	double t = (-b - __dsqrt_rd(discr)) / (2 * a);		// Minimize compute only go for 1 root;

	return t;
}

__device__ Intersection ClosestIntersection(const Vect3D& O, const Vect3D& D, const double& t_min, const double& t_max)
{
	double closest_t = INFINITY;
	Sphere* closest_sphere = NULL;
	double d_dot_d = D.dot(D);		// Cache immutable value
	for (auto& x : scene)
	{
		double t = IntersectRaySphere(O, D, x, d_dot_d);

		if (IntersectionBounds(t, t_min, t_max) && t < closest_t)
		{
			closest_t = t;
			closest_sphere = const_cast<Sphere*>(&x);
		}
	}
	return Intersection(closest_sphere, closest_t);
}

__device__ COLORREF TraceRay(const Vect3D& O, const Vect3D& D, const double& t_min, const double& t_max, const int& recursionDepth)
{
	Vect3D N = {};
	Vect3D P = {};
	Vect3D R = {};

	Intersection result_closest = ClosestIntersection(O,D,t_min,t_max);
	Sphere *closest_sphere = result_closest.closest_sphere;
	double closest_t = result_closest.closest_t;
	if (closest_sphere == NULL)
	{
		return RGB(0, 0, 0);
	}

	P = O + (D * closest_t);
	N = (P - closest_sphere->center).norm();

	double res = CalcLight(P, N, D.invert(), closest_sphere->specularity);
	int r = (int)round(GetRValue(closest_sphere->color) * res);
	int g = (int)round(GetGValue(closest_sphere->color) * res);
	int b = (int)round(GetBValue(closest_sphere->color) * res);

	double refl = closest_sphere->reflective;

	if (recursionDepth <= 0 || refl <= 0)
	{
		return RGB(max(0, min(255, r)),
			max(0, min(255, g)),
			max(0, min(255, b)));
	}


	R = ReflectRay(D.invert(), N);
	COLORREF reflectedColor = TraceRay(P, R, t_min, t_max, recursionDepth - 1);

	int reflected_r = (int)round(GetRValue(reflectedColor)) * refl;
	int reflected_g = (int)round(GetGValue(reflectedColor)) * refl;
	int reflected_b = (int)round(GetBValue(reflectedColor)) * refl;


	return RGB(max(0, min(255, static_cast<int>(r * (1 - refl) + reflected_r))),
		max(0, min(255, static_cast<int>(g * (1 - refl) + reflected_g))),
		max(0, min(255, static_cast<int>(b * (1 - refl) + reflected_b))));

}

__device__ Vect3D CudaCanvasToViewPort(const int &x, const int &y)
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	double aspectRatio = static_cast<double>(WIDTH) / HEIGHT;

	// Map x and y to the viewport, adjusting by aspect ratio
	double fovMod = 1;
	double viewportX = (x - WIDTH / 2.0) * ((1.0 * fovMod) / WIDTH) * aspectRatio;
	double viewportY = -(y - HEIGHT/ 2.0) * ((1.0 * fovMod) / HEIGHT); // Flip Y to match 3D space orientation

	return Vect3D(viewportX, viewportY, 1);  // Z=1 for perspective projection
}

__global__ void cuda_Draw(BYTE *pLpvBits)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = (y * WIDTH + x) * 4;
	if (offset >= 0 && offset < WIDTH * HEIGHT * 4 - 4)
	{
		pLpvBits[offset + 0] = 255 - blockDim.x * 3; 
		pLpvBits[offset + 1] = 255 - blockIdx.y * 3; 
		pLpvBits[offset + 2] = 255 - blockIdx.z * 3; 
		pLpvBits[offset + 3] = 255;
	}

}


__host__ void Draw_Caller(BYTE ** pLpvBits)
{
	int buffer_size = WIDTH * HEIGHT * sizeof(BYTE) * 4;

	int N = 1024;

	dim3 threadsPB(16,16);
	dim3 numB(N/threadsPB.x, N / threadsPB.y);

	BYTE *cudaLpvBits;
	size_t src_pitch = ((WIDTH * 4 + 3) & ~3);	// AND with (NOT 3) ensures last two digits are always 0
	size_t dest_pitch;

	cudaMallocPitch(
				&cudaLpvBits,					
				&dest_pitch, 
				WIDTH * 4 * sizeof(BYTE),		// 4 bytes for each pixel; R, G, B, alpha
				HEIGHT);						// number of rows

	cudaMemcpy2D(
				cudaLpvBits,					// Destinaion
				dest_pitch,						
				*pLpvBits,						// Source
				src_pitch,		
				WIDTH * 4 * sizeof(BYTE),		
				HEIGHT,
				cudaMemcpyHostToDevice);

	
	cuda_Draw<<<numB, threadsPB>>>(cudaLpvBits);

	cudaDeviceSynchronize();

	cudaMemcpy2D(
				*pLpvBits,						// Destination 
				src_pitch,			
				cudaLpvBits,					// Source
				dest_pitch,		
				WIDTH * 4 * sizeof(BYTE),		
				HEIGHT, 
				cudaMemcpyDeviceToHost);


	cudaFree(cudaLpvBits);
}
