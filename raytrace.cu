#include "raytrace.cuh"


#define HEIGHT 1024
#define WIDTH 1024

__global__ void cuda_Draw(BYTE* pLpvBits)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = (i * WIDTH * j) * 4;
	if (offset >= 0 && offset < WIDTH * HEIGHT * 4 - 4)
	{
		pLpvBits[offset + 0] = (int)GetBValue(blockIdx.x * blockIdx.y);
		pLpvBits[offset + 1] = (int)GetGValue(blockDim.x * blockDim.y);
		pLpvBits[offset + 2] = (int)GetRValue(threadIdx.x * threadIdx.y);
		pLpvBits[offset + 3] = 255;
	}

}


void Draw_Caller(BYTE ** pLpvBits)
{
	int N = 1024;
	dim3 threadsPB(16,16);
	dim3 numB(N/threadsPB.x, N / threadsPB.y);
	

	BYTE *cudaLpvBits;
	cudaMalloc(&cudaLpvBits, 4*WIDTH*HEIGHT);

	cudaMemcpy(&cudaLpvBits, (*pLpvBits),4*WIDTH*HEIGHT, cudaMemcpyHostToDevice);

	cuda_Draw<<<numB, threadsPB>>>(cudaLpvBits);
	
	cudaMemcpy((*pLpvBits), &cudaLpvBits,4*WIDTH*HEIGHT, cudaMemcpyDeviceToHost);
	}
