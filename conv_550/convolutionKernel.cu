#include "convolution.h"

#define KERNAL_RAD 8
#define KERNEL_LENGTH 17


__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setKernel(float *h_Kernel)
{
	cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

__global__ void convKernelFullNaiveSepKernel(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int loc = row * imageW + col;

    float s = 0;
    float t = 0;

    for (int i = -kernelR; i <= kernelR; i++)
        for (int j = -kernelR; j <= kernelR; j++)
        {
            t = 0;

            if (row  + i >= 0 && row  + i < imageH && col  + j >= 0 && col  + j < imageW )
                t = d_Input[loc + i * imageW + j];

            s += t * d_Kernel[kernelR - i] * d_Kernel[kernelR - j];
        }
        d_Output[loc] = s;
}

__global__ void convKernelFullNaive(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int loc = row * imageW + col;

    float s = 0;
    float t = 0;

    for (int i = -kernelR; i <= kernelR; i++)
        for (int j = -kernelR; j <= kernelR; j++)
        {
            t = 0;

            if (row  + i >= 0 && row  + i < imageH && col  + j >= 0 && col  + j < imageW )
                t = d_Input[loc + i * imageW + j];

            s += t * d_Kernel[(kernelR - i) * (kernelR + kernelR + 1) + kernelR - j];
        }
        d_Output[loc] = s;
}

__global__ void convKernelSeparableRowNaive(float* d_Input, float* d_Output, int  imageW, int imageH, int kernelR)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int loc = row * imageW + col;

    float s = 0;
    float t = 0;

    for (int j = -kernelR; j <= kernelR; j++)
    {
        t = 0;
        if (col  + j >= 0 && col  + j < imageW )
            t = d_Input[loc + j];
        s += t * c_Kernel[kernelR - j] ;
     }
     d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnNaive(float* d_Input, float* d_Output, int  imageW, int imageH, int kernelR)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int loc = row * imageW + col;

    float s = 0;
    float t = 0;

    for (int i = -kernelR; i <= kernelR; i++)
    {
        t = 0;
        if (row  + i >= 0 && row  + i < imageH)
            t = d_Input[loc + i * imageW];
        s += t * c_Kernel[kernelR - i];
    }
    d_Output[loc] = s;
}

__global__ void convKernelSeparableRowShared(float* d_Input, float* d_Output, int  imageW, int imageH, int kernelR)
{
    // Data cache
    __shared__ float data[ CACHE_H * (CACHE_W + KERNAL_RAD * 2)];

    // Initializations
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int loc = row * imageW + col;
    const int shift = threadIdx.y * (CACHE_W + kernelR * 2);

    // left apron
    data[threadIdx.x + shift] = (col - kernelR >= 0) ? d_Input[ loc - kernelR] : 0;
    // right apron
    data[threadIdx.x + blockDim.x + shift] = (col + kernelR < imageW) ? d_Input[loc + kernelR] : 0;

    //compute and store results
    __syncthreads();

    // convolution
    float s = 0;
	for (int j = -kernelR; j <= kernelR; j++)
	{
		s += data[kernelR + threadIdx.x + j + shift] * c_Kernel[kernelR - j];
	}

    d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnShared(float* d_Input, float* d_Output, int  imageW, int imageH, int kernelR)
{
    // Data cache
    __shared__ float data[CACHE_W * (CACHE_H + KERNAL_RAD * 2)];

    // Initializations
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int loc = row * imageW + col;
    const int shift = threadIdx.y * CACHE_W;

    // top apron
    data[threadIdx.x + shift] = (row - kernelR >=0) ? d_Input[ loc - imageW * kernelR] : 0;
    // bottom apron
    data[threadIdx.x + shift + blockDim.y * CACHE_W] = (row + kernelR < imageH) ? d_Input[loc + imageW * kernelR] : 0;

    //Compute and store results
    __syncthreads();

    // convolution
    float s = 0;

    for (int i = -kernelR; i <= kernelR; i++)
        s += data[threadIdx.x + (threadIdx.y + i + kernelR) * CACHE_W] * c_Kernel[kernelR - i];

    d_Output[loc] = s;
}

__global__ void convKernelSeparableRowSharedUnroll(float* d_Input, float* d_Output, int  imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int row = blockDim.y * blockIdx.y + threadIdx.y;
	const int col = blockDim.x * blockIdx.x + threadIdx.x;
	const int loc = row * imageW + col;
	const int shift = threadIdx.y * (CACHE_W + KERNAL_RAD * 2);

	// left apron
	data[threadIdx.x + shift] = (col - KERNAL_RAD >= 0) ? d_Input[loc - KERNAL_RAD] : 0;
	// right apron
	data[threadIdx.x + blockDim.x + shift] = (col + KERNAL_RAD < imageW) ? d_Input[loc + KERNAL_RAD] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;

#pragma unroll

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[KERNAL_RAD + threadIdx.x + j + shift] * c_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnSharedUnroll(float* d_Input, float* d_Output, int  imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_W * (CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int row = blockDim.y * blockIdx.y + threadIdx.y;
	const int col = blockDim.x * blockIdx.x + threadIdx.x;
	const int loc = row * imageW + col;
	const int shift = threadIdx.y * CACHE_W;

	// top apron
	data[threadIdx.x + shift] = (row - KERNAL_RAD >= 0) ? d_Input[loc - imageW * KERNAL_RAD] : 0;
	// bottom apron
	data[threadIdx.x + shift + blockDim.y * CACHE_W] = (row + KERNAL_RAD < imageH) ? d_Input[loc + imageW * KERNAL_RAD] : 0;

	//Compute and store results
	__syncthreads();

	// convolution
	float s = 0;

#pragma unroll

	for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++) {
		s += data[threadIdx.x + (threadIdx.y + i + KERNAL_RAD) * CACHE_W] * c_Kernel[KERNAL_RAD - i];
	}

	d_Output[loc] = s;
}


__global__ void convKernelSeparableRowSharedMul(float* d_Input, float* d_Output, int  imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, (CACHE_W + KERNAL_RAD * 2));

	// left apron
	data[threadIdx.x + shift] = (col - KERNAL_RAD >= 0) ? d_Input[loc - KERNAL_RAD] : 0;
	// right apron
	data[threadIdx.x + blockDim.x + shift] = (col + KERNAL_RAD < imageW) ? d_Input[loc + KERNAL_RAD] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;

#pragma unroll

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[KERNAL_RAD + threadIdx.x + j + shift] * c_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnSharedMul(float* d_Input, float* d_Output, int  imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_W * (CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, CACHE_W);

	// top apron
	data[threadIdx.x + shift] = (row - KERNAL_RAD >= 0) ? d_Input[loc - __mul24(imageW, KERNAL_RAD)] : 0;
	// bottom apron
	data[threadIdx.x + shift + __mul24(blockDim.y, CACHE_W)] = (row + KERNAL_RAD < imageH) ? d_Input[loc + __mul24(imageW, KERNAL_RAD)] : 0;

	//Compute and store results
	__syncthreads();

	// convolution
	float s = 0;

#pragma unroll

	for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++)
	{
		s += data[threadIdx.x + __mul24((threadIdx.y + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
	}

	d_Output[loc] = s;
}

#define TILES 2

__global__ void convKernelSeparableRowSharedTile(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (TILES * CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int blockDimX =__mul24(TILES, blockDim.x);
	const int threadX = __mul24(TILES, threadIdx.x);
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDimX, blockIdx.x) + threadX;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, (TILES * CACHE_W + KERNAL_RAD * 2));

	// left apron
	data[threadX + shift] = (col - KERNAL_RAD >= 0) ? d_Input[loc - KERNAL_RAD] : 0;
	data[threadX + shift + 1] = (col - KERNAL_RAD + 1 >= 0) ? d_Input[loc - KERNAL_RAD + 1] : 0;
	// right apron
	data[threadX + blockDimX + shift] = (col + KERNAL_RAD < imageW) ? d_Input[loc + KERNAL_RAD] : 0;
	data[threadX + blockDimX + shift + 1] = (col + KERNAL_RAD + 1 < imageW) ? d_Input[loc + KERNAL_RAD + 1] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	float s1 = 0;

#pragma unroll

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[KERNAL_RAD + threadX + j + shift] * c_Kernel[KERNAL_RAD - j];
		s1 += data[KERNAL_RAD + threadX + j + shift + 1] * c_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
	d_Output[loc + 1] = s1;
}

__global__ void convKernelSeparableColumnSharedTile(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_W * (TILES * CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int threadY = __mul24(TILES, threadIdx.y);
	const int blockDimY = __mul24(TILES, blockDim.y);
	const int row = __mul24(blockDimY, blockIdx.y) + threadY;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadY, CACHE_W);

	// top apron
	data[threadIdx.x + shift] = (row - KERNAL_RAD >= 0) ? d_Input[loc - __mul24(imageW, KERNAL_RAD)] : 0;
	data[threadIdx.x + shift + CACHE_W] = (row + 1 - KERNAL_RAD >= 0) ? d_Input[loc + imageW - __mul24(imageW, KERNAL_RAD)] : 0;
	// bottom apron
	data[threadIdx.x + shift + __mul24(blockDimY, CACHE_W)] = (row + KERNAL_RAD < imageH) ? d_Input[loc + __mul24(imageW, KERNAL_RAD)] : 0;
	data[threadIdx.x + shift + __mul24(blockDimY, CACHE_W) + CACHE_W] = (row + 1 + KERNAL_RAD < imageH) ? d_Input[loc + imageW + __mul24(imageW, KERNAL_RAD)] : 0;

	//Compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	float s1 = 0;

#pragma unroll

	for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++)
	{
		s += data[threadIdx.x + __mul24((threadY + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
		s1 += data[threadIdx.x + __mul24((threadY + 1 + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
	}

	d_Output[loc] = s;
	d_Output[loc + imageW] = s1;
}

__global__ void convKernelSeparableRowSharedTile2(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (TILES * CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int blockDimX = __mul24(TILES, blockDim.x);
	const int threadX = __mul24(TILES, threadIdx.x);
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDimX, blockIdx.x) + threadX;
	const int loc = __mul24(row, imageW) + col;
	const int shiftBase = TILES * CACHE_W + KERNAL_RAD * 2;
	const int shift = __mul24(threadX, shiftBase);

	// left apron
	data[threadIdx.y + shift] = (col - KERNAL_RAD >= 0) ? d_Input[loc - KERNAL_RAD] : 0;
	data[threadIdx.y + shift + shiftBase] = (col - KERNAL_RAD + 1 >= 0) ? d_Input[loc - KERNAL_RAD + 1] : 0;
	// right apron
	data[threadIdx.y + shift + blockDimX] = (col + KERNAL_RAD < imageW) ? d_Input[loc + KERNAL_RAD] : 0;
	data[threadIdx.y + shift + blockDimX + shiftBase] = (col + KERNAL_RAD + 1 < imageW) ? d_Input[loc + KERNAL_RAD + 1] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	float s1 = 0;

#pragma unroll

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[KERNAL_RAD + threadIdx.y + j + shift] * c_Kernel[KERNAL_RAD - j];
		s1 += data[KERNAL_RAD + threadX + j + shift + 1] * c_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
	d_Output[loc + 1] = s1;
}

// coalescence 

__global__ void convKernelSeparableRowSharedTileCoales(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (TILES * CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int blockDimX = __mul24(TILES, blockDim.x);
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDimX, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, (TILES * CACHE_W + KERNAL_RAD * 2));

	// left apron
	data[threadIdx.x + shift] = (col - KERNAL_RAD >= 0) ? d_Input[loc - KERNAL_RAD] : 0;
	//data[threadIdx.x + blockDim.x + shift] = (col - KERNAL_RAD + blockDim.x >= 0) ? d_Input[loc - KERNAL_RAD + blockDim.x] : 0;
	data[threadIdx.x + blockDim.x + shift] = d_Input[loc - KERNAL_RAD + blockDim.x];
	// right apron
	data[threadIdx.x + blockDimX + shift] = (col + KERNAL_RAD < imageW) ? d_Input[loc + KERNAL_RAD] : 0;
	data[threadIdx.x + blockDimX + blockDim.x + shift] = (col + KERNAL_RAD + blockDim.x < imageW) ? d_Input[loc + KERNAL_RAD + blockDim.x] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	float s1 = 0;

	const int dataInitialShift = KERNAL_RAD + threadIdx.x + shift;

#pragma unroll

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[dataInitialShift + j] * c_Kernel[KERNAL_RAD - j];
		s1 += data[dataInitialShift + blockDim.x + j] * c_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
	d_Output[loc + blockDim.x] = s1;
}

__global__ void convKernelSeparableColumnSharedTileCoales(float* d_Input, float* d_Output, int  imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_W * (TILES * CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int threadY = __mul24(TILES, threadIdx.y);
	const int blockDimY = __mul24(TILES, blockDim.y);
	const int row = __mul24(blockDimY, blockIdx.y) + threadY;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadY, CACHE_W);

	// top apron
	data[threadIdx.x + shift] = (row - KERNAL_RAD >= 0) ? d_Input[loc - __mul24(imageW, KERNAL_RAD)] : 0;
	data[threadIdx.x + shift + CACHE_W] = (row + 1 - KERNAL_RAD >= 0) ? d_Input[loc + imageW - __mul24(imageW, KERNAL_RAD)] : 0;
	// bottom apron
	data[threadIdx.x + shift + __mul24(blockDimY, CACHE_W)] = (row + KERNAL_RAD < imageH) ? d_Input[loc + __mul24(imageW, KERNAL_RAD)] : 0;
	data[threadIdx.x + shift + __mul24(blockDimY, CACHE_W) + CACHE_W] = (row + 1 + KERNAL_RAD < imageH) ? d_Input[loc + imageW + __mul24(imageW, KERNAL_RAD)] : 0;

	//Compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	float s1 = 0;

#pragma unroll

	for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++)
	{
		s += data[threadIdx.x + __mul24((threadY + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
		s1 += data[threadIdx.x + __mul24((threadY + 1 + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
	}

	d_Output[loc] = s;
	d_Output[loc + imageW] = s1;
}





__global__ void convKernelSeparableRowSharedMultiTile(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_H * (TILES * CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int blockDimX = __mul24(TILES, blockDim.x);
	const int threadX = __mul24(TILES, threadIdx.x);
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDimX, blockIdx.x) + threadX;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, (TILES * CACHE_W + KERNAL_RAD * 2));

	// left apron
#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		data[threadX + shift + t] = (col - KERNAL_RAD + t >= 0) ? d_Input[loc - KERNAL_RAD + t] : 0;
	}

	// right apron
#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		data[threadX + blockDimX + shift + t] = (col + KERNAL_RAD + t < imageW) ? d_Input[loc + KERNAL_RAD + t] : 0;
	}

	//compute and store results
	__syncthreads();

#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		// convolution
		float s = 0;

#pragma unroll
		for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
		{
			s += data[KERNAL_RAD + threadX + t + j + shift] * c_Kernel[KERNAL_RAD - j];
		}

		d_Output[loc + t] = s;
	}
}

__global__ void convKernelSeparableColumnSharedMultiTile(float* d_Input, float* d_Output, int imageW, int imageH)
{
	// Data cache
	__shared__ float data[CACHE_W * (TILES * CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int threadY = __mul24(TILES, threadIdx.y);
	const int blockDimY = __mul24(TILES, blockDim.y);
	const int row = __mul24(blockDimY, blockIdx.y) + threadY;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadY, CACHE_W);

	// top apron
#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		data[threadIdx.x + shift + __mul24(t, CACHE_W)] = (row + t - KERNAL_RAD >= 0) ? d_Input[loc + __mul24(t, imageW) - __mul24(imageW, KERNAL_RAD)] : 0;
	}

	// bottom apron
#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		data[threadIdx.x + shift + __mul24(blockDimY + t, CACHE_W)] =
			(row + t + KERNAL_RAD < imageH) ?
			d_Input[loc + __mul24(t, imageW) + __mul24(imageW, KERNAL_RAD)]
			: 0;
	}

	//Compute and store results
	__syncthreads();

#pragma unroll
	for (int t = 0; t < TILES; t++)
	{
		// convolution
		float s = 0;

#pragma unroll
		for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++)
		{
			s += data[threadIdx.x + __mul24((threadY + t + i + KERNAL_RAD), CACHE_W)] * c_Kernel[KERNAL_RAD - i];
		}

		d_Output[loc + __mul24(t, imageW)] = s;
	}
}


void convolutionSeparableColumnNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableColumnNaive<< <gridSize, blockSize>> >(d_Input,d_Output, imageW, imageH, kernelR);
}

void convolutionSeparableRowNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableRowNaive<< <gridSize, blockSize>> >(d_Input,d_Output, imageW, imageH, kernelR);
}

void convolutionFullNaiveSepKernel(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelFullNaiveSepKernel<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionFullNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelFullNaive<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableColumnShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW, int imageH, int kernelR)
{
    convKernelSeparableColumnShared<< <gridSize, blockSize>> >(d_Input,d_Output, imageW, imageH, kernelR);
}

void convolutionSeparableRowShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int imageW, int imageH, int kernelR)
{
    convKernelSeparableRowShared<< <gridSize, blockSize>> >(d_Input,d_Output, imageW, imageH, kernelR);
}

void convolutionSeparableRowSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int imageW, int imageH)
{
	convKernelSeparableRowSharedUnroll << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableColumnSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int imageW, int imageH)
{
	convKernelSeparableColumnSharedUnroll << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableColumnSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH)
{
	convKernelSeparableColumnSharedMul << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableRowSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int imageW, int imageH)
{
	convKernelSeparableRowSharedMul << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableColumnSharedTile(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH)
{
	convKernelSeparableColumnSharedTile << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
	//convKernelSeparableColumnSharedMultiTile << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableRowSharedTile(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int imageW, int imageH)
{
	convKernelSeparableRowSharedTile << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
	//convKernelSeparableRowSharedMultiTile << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableColumnSharedTileCoales(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH)
{
	convKernelSeparableColumnSharedTileCoales << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}

void convolutionSeparableRowSharedTileCoales(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int imageW, int imageH)
{
	convKernelSeparableRowSharedTileCoales << <gridSize, blockSize >> >(d_Input, d_Output, imageW, imageH);
}