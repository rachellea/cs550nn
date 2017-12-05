#include "convolution.h"

#define KERNAL_RAD 8

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

__global__ void convKernelSeparableRowNaive(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
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
        s += t * d_Kernel[kernelR - j] ;
     }
     d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnNaive(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
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
        s += t * d_Kernel[kernelR - i];
    }
    d_Output[loc] = s;
}

__global__ void convKernelSeparableRowShared(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
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
		s += data[kernelR + threadIdx.x + j + shift] * d_Kernel[kernelR - j];
	}

    d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnShared(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
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
        s += data[threadIdx.x + (threadIdx.y + i + kernelR) * CACHE_W] * d_Kernel[kernelR - i];

    d_Output[loc] = s;
}

__global__ void convKernelSeparableRowSharedUnroll(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH)
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

	/*s = data[KERNAL_RAD + threadIdx.x - 8 + shift] * d_Kernel[16] +
	data[KERNAL_RAD + threadIdx.x - 7 + shift] * d_Kernel[15] +
	data[KERNAL_RAD + threadIdx.x - 6 + shift] * d_Kernel[14] +
	data[KERNAL_RAD + threadIdx.x - 5 + shift] * d_Kernel[13] +
	data[KERNAL_RAD + threadIdx.x - 4 + shift] * d_Kernel[12] +
	data[KERNAL_RAD + threadIdx.x - 3 + shift] * d_Kernel[11] +
	data[KERNAL_RAD + threadIdx.x - 2 + shift] * d_Kernel[10] +
	data[KERNAL_RAD + threadIdx.x - 1 + shift] * d_Kernel[9] +
	data[KERNAL_RAD + threadIdx.x + 0 + shift] * d_Kernel[8] +
	data[KERNAL_RAD + threadIdx.x + 1 + shift] * d_Kernel[7] +
	data[KERNAL_RAD + threadIdx.x + 2 + shift] * d_Kernel[6] +
	data[KERNAL_RAD + threadIdx.x + 3 + shift] * d_Kernel[5] +
	data[KERNAL_RAD + threadIdx.x + 4 + shift] * d_Kernel[4] +
	data[KERNAL_RAD + threadIdx.x + 5 + shift] * d_Kernel[3] +
	data[KERNAL_RAD + threadIdx.x + 6 + shift] * d_Kernel[2] +
	data[KERNAL_RAD + threadIdx.x + 7 + shift] * d_Kernel[1] +
	data[KERNAL_RAD + threadIdx.x + 8 + shift] * d_Kernel[0];*/

	for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
	{
		s += data[KERNAL_RAD + threadIdx.x + j + shift] * d_Kernel[KERNAL_RAD - j];
	}

	d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnSharedUnroll(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH)
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
	/* s = data[threadIdx.x + (threadIdx.y + 0) * CACHE_W] * d_Kernel[16] +
		data[threadIdx.x + (threadIdx.y + 1) * CACHE_W] * d_Kernel[15] +
		data[threadIdx.x + (threadIdx.y + 2) * CACHE_W] * d_Kernel[14] +
		data[threadIdx.x + (threadIdx.y + 3) * CACHE_W] * d_Kernel[13] +
		data[threadIdx.x + (threadIdx.y + 4) * CACHE_W] * d_Kernel[12] +
		data[threadIdx.x + (threadIdx.y + 5) * CACHE_W] * d_Kernel[11] +
		data[threadIdx.x + (threadIdx.y + 6) * CACHE_W] * d_Kernel[10] +
		data[threadIdx.x + (threadIdx.y + 7) * CACHE_W] * d_Kernel[9] +
		data[threadIdx.x + (threadIdx.y + 8) * CACHE_W] * d_Kernel[8] +
		data[threadIdx.x + (threadIdx.y + 9) * CACHE_W] * d_Kernel[7] +
		data[threadIdx.x + (threadIdx.y + 10) * CACHE_W] * d_Kernel[6] +
		data[threadIdx.x + (threadIdx.y + 11) * CACHE_W] * d_Kernel[5] +
		data[threadIdx.x + (threadIdx.y + 12) * CACHE_W] * d_Kernel[4] +
		data[threadIdx.x + (threadIdx.y + 13) * CACHE_W] * d_Kernel[3] +
		data[threadIdx.x + (threadIdx.y + 14) * CACHE_W] * d_Kernel[2] +
		data[threadIdx.x + (threadIdx.y + 15) * CACHE_W] * d_Kernel[1] +
		data[threadIdx.x + (threadIdx.y + 16) * CACHE_W] * d_Kernel[0];*/

	for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++) {
		s += data[threadIdx.x + (threadIdx.y + i + KERNAL_RAD) * CACHE_W] * d_Kernel[KERNAL_RAD - i];
	}

	d_Output[loc] = s;
}


__global__ void convKernelSeparableRowSharedMul(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
	// Data cache
	__shared__ float data[CACHE_H * (CACHE_W + KERNAL_RAD * 2)];

	// Initializations
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, (CACHE_W + kernelR * 2));

	// left apron
	data[threadIdx.x + shift] = (col - kernelR >= 0) ? d_Input[loc - kernelR] : 0;
	// right apron
	data[threadIdx.x + blockDim.x + shift] = (col + kernelR < imageW) ? d_Input[loc + kernelR] : 0;

	//compute and store results
	__syncthreads();

	// convolution
	float s = 0;
	for (int j = -kernelR; j <= kernelR; j++)
	{
		s += data[kernelR + threadIdx.x + j + shift] * d_Kernel[kernelR - j];
	}

	d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnSharedMul(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
	// Data cache
	__shared__ float data[CACHE_W * (CACHE_H + KERNAL_RAD * 2)];

	// Initializations
	const int row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int loc = __mul24(row, imageW) + col;
	const int shift = __mul24(threadIdx.y, CACHE_W);

	// top apron
	data[threadIdx.x + shift] = (row - kernelR >= 0) ? d_Input[loc - __mul24(imageW, kernelR)] : 0;
	// bottom apron
	data[threadIdx.x + shift + __mul24(blockDim.y, CACHE_W)] = (row + kernelR < imageH) ? d_Input[loc + __mul24(imageW, kernelR)] : 0;

	//Compute and store results
	__syncthreads();

	// convolution
	float s = 0;

	for (int i = -kernelR; i <= kernelR; i++)
		s += data[threadIdx.x + __mul24((threadIdx.y + i + kernelR), CACHE_W)] * d_Kernel[kernelR - i];

	d_Output[loc] = s;
}



void convolutionSeparableColumnNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableColumnNaive<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableRowNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableRowNaive<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionFullNaiveSepKernel(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelFullNaiveSepKernel<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionFullNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelFullNaive<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableColumnShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW, int imageH, int kernelR)
{
    convKernelSeparableColumnShared<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableRowShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int imageW, int imageH, int kernelR)
{
    convKernelSeparableRowShared<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableRowSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, float* d_Kernel, int imageW, int imageH)
{
	convKernelSeparableRowSharedUnroll << <gridSize, blockSize >> >(d_Input, d_Output, d_Kernel, imageW, imageH);
}

void convolutionSeparableColumnSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, float* d_Kernel, int imageW, int imageH)
{
	convKernelSeparableColumnSharedUnroll << <gridSize, blockSize >> >(d_Input, d_Output, d_Kernel, imageW, imageH);
}

void convolutionSeparableColumnSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
	convKernelSeparableColumnShared << <gridSize, blockSize >> >(d_Input, d_Output, d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableRowSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, float* d_Kernel, int imageW, int imageH, int kernelR)
{
	convKernelSeparableRowShared << <gridSize, blockSize >> >(d_Input, d_Output, d_Kernel, imageW, imageH, kernelR);
}
