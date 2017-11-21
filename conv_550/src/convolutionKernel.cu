#include "convolution.h"

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
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int loc = row * imageW + col;
    int shift = threadIdx.y * (CACHE_W + kernelR * 2);

    // left apron
    data[threadIdx.x + shift] = (col - kernelR >= 0) ? d_Input[ loc - kernelR] : 0;
    // right apron
    data[threadIdx.x + blockDim.x + shift] = (col + kernelR < imageW) ? d_Input[loc + kernelR] : 0;

    //compute and store results
    __syncthreads();

    // convolution
    float s = 0;
    for (int j = -kernelR; j <= kernelR; j++)
        s += data[kernelR + threadIdx.x + j + shift] * d_Kernel[kernelR - j];

    d_Output[loc] = s;
}

__global__ void convKernelSeparableColumnShared(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
    // Data cache
    __shared__ float data[CACHE_W * (CACHE_H + KERNAL_RAD * 2)];

    // Initializations
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int loc = row * imageW + col;
    int shift = threadIdx.y * CACHE_W;

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

void convolutionSeparableColumnShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableColumnShared<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

void convolutionSeparableRowShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH,int kernelR)
{
    convKernelSeparableRowShared<< <gridSize, blockSize>> >(d_Input,d_Output,d_Kernel, imageW, imageH, kernelR);
}

