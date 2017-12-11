/*
 * Original source from NVIDIA_CUDA 8.0 Samples
 * Modified for the pupposes of running experiments for different
 * convolution kernels implementations in terms of 550 final project
 */

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"
#include "convolutionMethods.h"
#include <iostream>

using namespace std;


// helper functions

bool imgL2error(float* img1, float* img2, const int & imageW, const int & imageH);


int main(int argc, char **argv) {

    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPUSep,
    *h_OutputGPUSepNaive,
    *h_OutputGPUFullNaive,
    *h_KernelFull,
    *h_OutputGPU;

    float
    *d_Input,
    *d_Output,
    *d_Buffer,
    *d_Kernel;


    const int imageW = 1024;
    const int imageH = 1024;
    const int iterations = 1;

    StopWatchInterface *hTimer = NULL;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    //printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    //printf("Allocating and initializing host arrays...\n");
    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_KernelFull= (float *)malloc(KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPUSep = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPUSepNaive = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPUFullNaive = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    srand(200);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
        for (unsigned int j = 0; j < KERNEL_LENGTH; j++)
        {
            h_KernelFull[i*KERNEL_LENGTH + j] = h_Kernel[i] * h_Kernel[j];
        }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    cout << "Image size: " << imageW << " " << imageH << endl;
    cout << "Kernel length: " << KERNEL_LENGTH << endl;

//    // Benchmark: NVIDIA convolution on CPU
//    seperableConvolutionCPU(h_Input,h_OutputCPU,h_Buffer,h_Kernel,imageW,imageH,iterations,hTimer);

//    // Benchmark: NVIDIA convolution on GPU
//    seperableConvolutionGPU(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 1: Naive full kernel approach
//    straigtforwardKernel(d_Input,d_Output,d_Buffer,d_Kernel,h_KernelFull,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 2: Naive separable kernel approach
    separableKernel(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 3: Separable kernel shared memory approach
//    separableSharedKernel(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 4: Separable kernel shared memory approach with loop unrolling
//    seperableSharedKernelUnroll(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 5: Separable kernel shared memory approach with loop unrolling and tiling
//    seperableSharedKernelTile(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

//    // Algorithm 6: Separable kernel shared memory approach with loop unrolling, tiling and coales
//    seperableSharedKernelTileCoales(d_Input,d_Output,d_Buffer,d_Kernel,h_Kernel,h_Input,h_OutputGPU,imageW,imageH,iterations,hTimer);
//    imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH);

    // free memory
    free(h_OutputGPUSep);
    free(h_OutputGPUSepNaive);
    free(h_OutputGPUFullNaive);
    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);
    free(h_KernelFull);

    sdkDeleteTimer(&hTimer);
	
	return 0;
}

bool imgL2error(float* img1, float* img2, const int & imageW, const int & imageH)
{
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        delta += (img1[i] - img2[i]) * (img1[i] - img2[i]);
        sum   += img2[i] * img2[i];
    }

    double L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);

    if (L2norm > 1e-6)
        return 0;
    return 1;
}
