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
#include "convolution.h"

#include <iostream>

using namespace std;


// helper functions
int isDividedUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
bool imgL2error(float* img1, float* img2, const int & imageW, const int & imageH);


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputGPUSep;

    float
    *d_Input,
    *d_Output,
    *d_Buffer;


    const int imageW =  3072;
    const int imageH =  3072;
    const int iterations = 10;

    StopWatchInterface *hTimer = NULL;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    //printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    //printf("Allocating and initializing host arrays...\n");
    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPUSep = (float *)malloc(imageW * imageH * sizeof(float));

    srand(200);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }


    // Separable Convolution on GPU
    {
        printf("Running GPU separable convolution (%u identical iterations).\n", iterations);
        //printf("Allocating and initializing CUDA arrays...\n");
        checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float)));

        setConvolutionKernel(h_Kernel);
        checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

        //printf("Running GPU separable convolution (%u identical iterations)...\n\n", iterations);

        for (int i = -1; i < iterations; i++)
        {
            //i == -1 -- warmup iteration
            if (i == 0)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                sdkResetTimer(&hTimer);
                sdkStartTimer(&hTimer);
            }

            convolutionRowsGPU(
                d_Buffer,
                d_Input,
                imageW,
                imageH
            );

            convolutionColumnsGPU(
                d_Output,
                d_Buffer,
                imageW,
                imageH
            );
        }

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double gpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
        printf("convolutionSeparableGPU_NVIDIA, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
               (1.0e-6 * (double)(imageW * imageH)/ gpuTime), gpuTime, (imageW * imageH));

        //printf("\nReading back GPU results...\n\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPUSep, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_Buffer));
        checkCudaErrors(cudaFree(d_Output));
        checkCudaErrors(cudaFree(d_Input));
    }

//    cout << imgL2error(h_OutputGPUSepShared, h_OutputCPU, imageW, imageH) << endl;

    // free memory
    free(h_OutputGPUSep);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    exit(EXIT_SUCCESS);
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
