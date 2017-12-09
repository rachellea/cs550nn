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

// main methods
int profile(int argc, char **argv);
int runAll(int argc, char **argv);

// helper functions
int isDividedUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
bool imgL2error(float* img1, float* img2, const int & imageW, const int & imageH);

// convolution methods
void seperableConvolutionCPU(
	float *h_Input,
	float *h_OutputCPU,
	float *h_Buffer,
	float *h_Kernel,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void seperableConvolutionGPU(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void straigtforwardKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void separableKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void separableSharedKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void seperableSharedKernelUnroll(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void seperableSharedKernelMul(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void seperableSharedKernelTile(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

void seperableSharedKernelTileCoales(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer);

int main(int argc, char **argv) {

	profile(argc, argv);

	//runAll(argc, argv);

	exit(EXIT_SUCCESS);
}


////////////////////////////////////////////////////////////////////////////////
// Profiling KernelSize vs throughput
////////////////////////////////////////////////////////////////////////////////
int profile(int argc, char **argv)
{

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


	const int imageW = 3072;
	const int imageH = 3072;
	const int iterations = 10;

	StopWatchInterface *hTimer = NULL;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, (const char **)argv);

	sdkCreateTimer(&hTimer);

	h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
	h_KernelFull = (float *)malloc(KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));
	h_Input = (float *)malloc(imageW * imageH * sizeof(float));
	h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
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

	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	// run cpu to get standard result
	seperableConvolutionCPU(
		h_Input,
		h_OutputCPU,
		h_Buffer,
		h_Kernel,
		imageW,
		imageH,
		iterations,
		hTimer);

	seperableConvolutionGPU(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	// Simple Kernel straigtforward separable approach
	separableKernel(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	// Shared Kernel separable approach
	separableSharedKernel(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	seperableSharedKernelTile(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	seperableSharedKernelTileCoales(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Run different algorithms
////////////////////////////////////////////////////////////////////////////////
int runAll(int argc, char **argv)
{

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


	const int imageW = 3072;
	const int imageH = 3072;
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

	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	seperableConvolutionCPU(
		h_Input,
		h_OutputCPU,
		h_Buffer,
		h_Kernel,
		imageW,
		imageH,
		iterations,
		hTimer);

    // Separable Convolution on GPU
	seperableConvolutionGPU(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

    // Simple Kernel straigtforward full approach
    straigtforwardKernel(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_KernelFull,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
        hTimer);

    // Simple Kernel straigtforward separable approach
	separableKernel(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

    // Shared Kernel separable approach
	separableSharedKernel(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	seperableSharedKernelUnroll(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

	cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	seperableSharedKernelTile(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

    cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

	seperableSharedKernelTileCoales(
		d_Input,
		d_Output,
		d_Buffer,
		d_Kernel,
		h_Kernel,
		h_Input,
		h_OutputGPU,
		imageW,
		imageH,
		iterations,
		hTimer);

 /*   for (int i = 0; i < imageW; ++i)
    {
        for(int j = 0; j < imageH; ++j)
        {
            cout << h_OutputCPU[i* imageW + j] << " ";
        }
        cout << endl;
    }
   cout << endl << endl;
    for (int i = 0; i < imageW; ++i)
    {
        for(int j = 0; j < imageH; ++j)
        {
            cout << h_OutputGPU[i* imageW + j] << " ";
        }
        cout << endl;
    }*/

    //cout << imgL2error(h_OutputGPU, h_OutputCPU, imageW, imageH) << endl;

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

void seperableConvolutionCPU(
	float *h_Input,
	float *h_OutputCPU,
	float *h_Buffer,
	float *h_Kernel,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	// Separable convolution on CPU
	printf("Running CPU separable convolution (%u identical iterations).\n", iterations);

	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionRowCPU(
			h_Buffer,
			h_Input,
			h_Kernel,
			imageW,
			imageH,
			KERNEL_RADIUS
			);


		convolutionColumnCPU(
			h_OutputCPU,
			h_Buffer,
			h_Kernel,
			imageW,
			imageH,
			KERNEL_RADIUS
			);
	}

	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableCPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));
}

void seperableConvolutionGPU(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	printf("Running GPU separable convolution (%u identical iterations).\n", iterations);

	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	setConvolutionKernel(h_Kernel);
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

	//printf("Running GPU separable convolution (%u identical iterations)...\n\n", iterations);

	for (int i = -1; i < iterations; i++)
	{
		
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
		(1.0e-6 * (double)(imageW * imageH) / gpuTime), gpuTime, (imageW * imageH));

	//printf("\nReading back GPU results...\n\n");
	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Buffer));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void straigtforwardKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_KernelFull,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_KernelFull, KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

	dim3 gridSize(isDividedUp(imageH, kBlockDimX), isDividedUp(imageW, kBlockDimY));
	dim3 blockSize(kBlockDimX, kBlockDimY);

	printf("Running GPU naive full convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionFullNaive(
			gridSize,
			blockSize,
			d_Input,
			d_Output,
			d_Kernel,
			imageW,
			imageH,
			KERNEL_RADIUS);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionFullNaiveGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void separableKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

	setKernel(h_Kernel);

	dim3 gridSize(isDividedUp(imageH, 16), isDividedUp(imageW, 16));
	dim3 blockSize(16, 16);

	printf("Running GPU naive separable convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionSeparableRowNaive(gridSize, blockSize, d_Input, d_Buffer, imageW, imageH, KERNEL_RADIUS);
		convolutionSeparableColumnNaive(gridSize, blockSize, d_Buffer, d_Output, imageW, imageH, KERNEL_RADIUS);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableNaiveGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void separableSharedKernel(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	setKernel(h_Kernel);

	dim3 gridSize(isDividedUp(imageH, 16), isDividedUp(imageW, 16));
	dim3 blockSize(16, 16);

	printf("Running GPU shared separable convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionSeparableRowShared(gridSize, blockSize, d_Input, d_Buffer, imageW, imageH, KERNEL_RADIUS);
		checkCudaErrors(cudaDeviceSynchronize());

		convolutionSeparableColumnShared(gridSize, blockSize, d_Buffer, d_Output, imageW, imageH, KERNEL_RADIUS);
		checkCudaErrors(cudaDeviceSynchronize());

	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableSharedGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void seperableSharedKernelUnroll(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
		checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

		setKernel(h_Kernel);

		dim3 gridSize(isDividedUp(imageH, 16), isDividedUp(imageW, 16));
		dim3 blockSize(16, 16);

		printf("Running GPU naive separable convolution (%u identical iterations).\n", iterations);
		for (int i = -1; i < iterations; i++)
		{
			
			if (i == 0)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);
			}

			convolutionSeparableRowSharedUnroll(gridSize, blockSize, d_Input, d_Buffer, imageW, imageH);
			convolutionSeparableColumnSharedUnroll(gridSize, blockSize, d_Buffer, d_Output, imageW, imageH);
		}

		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&hTimer);
		double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
		printf("convolutionSeparableSharedUnrollGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
			(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

		checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_Kernel));
		checkCudaErrors(cudaFree(d_Output));
		checkCudaErrors(cudaFree(d_Input));
}

void seperableSharedKernelMul(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));
	
	setKernel(h_Kernel);

	dim3 gridSize(isDividedUp(imageH, 16), isDividedUp(imageW, 16));
	dim3 blockSize(16, 16);

	printf("Running GPU naive separable convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionSeparableRowSharedMul(gridSize, blockSize, d_Input, d_Buffer, imageW, imageH);
		convolutionSeparableColumnSharedMul(gridSize, blockSize, d_Buffer, d_Output, imageW, imageH);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableSharedMulGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void seperableSharedKernelTile(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	setKernel(h_Kernel);

	dim3 gridSizeRow(isDividedUp(imageW, 8 * 2), isDividedUp(imageH, 16));
	dim3 blockSizeRow(8, 16);

	dim3 gridSizeCol(isDividedUp(imageW, 16), isDividedUp(imageH, 8 * 2));
	dim3 blockSizeCol(16, 8);

	printf("Running GPU naive separable convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionSeparableRowSharedTile(gridSizeRow, blockSizeRow, d_Input, d_Buffer, imageW, imageH);
		convolutionSeparableColumnSharedTile(gridSizeCol, blockSizeCol, d_Buffer, d_Output, imageW, imageH);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableSharedTileGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
}

void seperableSharedKernelTileCoales(
	float *d_Input,
	float *d_Output,
	float *d_Buffer,
	float *d_Kernel,
	float *h_Kernel,
	float *h_Input,
	float *h_OutputGPU,
	const int imageW,
	const int imageH,
	const int iterations,
	StopWatchInterface *hTimer)
{
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

	setKernel(h_Kernel);

	dim3 gridSizeRow(isDividedUp(imageW, 8 * 2), isDividedUp(imageH, 16));
	dim3 blockSizeRow(8, 16);

	dim3 gridSizeCol(isDividedUp(imageW, 16), isDividedUp(imageH, 8 * 2));
	dim3 blockSizeCol(16, 8);

	printf("Running GPU naive separable convolution (%u identical iterations).\n", iterations);
	for (int i = -1; i < iterations; i++)
	{
		
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		convolutionSeparableRowSharedTileCoales(gridSizeRow, blockSizeRow, d_Input, d_Buffer, imageW, imageH);
		convolutionSeparableColumnSharedTileCoales(gridSizeCol, blockSizeCol, d_Buffer, d_Output, imageW, imageH);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double cpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
	printf("convolutionSeparableSharedTileCoalesGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n\n",
		(1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Kernel));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
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
