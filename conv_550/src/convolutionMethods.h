#ifndef CONVOLUTION_METHODS_H
#define CONVOLUTION_METHODS_H

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolution.h"
#include "convolutionSeparable_common.h"
#include <iostream>

using namespace std;

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


#endif
