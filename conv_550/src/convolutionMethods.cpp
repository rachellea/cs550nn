#include "convolutionMethods.h"

int isDividedUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

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
    printf("convolutionSeparableCPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
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


    checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

    setConvolutionKernel(h_Kernel);
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


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
    printf("convolutionSeparableGPU_NVIDIA, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
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
    printf("convolutionFullNaiveGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
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
    printf("convolutionSeparableNaiveGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
        (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Buffer));
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
    printf("convolutionSeparableSharedGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
        (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Buffer));
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
        checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_Kernel, KERNEL_LENGTH * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice));


        setKernel(h_Kernel);

        dim3 gridSize(isDividedUp(imageH, 16), isDividedUp(imageW, 16));
        dim3 blockSize(16, 16);


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
        printf("convolutionSeparableSharedUnrollGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
            (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_Kernel));
        checkCudaErrors(cudaFree(d_Output));
        checkCudaErrors(cudaFree(d_Input));
        checkCudaErrors(cudaFree(d_Buffer));
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
    printf("convolutionSeparableSharedMulGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
        (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Buffer));
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
    printf("convolutionSeparableSharedTileGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
        (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Buffer));
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
    printf("convolutionSeparableSharedTileCoalesGPU, Throughput = %.4f MPixels/sec, Time = %.9f s, Size = %u Pixels\n",
        (1.0e-6 * (double)(imageW * imageH) / cpuTime), cpuTime, (imageW * imageH));

    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Buffer));
}
