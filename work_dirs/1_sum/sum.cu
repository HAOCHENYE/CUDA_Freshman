# include <assert.h>
#include <stdio.h>
#include "freshman.h"


__global__ void sum_1dim_gpu(float *a, float *b, float *c)
{
    int x = threadIdx.x;
    c[x] = a[x] + b[x];
}

void sum_1dim_cpu(float *a, float *b, float *c, int n)
{
    for (int x = 0; x < n; x++)
        c[x] = a[x] + b[x];
}


int main()
{
    int num_ele = 10000;

    float *a_cpu = new float[num_ele];
    float *b_cpu = new float[num_ele];
    float *c_cpu = new float[num_ele];

    initialData(a_cpu, num_ele);
    initialData(b_cpu, num_ele);
    initialData(c_cpu, num_ele);

    TIMEIT(sum_1dim_cpu(a_cpu, b_cpu, c_cpu, num_ele), "sum_1dim_cpu");

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    float c_from_gpu[num_ele];
    cudaMalloc((float**)&a_gpu, num_ele * sizeof(float));
    cudaMalloc((float**)&b_gpu, num_ele * sizeof(float));
    cudaMalloc((float**)&c_gpu, num_ele * sizeof(float));
    
    cudaMemcpy(a_gpu, a_cpu, num_ele * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, num_ele * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu, c_cpu, num_ele * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(10000);
    dim3 grid(1);

    CUDATIMEIT(sum_1dim_gpu, grid, block, "sum_1dim_gpu", a_gpu, b_gpu, c_gpu);
    cudaMemcpy(c_from_gpu, c_gpu, num_ele * sizeof(float), cudaMemcpyDeviceToHost);

    checkResult(c_from_gpu, c_cpu, num_ele);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
