# include <assert.h>
#include <stdio.h>
#include "freshman.h"


__global__ void matrix_multiplt_gpu(float *a, float *b, float *c, uint const w, uint const h)
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y; 
    uint idx = w * y + x;
    for (int i = 0; i < w; i++)
    {
        c[idx] += a[w * y + i] * b[w * i + x];
    }
}

void matrix_multiply_cpu(float *a, float *b, float *c, uint const w, uint const h)
{
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            for (int i = 0; i < w; i++)
            {
                c[w * y + x] += a[w * y + i] * b[w * i + x];
            }
        }
    }
}


int main()
{
    uint w = 1000;
    uint h = 1000;
    uint num_ele = w * h;
    float *a_cpu = new float[num_ele];
    float *b_cpu = new float[num_ele];
    float *c_cpu = new float[num_ele];

    initialData(a_cpu, num_ele);
    initialData(b_cpu, num_ele);
    initialData(c_cpu, num_ele);

    TIMEIT(matrix_multiply_cpu(a_cpu, b_cpu, c_cpu, w, h), "matrix_multiply_cpu");

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

    dim3 block(100, 100);
    dim3 grid((w - 1) / block.x + 1, (h - 1) / block.y + 1);

    CUDATIMEIT(matrix_multiplt_gpu, grid, block, "matrix_multiplt_gpu", a_gpu, b_gpu, c_gpu, w, h);
    cudaMemcpy(c_from_gpu, c_gpu, num_ele * sizeof(float), cudaMemcpyDeviceToHost);

    checkResult(c_from_gpu, c_cpu, num_ele);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
