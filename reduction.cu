// Objective: Compute the sum of a large array, as fast as possible. Make use of blocks, shared memory, and atomics. Further, try using streams.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#define N pow(2, 27)
#define BLOCK_SIZE 256

__global__ void add(float *arr, float *sum)
{
    int local_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    __shared__ float smem_sum[BLOCK_SIZE];
    if (global_id < N)
    {

        smem_sum[local_id] = arr[global_id];
        __syncthreads();

        for (i = BLOCK_SIZE / 2; i > 0; i /= 2)
        {

            if (local_id < i)
            {
                smem_sum[local_id] += smem_sum[local_id + i];
            }

            __syncthreads();
        }

        if (local_id == 0)
        {
            atomicAdd(sum, smem_sum[local_id]);
        }
    }
}

int main()
{
    float *h_arr, *d_arr, *h_sum, *d_sum;
    int i;

    // We'll directly assigned pinned memory to reduce
    cudaMallocHost((void **)&h_arr, N * sizeof(float));
    cudaMallocHost((void **)&h_sum, sizeof(float));
    cudaMalloc((void **)&d_arr, N * sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));

    h_sum[0] = 0.0;

    for (i = 0; i < N; i++)
    {
        h_arr[i] = (float)rand() / RAND_MAX;
    }

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    cudaMemcpy(d_arr, h_arr, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, h_sum, sizeof(float), cudaMemcpyHostToDevice);

    add<<<grid, block>>>(d_arr, d_sum);

    cudaMemcpy(h_arr, d_arr, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // float sum_cpu = 0.0;
    // for (i = 0; i < N; i++)
    // {
    //     sum_cpu += h_arr[i];
    // }

    printf("The GPU sum is %f\n", h_sum[0]);
    // printf("The CPU sum is %f\n", sum_cpu);

    return 0;
}