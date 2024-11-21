#include<stdio.h>
#include<cuda.h>
#define N 3
#define M 2


__global__ void kernel()
{
    int id = threadIdx.x * blockDim.y + threadIdx.y;
    printf("ID: %d, Thread IDx: %d, Thread IDy: %d, Block IDx: %d, BlockIDy: %d, ThreadDim.x: %d, ThreadDim.y: %d, ThreadDim.z: %d\n", id, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    dim3 block(M, N, 1);  
    kernel<<<1, block>>>();
    cudaDeviceSynchronize();
    return 0;
}