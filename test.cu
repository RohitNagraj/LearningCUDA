#include<stdio.h>
#include<cuda.h>
#define BLOCK_SIZE 1024

__global__ void kernel(unsigned *vector, unsigned vectorsize)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < vectorsize)
        vector[id] = id;
}

int main(int argc, char *argv[])
{
    unsigned N = atoi(argv[1]);
    unsigned *vector, *hvector;
    printf("%s\n\n", argv[0]);
    cudaMalloc(&vector, N * sizeof(unsigned));

    hvector = (unsigned *)malloc(N * sizeof(unsigned));

    unsigned nblocks = ceil((float)N / BLOCK_SIZE);
    printf("nblocks: %d\n", nblocks);

    kernel<<<nblocks, BLOCK_SIZE>>>(vector, N);
    cudaMemcpy(hvector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
    for (unsigned ii=0; ii<5; ii++)
    {
        printf("%4d ", hvector[ii]);
    }
    return 0;

}