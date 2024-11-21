#include<stdio.h>
#include<stdlib.h>

#define size 1024

__global__ void add(int N, size_t pitch, int ***a, int **b, int *c)
{
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int res = a[idx][idy][idz] + b[idx][idy] + c[idx];
}

int main()
{
    int ***a, **b, *c, i, j;
    size_t pitch;
    int **slices_d;

    a = (int ***)malloc(sizeof(int) * size);
    b = (int **)malloc(sizeof(int) * size);
    c = (int *)malloc(sizeof(int) * size);

    for (i=0; i<size; i++)
    {
        a[i] = (int **)malloc(sizeof(int) * size);
        b[i] = (int *)malloc(sizeof(int) * size);
    }

    for (i=0; i<size;i++)
    {
        for (j=0; j< size; j++)
        {
            a[i][j] = (int *)malloc(sizeof(int) * size);
        }
    }

    cudaMalloc(&slices_d, size * sizeof(int *));
    for (int i = 0; i < size; ++i) {
        int *slice_d;
        cudaMallocPitch(&slice_d, &pitch, size * sizeof(int), size);
        cudaMemcpy(&slices_d[i], &slice_d, sizeof(int *), cudaMemcpyHostToDevice);
    }


    dim3 blockSize(4, 4, 4);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y, (size + blockSize.z - 1) / blockSize.z);

    add<<<gridSize, blockSize>>>(size, pitch, a, b, c);

    for (int i = 0; i < size; ++i) {
        int *slice_d;
        cudaMemcpy(&slice_d, &slices_d[i], sizeof(int *), cudaMemcpyDeviceToHost);
        cudaFree(slice_d);
    }
    cudaFree(slices_d);


    printf("Done");

}