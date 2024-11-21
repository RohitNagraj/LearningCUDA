// Execution time: 45ms on RTX3090

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 16
#define CEIL_DIV(x, y) ((x + y - 1) / y)

__global__ void matmul(int *a, int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n)
    {
        int i, res = 0;
        for (i = 0; i < n; i++)
        {
            res += a[idx * n + i] * b[i * n + idy];
        }
        if (n * idx + idy >= n * n)
        {
            printf("IDX: %d, IDY: %d", idx, idy);
        }
        c[n * idx + idy] = res;
    }
}

int main()
{
    int **a, **b, **c, i, j;
    int *a_d, *b_d, *c_d;
    int *a_h, *b_h, *c_h;

    struct timespec start, stop; 
    double time;

    a = (int **)malloc(N * sizeof(int *));
    b = (int **)malloc(N * sizeof(int *));
    c = (int **)malloc(N * sizeof(int *));

    for (i = 0; i < N; i++)
    {
        a[i] = (int *)malloc(sizeof(int) * N);
        b[i] = (int *)malloc(sizeof(int) * N);
        c[i] = (int *)malloc(sizeof(int) * N);
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1;
            b[i][j] = 2;
            c[i][j] = 0;
        }

    // Flattening the 2d array since CUDA expects continuous memory allocation. I know it's redundent,
    // but since the question asks to start with a matrix, I started with a 2D array and now flattening it.
    a_h = (int *)malloc(sizeof(int) * N * N);
    b_h = (int *)malloc(sizeof(int) * N * N);
    c_h = (int *)malloc(sizeof(int) * N * N);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a_h[i * N + j] = a[i][j];
            b_h[i * N + j] = b[i][j];
            c_h[i * N + j] = c[i][j];
        }
    }

    cudaMalloc((void **)&a_d, N * N * sizeof(int));
    cudaMalloc((void **)&b_d, N * N * sizeof(int));
    cudaMalloc((void **)&c_d, N * N * sizeof(int));

    cudaMemcpy(a_d, a_h, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    matmul<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N);

    cudaMemcpy(c_h, c_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %f ms\n", time*1e3);

    printf("Value of C[451][451] = %d\n", c_h[N * 451 + 451]);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for (int i = 0; i < N; i++)
    {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    free(a_h);
    free(b_h);
    free(c_h);
}