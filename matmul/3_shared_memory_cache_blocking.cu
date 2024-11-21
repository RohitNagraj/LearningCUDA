// This is a block matmul. Each thread block is responsible for calculating one block of C. C[i][j] += A[i][k] * B[k][j]. Iterate over k within the thread.

//Execution time: 19ms on RTX3090

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 16
#define CEIL_DIV(x, y) ((x + y - 1) / y)

__global__ void matmul(int *a, int *b, int *c, int n)
{
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if ((row < N)&&(col < N)) {
        int value_C = 0;
        for (int tileId=0; tileId<(N/(N/BLOCK_SIZE)); tileId++) {
            __shared__ int a_tile[N/(N/BLOCK_SIZE)][N/(N/BLOCK_SIZE)];
            __shared__ int b_tile[N/(N/BLOCK_SIZE)][N/(N/BLOCK_SIZE)];
            //loading the a and b tiles into shared memory
            a_tile[threadIdx.y][threadIdx.x] = a[row*N+(tileId*N/(N/BLOCK_SIZE)+threadIdx.x)];
            b_tile[threadIdx.y][threadIdx.x] = b[(tileId*N/(N/BLOCK_SIZE)+threadIdx.y)*N+col];
            __syncthreads();
            for (int k=0; k<N/(N/BLOCK_SIZE); k++) {
                value_C+=a_tile[threadIdx.y][k]*b_tile[k][threadIdx.x];
            }
            __syncthreads();
        }
        c[row*n+col] = value_C;
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