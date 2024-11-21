#include<stdio.h>
#include<cuda.h>
#define N 1024

__global__ void assign_zero(int *a)
{
    int id = threadIdx.x;
    a[id] = 0;
}

int main()
{
    int a[N], *a_d;
    for(int i=0; i<5; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
    cudaMalloc((void**) &a_d, N*sizeof(int));

    assign_zero<<<1, N>>>(a_d);

    cudaMemcpy(a, a_d, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<5; i++)
    {
        printf("%d ", a[i]);
    }
    return 0;

}