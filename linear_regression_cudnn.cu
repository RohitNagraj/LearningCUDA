#include <iostream>
#include <vector>
#include <cudnn.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_CUDNN(call)                                                      \
    {                                                                          \
        cudnnStatus_t status = call;                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

const int NUM_POINTS = 100;
const int NUM_EPOCHS = 1000;
const float LEARNING_RATE = 0.01;

// Kernel for initializing data
__global__ void init_data(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = float(idx) / n;
        y[idx] = 3.0f * x[idx] + 2.0f; // y = 3x + 2
    }
}

// Kernel for forward pass
__global__ void forward(float *x, float *w, float *b, float *y_pred, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y_pred[idx] = x[idx] * (*w) + (*b);
    }
}

// Kernel for calculating loss and gradients
__global__ void compute_loss_and_gradients(float *x, float *y, float *y_pred,
                                           float *w, float *b, float *loss,
                                           float *dw, float *db, int n) {
    __shared__ float loss_sum[256];
    __shared__ float dw_sum[256];
    __shared__ float db_sum[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float local_loss = 0;
    float local_dw = 0;
    float local_db = 0;

    if (idx < n) {
        float diff = y_pred[idx] - y[idx];
        local_loss = diff * diff;
        local_dw = 2 * diff * x[idx];
        local_db = 2 * diff;
    }

    loss_sum[tid] = local_loss;
    dw_sum[tid] = local_dw;
    db_sum[tid] = local_db;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            loss_sum[tid] += loss_sum[tid + stride];
            dw_sum[tid] += dw_sum[tid + stride];
            db_sum[tid] += db_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, loss_sum[0] / n);
        atomicAdd(dw, dw_sum[0] / n);
        atomicAdd(db, db_sum[0] / n);
    }
}

// Kernel for updating weights
__global__ void update_weights(float *w, float *b, float dw, float db) {
    *w -= LEARNING_RATE * dw;
    *b -= LEARNING_RATE * db;
}

int main() {
    // Allocate memory on device
    float *d_x, *d_y, *d_y_pred, *d_loss, *d_dw, *d_db, *d_w, *d_b;
    CHECK_CUDA(cudaMalloc(&d_x, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_pred, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dw, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float)));

    // Initialize data and weights
    init_data<<<(NUM_POINTS + 255) / 256, 256>>>(d_x, d_y, NUM_POINTS);
    CHECK_CUDA(cudaMemset(d_w, 0, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b, 0, sizeof(float)));

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Forward pass
        forward<<<(NUM_POINTS + 255) / 256, 256>>>(d_x, d_w, d_b, d_y_pred, NUM_POINTS);

        // Zero out loss and gradients
        CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dw, 0, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_db, 0, sizeof(float)));

        // Compute loss and gradients
        compute_loss_and_gradients<<<(NUM_POINTS + 255) / 256, 256>>>(
            d_x, d_y, d_y_pred, d_w, d_b, d_loss, d_dw, d_db, NUM_POINTS);

        // Update weights
        float h_dw, h_db;
        CHECK_CUDA(cudaMemcpy(&h_dw, d_dw, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_db, d_db, sizeof(float), cudaMemcpyDeviceToHost));

        update_weights<<<1, 1>>>(d_w, d_b, h_dw, h_db);

        // Print loss
        if (epoch % 100 == 0) {
            float h_loss;
            CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "Epoch " << epoch << " Loss: " << h_loss << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_pred);
    cudaFree(d_loss);
    cudaFree(d_dw);
    cudaFree(d_db);
    cudaFree(d_w);
    cudaFree(d_b);

    return 0;
}
