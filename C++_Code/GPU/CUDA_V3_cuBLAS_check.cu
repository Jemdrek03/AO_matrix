#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 2048
#define blocksize 32
#define timeNumber 100.0

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

__global__
void threading(const float *A, const float *B, float *C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N)
        return;

    float acc = 0.f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];

    C[row * N + col] = acc;
}

int main()
{
    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTimeCPU = 0.0;
    double sumTimeGPU = 0.0;

    size_t bytes = size_t(N) * size_t(N) * sizeof(float);

    float *A, *B, *C, *C_cublas;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    cudaMallocManaged(&C_cublas, bytes);

    std::mt19937 rng(420);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N*N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(A, bytes, dev);
    cudaMemPrefetchAsync(B, bytes, dev);
    cudaMemPrefetchAsync(C, bytes, dev);
    cudaMemPrefetchAsync(C_cublas, bytes, dev);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    dim3 block(blocksize, blocksize);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    threading<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double maxErr = 0.0;
    for (int times = 0; times < timeNumber; times++)
    {
        uint64_t startCPU = nanos();
        cudaEventRecord(start);

        threading<<<grid, block>>>(A, B, C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        uint64_t endCPU = nanos();

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sumTimeCPU += (endCPU - startCPU) * 1e-9;
        sumTimeGPU += ms * 1e-3;

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    B, N, 
                    A, N,
                    &beta,
                    C_cublas, N);

        cudaDeviceSynchronize();

        for (int i = 0; i < N*N; i++) {
            double diff = fabs(C[i] - C_cublas[i]);
            if (diff > maxErr) maxErr = diff;
        }
    }

    std::cout << "Max error vs cuBLAS = " << maxErr << "\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n---------------------------------------------\n";
    std::cout << "Average latency (CPU chrono): " << (sumTimeCPU / timeNumber) << " s\n";
    std::cout << "Average latency (GPU event):  " << (sumTimeGPU / timeNumber) << " s\n";
    std::cout << "GFLOPS (CPU chrono): " << gflop / (sumTimeCPU / timeNumber) << "\n";
    std::cout << "GFLOPS (GPU event):  " << gflop / (sumTimeGPU / timeNumber) << "\n";

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_cublas);

    return 0;
}
