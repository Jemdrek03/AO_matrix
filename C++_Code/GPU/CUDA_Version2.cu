#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define N 2048
#define blocksize 16
#define timeNumber 1


// !!!!!!! This one kinda sucks !!!!!!!


uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}


// !! See full comments in the CUDA_Version1 !!
float *A, *B, *C;


__global__
// ONLY for square matrices for now!!!
void threading(const float *A, const float *B, float *C)
{
    int tileId    = blockIdx.x * blockDim.x + threadIdx.x; // which tile this thread owns
    int startRow  = tileId * blocksize;
    if (startRow >= N) return;

    int yEnd = startRow + blocksize;
    if (yEnd > N) yEnd = N;

    for (int y = startRow; y < yEnd; ++y) {
        for (int x = 0; x < N; ++x) {
            float acc = 0.f;
            for (int k = 0; k < N; ++k) {
                acc += A[y * N + k] * B[k * N + x];
            }
            C[y * N + x] = acc;
        }
    }

}


int main()
{

    // Memory allocation
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y * N + x] = (y + x) * 0.001f;
            B[y * N + x] = (y - x) * 0.002f;
        }

    for(int times = 0; times < timeNumber; times++) {
        // Start to count the time needed
        uint64_t start = nanos();

        int tileRows = blocksize;            // 16
        int threadsPerBlock = 1;           // try 128/256/512 and benchmark
        int numTiles = (N + tileRows - 1) / tileRows;            // ceil(N / tileRows)
        int numBlocks = (numTiles + threadsPerBlock - 1) / threadsPerBlock;

        threading<<<numBlocks, threadsPerBlock>>>(A, B, C);
        cudaDeviceSynchronize();





        // Finalize time counting
        uint64_t end = nanos();
        double gflop = (N * N * 2.0 * N) * 1e-9;
        double s = (end - start) * 1e-9;
        std::cout << "GFLOPS " << gflop / s<<std::endl;
    }

    return 0;
}