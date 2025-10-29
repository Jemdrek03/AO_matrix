#include <cstdint>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 2048
#define timeNumber 1




uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

// Apparently on GPU you cant really use double pointers, so we will go with different indexing logic :3
float *A, *B, *C;

// ONLY for square matrices for now!!!
// We dont have to pass the rowNumber here, as its handled by the ThreadIDx
__global__
void threading(const float *A, const float *B, float *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // row index here
    for(int x = 0; x < N; x++)
    {
        float acc = 0;
        for(int k = 0; k < N; k++)
        {
            acc += A[row * N + k] * B[k* N + x];
        }
        C[row * N + x] = acc;
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

        // 1 thread per row
        dim3 threadsPerBlock(64);

        // In our case we could do just numBlocks = N / threadsPerBlock but the line below is a good practise
        // So I included it here
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);


        // Calling the kernel
        threading<<<numBlocks, threadsPerBlock>>>(A, B, C);


        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();


        // Finalize time counting
        uint64_t end = nanos();
        double gflop = (N * N * 2.0 * N) * 1e-9;
        double s = (end - start) * 1e-9;
        std::cout << "GFLOPS " << gflop / s<<std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}