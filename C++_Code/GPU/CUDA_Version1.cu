#include <cstdint>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define N 2048
#define timeNumber 10.0




uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

// Apparently on GPU you cant really use double pointers, so we will go with different indexing logic :3
float *A, *B, *C;

// ONLY for square matrices for now!!!
// We don't have to pass the rowNumber here, as it's handled by the ThreadIDx
__global__
void threading(const float *A, const float *B, float *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // row index here
    if( row >= N)
        return;

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
    uint64_t startone = nanos();
    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTime = 0.0;
    double sumTimeCPU = 0.0;
    double sumTimeGPU = 0.0;



    size_t bytes = size_t(N) * size_t(N) * sizeof(float);

    // Memory allocation (Unified Memory region)
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y * N + x] = (y + x) * 0.001f;
            B[y * N + x] = (y - x) * 0.002f;
        }

    // Set device used in computation
    int dev = 0;
    cudaGetDevice(&dev);

    // We need to specify a gpu
    cudaMemLocation loc = {};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;

    // Advise the Unified Memory subsystem about the usage pattern for the memory range starting at devPtr with a size of count bytes
    // Non-needed in problem like this, but it's a good thing to know for future implementation, so I added it here
    cudaMemAdvise(A, bytes, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemAdvise(B, bytes, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemAdvise(C, bytes, cudaMemAdviseSetPreferredLocation, loc);


    // For now the only option is 0
    unsigned int flags = 0;
    // On which stream we are working on (here we need to use the PCI, and since we have only one we need to do wait for the last prefetch to complete
    cudaStream_t s = 0;

    // Prefetches memory to the specified destination location
    cudaMemPrefetchAsync(A, bytes, loc, flags, s);
    cudaMemPrefetchAsync(B, bytes, loc, flags, s);
    cudaMemPrefetchAsync(C, bytes, loc, flags, s);
    cudaStreamSynchronize(s);



    // 1 thread per row
    dim3 threadsPerBlock(64);

    // In our case we could do just numBlocks = N / threadsPerBlock but the line below is a good practise
    // So I included it here
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Warmup
    for( int t = 0; t < 2; t++)
    {
        threading<<<numBlocks, threadsPerBlock>>>(A, B, C);
    }
    cudaDeviceSynchronize();

    // CUDA event setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    for(int times = 0; times < timeNumber; times++) {
        // Start to count the time needed
        uint64_t startCPU = nanos();

        // GPU timer start
        cudaEventRecord(start);

        // Calling the kernel
        threading<<<numBlocks, threadsPerBlock>>>(A, B, C);

        // GPU timer stop
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();


        // Finalize time counting
        uint64_t endCPU = nanos();


        // GPU elapsed time in ms
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double sGPU = ms / 1000.0;
        double sCPU = (endCPU - startCPU) * 1e-9;
        sumTimeCPU += sCPU;
        sumTimeGPU += sGPU;
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    uint64_t endone = nanos();


    //std::cout << " Total time " << (endone - startone) * 1e-9 << " seconds" <<std::endl;
    std::cout << "Average latency (CPU chrono): " << (sumTimeCPU / timeNumber) << " s\n";
    std::cout << "Average latency (GPU event):  " << (sumTimeGPU / timeNumber) << " s\n";
    std::cout << "GFLOPS (CPU chrono): " << gflop / (sumTimeCPU / timeNumber) << "\n";
    std::cout << "GFLOPS (GPU event):  " << gflop / (sumTimeGPU / timeNumber) << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}