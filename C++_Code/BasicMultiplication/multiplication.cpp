#include <cstdint>
#include <iostream>
#include <ctime>
#include <chrono>

#define N 2048

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

alignas(64) float A[N][N];
alignas(64) float B[N][N];
alignas(64) float C[N][N];

int main()
{

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y][x] = (y + x) * 0.001f;
            B[y][x] = (y - x) * 0.002f;
        }

    uint64_t start = nanos();
    for(int y = 0; y < N;y++){
        for(int x = 0; x < N; x++){
            float acc = 0;
            for(int k = 0; k < N; k++){
                acc += A[y][k] * B[k][x];
            }
            C[y][x] = acc;
        }
    }
    uint64_t end = nanos();
    double gflop = (N*N*2.0*N)*1e-9;
    double s = (end-start)*1e-9;
    std::cout<<"GFLOPS "<<gflop/s;
    return 0;
}