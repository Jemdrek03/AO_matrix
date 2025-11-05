#include <cstdint>
#include <iostream>
#include <ctime>
#include <chrono>

#define N 1024
#define timeNumber 10.0

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
    uint64_t startone = nanos();

    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTime = 0.0;

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y][x] = (y + x) * 0.001f;
            B[y][x] = (y - x) * 0.002f;
        }


    for(int t = 0; t < timeNumber; t++) {
        uint64_t start = nanos();
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float acc = 0;
                for (int k = 0; k < N; k++) {
                    acc += A[y][k] * B[k][x];
                }
                C[y][x] = acc;
            }
        }
        uint64_t end = nanos();
        double s = (end - start) * 1e-9;
        sumTime += s;
    }
    uint64_t endone = nanos();
    //std::cout << " Time " << (endone - startone) * 1e-9 << " seconds" <<std::endl;
    std::cout << " Average latency " << (sumTime / timeNumber)<< " seconds " <<std::endl;
    std::cout << " Average GFLOPS " << gflop / (sumTime / timeNumber)<<std::endl;
    return 0;
}